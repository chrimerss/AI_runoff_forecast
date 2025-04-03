import ee
import os
import requests
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import calendar
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("era5_download.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
# You need to authenticate first: earthengine authenticate
ee.Authenticate()

ee.Initialize(project='ee-chrimerss')

def get_monthly_date_ranges(start_date, end_date):
    """
    Generate a list of monthly date ranges between start_date and end_date.
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
    list: List of (start_date, end_date) tuples for each month
    """
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    date_ranges = []
    current = start
    
    while current < end:
        # Last day of current month
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = datetime.datetime(current.year, current.month, last_day)
        
        # If month_end is beyond the overall end date, use end date instead
        if month_end > end:
            month_end = end
        
        month_start_str = current.strftime('%Y-%m-%d')
        month_end_str = month_end.strftime('%Y-%m-%d')
        
        date_ranges.append((month_start_str, month_end_str))
        
        # Move to first day of next month
        current = (month_end + datetime.timedelta(days=1))
    
    return date_ranges

def download_era5_land_monthly(start_date, end_date, output_dir, retry_max=3):
    """
    Download ERA5-Land surface runoff data from Google Earth Engine for a specific month.
    Uses getDownloadURL for direct downloading.
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    output_dir (str): Directory to save the downloaded files
    retry_max (int): Maximum number of retry attempts
    
    Returns:
    list: List of downloaded file paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert dates to ee.Date objects
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    # Get ERA5-Land collection
    era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    
    # Filter by date and select surface runoff band
    filtered = era5_land.filterDate(start, end).select('surface_runoff')
    
    # This month's period identifier for filenames
    period_str = f"{start_date}_to_{end_date}"
    logger.info(f"Processing period: {period_str}")
    
    # Get list of image times
    time_list = filtered.aggregate_array('system:time_start').getInfo()
    if not time_list:
        logger.warning(f"No data found for period {period_str}")
        return []
    
    # Count number of images
    count = len(time_list)
    logger.info(f"Found {count} images to download")
    
    # List to store downloaded files
    downloaded_files = []
    
    # Download in smaller batches to avoid timeout/memory issues
    batch_size = 24  # 24 hours = 1 day at a time
    
    for i in range(0, count, batch_size):
        # Get the current batch indices
        batch_end = min(i + batch_size, count)
        batch_times = time_list[i:batch_end]
        batch_images = ee.ImageCollection.fromImages([
            filtered.filter(ee.Filter.eq('system:time_start', time)).first()
            for time in batch_times
        ])
        
        # Progress indicator
        batch_str = f"Batch {i//batch_size + 1}/{(count+batch_size-1)//batch_size}: images {i+1}-{batch_end}"
        logger.info(f"Processing {batch_str}")
        
        # Create a temporary collection of just this batch
        batch_collection = ee.ImageCollection(batch_images)
        
        # Generate download URL with getDownloadURL
        download_url = None
        retry_count = 0
        
        while retry_count < retry_max:
            try:
                # We're using getDownloadURL on the image collection
                download_url = batch_collection.getDownloadURL({
                    'scale': 10000,  # ~0.1 deg resolution
                    'crs': 'EPSG:4326',
                    'region': ee.Geometry.Rectangle([-180, -90, 180, 90]),
                    'format': 'ZIPPED_GEO_TIFF'
                })
                break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error generating download URL (attempt {retry_count}/{retry_max}): {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
        
        if download_url is None:
            logger.error(f"Failed to generate download URL for {batch_str} after {retry_max} attempts")
            continue
        
        # Download the file
        batch_file = f"{output_dir}/era5_land_runoff_{period_str}_batch_{i//batch_size}.tif"
        retry_count = 0
        
        while retry_count < retry_max:
            try:
                logger.info(f"Downloading batch to {batch_file}")
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                with open(batch_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_files.append(batch_file)
                logger.info(f"Successfully downloaded {batch_file}")
                break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error downloading (attempt {retry_count}/{retry_max}): {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
        
        if retry_count >= retry_max:
            logger.error(f"Failed to download {batch_str} after {retry_max} attempts")
    
    return downloaded_files

def upsample_and_save_to_zarr(input_files, zarr_store, first_month=False):
    """
    Upsample ERA5-Land data and save to zarr store.
    If first_month is True, create a new zarr store, otherwise append.
    
    Parameters:
    input_files (list): List of input file paths
    zarr_store (str): Path to zarr store
    first_month (bool): Whether this is the first month (create new store)
    
    Returns:
    str: Path to zarr store
    """
    logger.info(f"Processing {len(input_files)} files for upsampling")
    
    # Define target grid (0.25 deg resolution)
    target_lats = np.arange(-90, 90.25, 0.25)
    target_lons = np.arange(-180, 180.25, 0.25)
    
    # Process each file
    datasets = []
    for file in tqdm(input_files, desc="Processing files"):
        try:
            ds = xr.open_dataset(file, engine='rasterio')
            
            # Extract timestamps from the file metadata
            # This will vary based on how Earth Engine structures its output
            # We'll use a placeholder approach here - you'll need to adapt based on actual output
            
            # Get the time from file name or metadata
            # Here assuming the file has metadata about the time
            time_str = ds.attrs.get('time_coverage_start', None)
            if time_str is None:
                # Try to extract from filename
                file_basename = os.path.basename(file)
                if "_batch_" in file_basename:
                    # Parse from complex filename pattern - adjust to match your actual naming convention
                    parts = file_basename.split('_')
                    date_parts = []
                    for part in parts:
                        if part.isdigit() and len(part) == 4:  # Year
                            date_parts.append(part)
                        elif part.isdigit() and len(part) in [1, 2]:  # Month or day
                            date_parts.append(part.zfill(2))
                    
                    if len(date_parts) >= 3:
                        time_str = '-'.join(date_parts[:3])
            
            if time_str:
                dt = datetime.datetime.fromisoformat(time_str)
            else:
                # If we can't determine the time, use a timestamp from the file's modification time
                file_stat = os.stat(file)
                dt = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            
            # Add time coordinate
            ds = ds.expand_dims(time=[dt])
            
            # Spatial upsampling: 0.1 deg to 0.25 deg
            # Create interpolator for this dataset
            orig_lats = ds.y.values
            orig_lons = ds.x.values
            
            interp = RegularGridInterpolator(
                (orig_lats, orig_lons),
                ds.band_1.values[0],  # First time slice
                bounds_error=False,
                fill_value=None
            )
            
            # Create meshgrid for target points
            lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
            points = np.column_stack((lat_grid.flatten(), lon_grid.flatten()))
            
            # Interpolate
            upsampled_values = interp(points).reshape(len(target_lats), len(target_lons))
            
            # Create upsampled dataset
            upsampled_ds = xr.Dataset(
                data_vars={
                    'runoff': (['time', 'latitude', 'longitude'], 
                             np.expand_dims(upsampled_values, axis=0))
                },
                coords={
                    'time': [dt],
                    'latitude': target_lats,
                    'longitude': target_lons
                }
            )
            
            datasets.append(upsampled_ds)
            
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
    
    if not datasets:
        logger.warning("No datasets were successfully processed")
        return zarr_store
    
    # Combine all datasets
    try:
        combined_ds = xr.concat(datasets, dim='time')
        
        # Select only data points at 00, 06, 12, 18 UTC for 6-hourly data
        six_hourly = combined_ds.sel(time=combined_ds.time.dt.hour.isin([0, 6, 12, 18]))
        
        # Sort by time
        six_hourly = six_hourly.sortby('time')
        
        # Save to zarr
        if first_month:
            logger.info(f"Creating new zarr store at {zarr_store}")
            encoding = {var: {'compressor': zarr.Blosc(cname='zstd', clevel=3)} 
                       for var in six_hourly.data_vars}
            six_hourly.to_zarr(zarr_store, mode='w', encoding=encoding)
        else:
            logger.info(f"Appending to existing zarr store at {zarr_store}")
            # Open existing zarr to get current times
            existing_ds = xr.open_zarr(zarr_store)
            existing_times = existing_ds.time.values
            
            # Filter out times that already exist in the store
            new_ds = six_hourly.sel(time=~six_hourly.time.isin(existing_times))
            
            if len(new_ds.time) > 0:
                new_ds.to_zarr(zarr_store, append_dim='time')
                logger.info(f"Appended {len(new_ds.time)} new time points to zarr store")
            else:
                logger.info("No new time points to append")
        
        logger.info(f"Successfully processed and saved data to {zarr_store}")
        
    except Exception as e:
        logger.error(f"Error combining datasets or saving to zarr: {e}")
    
    # Clean up input files to save space
    for file in input_files:
        try:
            os.remove(file)
        except Exception as e:
            logger.warning(f"Error removing temporary file {file}: {e}")
    
    return zarr_store

def main(start_date, end_date, temp_dir, zarr_store):
    """
    Main function to download ERA5-Land data monthly and save to zarr.
    
    Parameters:
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'
    temp_dir (str): Directory for temporary GeoTIFF files
    zarr_store (str): Path to zarr store
    """
    # Get monthly date ranges
    date_ranges = get_monthly_date_ranges(start_date, end_date)
    logger.info(f"Processing {len(date_ranges)} months from {start_date} to {end_date}")
    
    # Process each month
    for i, (month_start, month_end) in enumerate(date_ranges):
        logger.info(f"Processing month {i+1}/{len(date_ranges)}: {month_start} to {month_end}")
        
        # Download data for this month
        downloaded_files = download_era5_land_monthly(month_start, month_end, temp_dir)
        
        if not downloaded_files:
            logger.warning(f"No files downloaded for period {month_start} to {month_end}")
            continue
        
        # Upsample and save to zarr
        is_first_month = (i == 0)
        upsample_and_save_to_zarr(downloaded_files, zarr_store, first_month=is_first_month)
        
        # Brief pause to avoid hitting rate limits
        time.sleep(5)
    
    logger.info(f"Completed processing all months. Final zarr store saved at {zarr_store}")

if __name__ == "__main__":
    # Configuration
    START_DATE = '1959-01-01'
    END_DATE = '2023-01-10'
    TEMP_DIR = './era5_land_temp'
    ZARR_STORE = './era5_land_runoff_upsampled.zarr'
    
    # Run the process
    main(START_DATE, END_DATE, TEMP_DIR, ZARR_STORE)