# /**
#  * @file dynamicInputs.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Open all weather files, preprocess, & save the dynamic input in dataframe.
#  * @version 3.0
#  * @date 2021-11-12
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries.
import os
import gc
import geopandas
import pandas as pd
import xarray as xr
from constants import *
from shapely.geometry import mapping
from dask.diagnostics import ProgressBar


def clipData(shape_file_path, Data):
    """ Clip the .nc4 data using shape file passed.

    Args:
        shapeFile (str): Path to shape file for clipping the data.
        Data (xarray.Dataset): netCDF Data of AgMerra files.

    Returns:
        Data (xarray.Dataset): Clipped netCDF AgMerras data.
    """
    # Reading the shape file using geopandas.
    geodf = geopandas.read_file(shape_file_path)

    # Configuring the rio engine for clipping.
    Data = Data.rio.write_crs("EPSG:4326", inplace=True)

    # Clipping the data.
    Data = Data.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)

    # Renaming the cooridnates of the data.
    Data = Data.rename(x='lon', y='lat')
    Data = Data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    return Data


def loadDatasets(agmerra_files_path):
    """ Load the weather netcdf files.

    Args:
        agmerra_files_path (str): Path to directory which contains all of the weather nc4 files.

    Returns:
        DF ([pandas.DataFrame]): Weather Dataframe.
    """
    # Loading datasets files without decoding time variable.
    print("\n[PHASE 1]. Starting Loading Datasets.")
    Data = xr.open_mfdataset(os.path.join(agmerra_files_path, "*.nc4"), engine='rasterio', parallel=True, chunks=10)
    print("[PHASE 1]. Datasets Loaded Successfully.")

    # Clipping the data.
    print("\n[PHASE 2]. Starting Clipping.")
    Data = clipData(os.path.join(shape_file_path, '4_states/4_states.shp'), Data)
    print("[PHASE 2]. Clipped Successfully.")

    # Converting cftime to datetime format.
    Data['time'] = Data.indexes['time'].to_datetimeindex()

    # Converting dataset in DataFrame format for modification.
    print("\n[PHASE 3]. Conversion to Dataframe started.")
    with ProgressBar():
        DF = Data.to_dataframe()
    print("\n[PHASE 3]. Conversion to Dataframe completed.")

    # Deleting unnecessary variables to conserve space in the system.
    gc.collect()
    del Data
    return DF


if __name__ == "__main__":
    """
    Driver Code which saves the finalized dynamic inputs dataset in .feather format.
    """
    # Initializing variables.
    DF = loadDatasets(agmerra_files_path)

    # Pre-process the dynamic inputs Dataframe.
    DF = DF.dropna(how='any')
    DF = DF.reset_index()
    DF = DF.drop(columns=['spatial_ref'])

    # Grouping the dataframe in monthns for each pair of (lat, lon) and computing tasmax (max temperature),
    # tasmin (min temperature), pr (precipitation), and gdd (growing degree days).
    DF2 = pd.DataFrame(DF.groupby([pd.Grouper(key='time', freq='M'), 'lat', 'lon']).tasmax.max())
    DF2['tasmin'] = DF.groupby([pd.Grouper(key='time', freq='M'), 'lat', 'lon']).tasmin.min()
    DF2['pr'] = DF.groupby([pd.Grouper(key='time', freq='M'), 'lat', 'lon']).pr.sum()
    DF2['gdd'] = (DF2.tasmax + DF2.tasmin) / 2

    # Changing the format of time in the dataframe.
    DF2 = DF2.reset_index()
    DF2['time'] = pd.DatetimeIndex(DF2['time']).year
    DF2 = DF2.sample(frac=1)

    # Clearing the memory buffer and removing the unnecessary variables.
    gc.collect()
    del DF

    # Saving the dataframe in feather format with 'lz4' compression.
    DF2 = DF2.reset_index()
    DF2.to_feather(os.path.join(input_dir, "dynamic.feather"), compression='lz4')
