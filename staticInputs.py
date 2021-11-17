# /**
#  * @file staticInputs.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Open all dataset file, preprocess, & save the static input in feather file incrementally.
#  * @version 3.0
#  * @date 2021-11-12
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries.
import os
import gc
import glob
import time
import geopandas
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from constants import *
from shapely.geometry import mapping


def clipData(shape_file_path, plantData, matyData, yieldData):
    """Clip the netcdf data using shape file passed.

    Args:
        shape_file_path (str): Path to shape file for clipping the data.
        plantData (xarray.Dataset): nc4 Data of plant-file
        matyData (xarray.Dataset): nc4 Data of maturity-file
        yieldData (xarray.Dataset): nc4 Data of yield

    Returns:
        [xarray.Dataset]: Clipped nc4 data.
    """
    # Reading the shape file using geopandas.
    geodf = geopandas.read_file(shape_file_path)

    # Configuring the rio engine for clipping.
    plantData = plantData.rio.write_crs("EPSG:4326", inplace=True)
    matyData = matyData.rio.write_crs("EPSG:4326", inplace=True)
    yieldData = yieldData.rio.write_crs("EPSG:4326", inplace=True)

    # Clipping the data.
    plantData = plantData.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)
    matyData = matyData.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)
    yieldData = yieldData.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)

    # Renaming the cooridnates of the data.
    yieldData = yieldData.rename(x='lon', y='lat')
    yieldData = yieldData.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)

    return plantData, matyData, yieldData


def loadDatasets(yield_filePath, plant_filePath, maty_filePath, crop_name):
    """Load datasets, and unify them in a data frame with all essential features.

    Args:
        yield_filePath ([str]): Complete yield file path.
        plant_filePath ([str]): Complete plant-day file path.
        maty_filePath ([str]): Complete maty-day file path.
        crop_name ([str]): Name of crop.

    Returns:
        yieldDF([pd.DataFrame]): Pandas DataFrame with all features combined.
    """
    # Loading datasets files without decoding time variable.
    plantData = xr.open_dataset(plant_filePath, engine='rasterio', decode_times=False)
    matyData = xr.open_dataset(maty_filePath, engine='rasterio', decode_times=False)
    yieldData = xr.open_dataset(yield_filePath, engine='rasterio', decode_times=False)

    # Clipping the data.
    plantData, matyData, yieldData = clipData(os.path.join(shape_file_path, "4_states/4_states.shp"), plantData, matyData, yieldData)

    # Decoding time variable of the dataset using pandas.
    initialYear = 1979
    timeArray = initialYear + yieldData.variables['time'].values

    plantData['time'] = timeArray.astype(int)
    matyData['time'] = timeArray.astype(int)
    yieldData['time'] = timeArray.astype(int)

    # Converting dataset in DataFrame format for modification.
    plantDF = plantData['plant-day_{}'.format(crop_name)].to_dataframe()
    matyDF = matyData['maty-day_{}'.format(crop_name)].to_dataframe()
    yieldDF = yieldData['yield_{}'.format(crop_name)].to_dataframe()

    # Adding columns from plantDF and matyDF in yieldDF.
    yieldDF['plant-day'] = plantDF['plant-day_{}'.format(crop_name)]
    yieldDF['maturity-day'] = yieldDF['plant-day'] + matyDF['maty-day_{}'.format(crop_name)]
    yieldDF = yieldDF.rename(columns={'x': 'lon', 'y': 'lat'})

    # Deleting unnecessary variables to conserve space in the system.
    del plantDF, matyDF, plantData, matyData, yieldData
    return yieldDF


def surfaceFeatureExtractor(yieldDF, yield_filePath):
    """Extract surface featueres from individual netCDF file and combine it with yield DataFrame.

    Args:
        yieldDF (pandas.DataFrame): DataFrame with plant-day, maty-day and yield features.
        yield_filePath (str): Path of yield netCDF file.

    Returns:
        yieldDF (pandas.DataFrame): Return Dataframe with additional CWTN-A features.
    """
    # Splitting filename to get individual un-processed feature.
    splittedFilename = yield_filePath.split('/')[-1].split('_')
    raw_CO2, raw_W = splittedFilename[9], splittedFilename[11]
    raw_T, raw_N = splittedFilename[10], splittedFilename[12]
    raw_A = splittedFilename[13]

    # Processing individual feature.
    CO2, N, A = int(raw_CO2[1:]), int(raw_N[1:]), int(raw_A[1])
    T, W = raw_T.split('-'), raw_W.split('-')

    if len(T) > 1:
        T = int(T[-1])
    else:
        T = int(T[0][1:])

    if len(W) > 1:
        W = int(W[-1])
    else:
        W = W[0].strip()
        if len(W) > 3:
            W = np.inf
        else:
            W = int(W[1:])

    # print("[INFO]. CO2 : ", CO2)
    # print("[INFO].  W  : ", W)
    # print("[INFO].  T : ", T)
    # print("[INFO].  N : ", N)
    # print("[INFO].  A : ", A)

    # Adding surface features in yield DataFrame.
    yieldDF['CO2'] = CO2
    yieldDF['W'] = W
    yieldDF['T'] = T
    yieldDF['N'] = N
    yieldDF['A'] = A

    # Clearing the memory buffer and deleting the un-necessary variables.
    gc.collect()
    del raw_CO2, raw_A, raw_N, raw_T, raw_W, splittedFilename, CO2, W, T, N, A

    return yieldDF


def soilFeatureCombine(yieldDF, soil_file_path):
    """Access the HWSD soil v2.2 netcdf file and add soil features to the yield DataFrame.

    Args:
        yieldDF (pandas.DataFrame): yield Dataframe with surface features.
        soil_file_path (str): HWSD soil netCDF path.

    Returns:
        yieldDF (pandas.DataFrame): DataFrame with soil features.
    """
    # Reading HWSD file and converting it to dataframe.
    soilData = xr.open_dataset(soil_file_path)
    soilDF = soilData.to_dataframe().reset_index()

    # Dropping additional features from the file to conserve computation power.
    soilDF = soilDF.drop(columns=['mu_global', 'bulk_density', 'root_obstacles', 'impermeable_layer', 'ece', 'bs_soil', 'issoil'])

    # Dropping any row with null value.
    soilDF = soilDF.dropna(how='any')

    # Merging yield DF and soil DF on latitude and longitude basis.
    yieldDF = pd.merge(yieldDF, soilDF, on=['lat', 'lon'], how='inner')
    return yieldDF


if __name__ == "__main__":
    """
    Driver Code which saves the finalized static inputs dataset in .feather format.
    """
    # Initializing variables.
    # plant_dir = "./ggcmi/phase2_outputs/dataset/plant-day/"
    # maty_dir = "./ggcmi/phase2_outputs/dataset/maty-day/"
    # yield_dir = "./ggcmi/phase2_outputs/dataset/yield/"
    # soil_file_path = "./ggcmi/HWSD/HWSD_soil_data_on_cropland_v2.2.nc"

    count, total_files = 1, len(glob.glob(yield_dir + "*.nc4"))
    prevFile, DF = pd.DataFrame(), pd.DataFrame()

    # Loop to go through each file in plant-day, maty-day and yield folder.
    for filename in tqdm(glob.glob(yield_dir + "*.nc4")):

        yield_filePath = filename
        filenameList = yield_filePath.split('/')[-1].split('_')

        crop_name = filenameList[4]

        filenameList[3] = 'plant-day'
        plant_filePath = plant_dir + '_'.join(filenameList)

        filenameList[3] = 'maty-day'
        maty_filePath = maty_dir + '_'.join(filenameList)

        # Condition to check if the corresponding files of plant-day and maty-day exists.
        if not os.path.exists(plant_filePath) or not os.path.exists(maty_filePath):
            print("[ERROR]. File does not exists.")
            print("[ERROR]. Passing through this iteration without change.")
            count += 1
            time.sleep(3)
            pass

        # print("[INFO]. plant-day file Path  : ", plant_filePath)
        # print("[INFO]. maty-day file Path   : ", maty_filePath)
        # print("[INFO]. yield file Path      : ", yield_filePath)

        # print("\n\n\n\n[PHASE 1]. Starting Loading Datasets.")
        yieldDF = loadDatasets(yield_filePath, plant_filePath, maty_filePath, crop_name)
        # print("[PHASE 1]. Datasets Loaded Successfully.")

        # print("\n[PHASE 2]. Starting Surface Feature Extraction.")
        yieldDF = surfaceFeatureExtractor(yieldDF, yield_filePath)
        # print("[PHASE 2]. Surface Feature Extracted Successfully.")

        yieldDF = yieldDF.reset_index()
        yieldDF = yieldDF.dropna(how='any')

        # print("\n[PHASE 3]. Starting Soil Feature Extraction.")
        yieldDF = soilFeatureCombine(yieldDF, soil_file_path)
        # print("[PHASE 3]. Soil Feature Extracted Successfully.")

        # Changing the column format of the dataframe.
        # yieldDF.gravel = yieldDF.gravel.astype(int)
        # yieldDF.clay = yieldDF.clay.astype(int)
        # yieldDF.silt = yieldDF.silt.astype(int)
        # yieldDF.sand = yieldDF.sand.astype(int)
        # yieldDF.awc = yieldDF.awc.astype(int)
        # yieldDF.cec_soil = yieldDF.cec_soil.astype(int)
        # yieldDF.texture_class = yieldDF.texture_class.astype(int)
        # yieldDF.CO2 = yieldDF.CO2.astype(int)
        # yieldDF['plant-day'] = yieldDF['plant-day'].astype(int)
        # yieldDF['maturity-day'] = yieldDF['maturity-day'].astype(int)

        # Shuffling the dataframe.
        yieldDF = yieldDF.sample(frac=1)

        # Concatenating the finalized DF from each file.
        prevFile = pd.concat([prevFile, yieldDF], ignore_index=True)

        yieldDF["yield_{}".format(crop_name)] = np.round(yieldDF["yield_{}".format(crop_name)])
        count += 1

        # Saving the DF in feather format after every specified iteration or when finished.
        if count % 10 == 0 or count == total_files:
            if os.path.isfile(os.path.join(input_dir, 'static.feather')):
                DF = pd.read_feather(os.path.join(input_dir, 'static.feather'))
                DF = DF.sample(frac=1)
                DF = pd.concat([DF, prevFile], ignore_index=True)
                os.system("rm -rf {}".format(os.path.join(input_dir, 'static.feather')))
            else:
                DF = DF.reset_index()

            DF = DF.reset_index(drop=True)
            DF.to_feather(os.path.join(input_dir, 'static.feather'), compression='lz4')

            # Clearing the memory buffer, deleting un-necessary variables and resetting prevFile and DF.
            del prevFile, DF
            prevFile, DF = pd.DataFrame(), pd.DataFrame()
            gc.collect()
