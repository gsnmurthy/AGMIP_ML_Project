# /**
#  * @file staticInput.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Open all dataset file, preprocess, & save the static input in csv.
#  * @version 2.1
#  * @date 2021-09-09
#  * @copyright Copyright (c) 2021
#  */
import os
import time
import numpy as np
import pandas as pd
import xarray as xr


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
    plantData = xr.open_dataset(plant_filePath, decode_times=False)
    matyData = xr.open_dataset(maty_filePath, decode_times=False)
    yieldData = xr.open_dataset(yield_filePath, decode_times=False)

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
    yieldDF['maturity-day'] = matyDF['maty-day_{}'.format(crop_name)]

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

    print("[INFO]. CO2 : ", CO2)
    print("[INFO].  W  : ", W)
    print("[INFO].  T : ", T)
    print("[INFO].  N : ", N)
    print("[INFO].  A : ", A)

    # Adding surface features in yield DataFrame.
    yieldDF['CO2'] = CO2
    yieldDF['W'] = W
    yieldDF['T'] = T
    yieldDF['N'] = N
    yieldDF['A'] = A

    del raw_CO2, raw_A, raw_N, raw_T, raw_W, splittedFilename, CO2, W, T, N, A
    return yieldDF


def soilFeatureCombine(yieldDF, soilFile):
    """Access the HWSD soil v2.2 netcdf file and add soil features to the yield DataFrame.

    Args:
        yieldDF (pandas.DataFrame): yield Dataframe with surface features.
        soilFile (str): HWSD soil netCDF path.

    Returns:
        yieldDF (pandas.DataFrame): DataFrame with soil features.
    """
    # Reading HWSD file and converting it to dataframe.
    soilData = xr.open_dataset(soilFile)
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
    Driver Code which puts every phase in motion and saves the finalized
    static inputs dataset in .csv format.
    """
    # Initializing variables.
    plantDir = "./ggcmi/phase2_outputs/dataset/plant-day/"
    matyDir = "./ggcmi/phase2_outputs/dataset/maty-day/"
    yieldDir = "./ggcmi/phase2_outputs/dataset/yield/"
    soilFile = "./ggcmi/HWSD/HWSD_soil_data_on_cropland_v2.2.nc"

    count = 1
    total_files = len(os.listdir(yieldDir))

    # Loop to go through each file in plant-day, maty-day and yield folder.
    for filename in os.listdir(yieldDir):
        yield_filePath = yieldDir + filename
        filenameList = yield_filePath.split('/')[-1].split('_')

        crop_name = filenameList[4]

        filenameList[3] = 'plant-day'
        plant_filePath = plantDir + '_'.join(filenameList)

        filenameList[3] = 'maty-day'
        maty_filePath = matyDir + '_'.join(filenameList)

        print("[INFO]. Plant Day File    : ", plant_filePath)
        print("[INFO]. Maturity Day File : ", maty_filePath)
        print("[INFO]. Yield Day File    : ", yield_filePath)

        # Condition to check if the corresponding files of plant-day and maty-day exists.
        if not os.path.exists(plant_filePath) or not os.path.exists(maty_filePath):
            print("[ERROR]. File does not exists.")
            print("[ERROR]. Passing through this iteration without change.")
            count += 1
            time.sleep(3)
            pass

        print("\n\n\n\n[PHASE 1]. Starting Loading Datasets.")
        yieldDF = loadDatasets(yield_filePath, plant_filePath, maty_filePath, crop_name)
        print("[PHASE 1]. Datasets Loaded Successfully.")

        print("\n[PHASE 2]. Starting Surface Feature Extraction.")
        yieldDF = surfaceFeatureExtractor(yieldDF, yield_filePath)
        print("[PHASE 2]. Surface Feature Extracted Successfully.")

        yieldDF = yieldDF.reset_index()
        yieldDF = yieldDF.dropna(how='any')

        print("\n[PHASE 3]. Starting Soil Feature Extraction.")
        yieldDF = soilFeatureCombine(yieldDF, soilFile)
        print("[PHASE 3]. Soil Feature Extracted Successfully.")

        # Concatenating the finalized DF from each file.
        if count == 1:
            prevFile = yieldDF
        else:
            yieldDF = pd.concat([prevFile, yieldDF], ignore_index=True)
            prevFile = yieldDF

        yieldDF["yield_{}".format(crop_name)] = np.round(yieldDF["yield_{}".format(crop_name)])

        # Saving the DF in CSV format after every specified iteration or when finished.
        if count % 10 == 0 or count == total_files:
            os.system("rm -rvf ./staticInput.csv")
            yieldDF.to_csv("./staticInput.csv", index=False)
            print("\n[INFO]. CSV File saved successfully.")

        print("\n[INFO]. {}/{} File Completed.".format(count, total_files))
        count += 1
