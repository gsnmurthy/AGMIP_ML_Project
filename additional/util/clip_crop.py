# /**
#  * @file clipCropData.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Clip the crop netCDF (.nc4) file using shape (.shp), and save them.
#  * @version 1.0
#  * @date 2021-09-29
#  * @copyright Copyright (c) 2021
#  */
import os
import argparse
import xarray
import rioxarray
import geopandas
import rasterio
from shapely.geometry import mapping
from tqdm import tqdm

# Arguement parser for command-line interface.
parser = argparse.ArgumentParser()
parser.add_argument("--basepath", "-b", type=str, help="Enter the location of unclipped dataset the .nc4 files.")
parser.add_argument("--destination", "-d", type=str, help="Enter the destination to save clipped .nc4 files.", default="./clipped_files")
parser.add_argument("--shapefile", "-s", type=str, help="Enter the location of shape file.")
args = parser.parse_args()

if not os.path.exists(args.destination):
    """ Checks if the destination folder exists and if not then creates the
        directory to save the clipped files.
    """
    print("[INFO]. Destination path did not exists. Creating the folder for clipped files.")
    os.makedirs(args.destination)
    os.makedirs(os.path.join(args.destination, 'plant-day'))
    os.makedirs(os.path.join(args.destination, 'maty-day'))
    os.makedirs(os.path.join(args.destination, 'yield'))

if not os.path.exists(args.shapefile):
    """Checks if the shapefile exists at the given location or not.
    """
    print("[ERROR]. Shape file does not exists. Aborting the Procedure.")
    exit()

print("[INFO]. Loading the shape file for clipping the data.")
geodf = geopandas.read_file(args.shapefile)

subDirs = ['plant-day', 'maty-day', 'yield']
errorFiles = []

for subDir in subDirs:
    """Loop to go through each sub-directory for procedure.
    """
    totalFiles = len(os.listdir(os.path.join(args.basepath, subDir)))
    print("[INFO]. Clipping files of Directory : ", subDir)

    for file in tqdm(os.listdir(os.path.join(args.basepath, subDir))):
        """Loop to fo through each file in the sub-directory, clipping, and
           saving them at the destination folder.
        """
        xds = rioxarray.open_rasterio(os.path.join(args.basepath, subDir, file), decode_times=False)

        # Setting the Parameters.
        xds = xds.rename(x='lon', y='lat')
        xds = xds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        xds = xds.rio.write_crs("EPSG:4326", inplace=True)
        # print(xds)
        try:
            # Clipping the file.
            clipped = xds.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)
            clipped = clipped.rename(x='lon', y='lat')
            clipped = clipped.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)

            # Saving the clipped file at the location.
            clipped.to_netcdf("{}".format(os.path.join(args.destination, subDir, file)))

            # Closing the file.
            xarray.Dataset.close(xds)

            # Deleting the file from the memory.
            del xds, clipped
        except:
            errorFiles.append(file)

print("\n[FINISH]. The procedure has finished. Displaying the error files : {}".format(errorFiles))
