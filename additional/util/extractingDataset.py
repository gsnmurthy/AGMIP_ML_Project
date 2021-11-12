# /**
#  * @file extractingDataset.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Python script to extract all the compressed (.tar) dataset file to a specific place.
#  * @version 1.0
#  * @date 2021-09-01
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries.
import os
import time
import argparse
from tqdm import tqdm

# Command-line arguement parsing code.
parser = argparse.ArgumentParser()
parser.add_argument("--basepath", "-b", type=str, help="Enter the location of all the .tar files.")
parser.add_argument("--destination", "-d", type=str, help="Enter the destination to extract all files.")
args = parser.parse_args()

errorFiles = []

# Creating the destination folder if does not exist already.
if not os.path.exists(args.destination):
    os.makedirs(args.destination)

# Loop to go through every .tar file and extract it in the destination folder.
for file in tqdm(os.listdir(args.basepath)):
    if file.endswith('.tar') and 'mai' in file and 'pDSSAT' in file:
        try:
            #print("[INFO]. Extracting {}".format(os.path.join(args.basepath, file)))
            os.system("tar -xf {} --directory={}".format(os.path.join(args.basepath, file), args.destination))
        except:
            print("[ERROR]. Cannot extract {} file properly.".format(os.path.join(args.basepath, file)))
            errorFiles.append(os.path.join(args.basepath, file))
            print("[INFO]. Process will continue shortly.")
            time.sleep(5)

# Checking if encountered error in any .tar file.
if len(errorFiles) == 0:
    print("[FINISH]. Successfully extracted all the files.")
else:
    print("[INCOMPLETE][ERROR]. Some Files are not extracted properly.")
    print("[INCOMPLETE][ERROR]. List : ", errorFiles)
