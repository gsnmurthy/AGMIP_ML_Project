# /**
#  * @file inputFinalizer.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Read static input file in chunks, merge it with dynamic input file on ['time', 'lat', 'lon'] basis and save them.
#  * @version 3.0
#  * @date 2021-11-12
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries.
import os
import gc
import pandas as pd
from constants import *
import pyarrow.ipc as ipc


def read_feather_in_chunks(filepath):
    """Read feather file in chunks instead of all at once.

    Args:
        filepath (str): Path of final_input feather file.

    Yields:
        data_df [pandas.DataFrame]: return pandas Dataframe from the feather file.
    """
    with ipc.RecordBatchFileReader(filepath) as reader:
        for batch_index in range(reader.num_record_batches):
            if batch_index == 0:
                batch = reader.get_batch(batch_index).to_pandas(use_threads=True, timestamp_as_object=True, )
            else:
                new_batch = reader.get_batch(batch_index).to_pandas(use_threads=True, timestamp_as_object=True, )
                data_df = pd.concat([batch, new_batch], ignore_index=True)
                batch = data_df

            # Instead of taking just one batch with 65,000 rows (approx.),
            # we let the loop iterate over batches until it triggers the condition below.
            if (batch_index + 1) % 2 == 0:
                batch = pd.DataFrame()
                yield data_df


if __name__ == "__main__":
    """ Driver code which starts the whole process and saves the final input files in the corresponding directory. """
    file_count = 1

    # Reading dynamic input file.
    dynamic = pd.read_feather(os.path.join(input_dir, "dynamic.feather"))

    if not os.path.isdir(os.path.join(input_dir, final_inputs_dir)):
        os.makedirs(os.path.join(input_dir, final_inputs_dir))

    for batch in read_feather_in_chunks(os.path.join(input_dir, "static.feather")):
        # Formatting the time column in batch for merger.
        batch.time = batch.time.astype(int)

        # Mergine the dataframes on the ['time', 'lat', 'lon'] basis.
        final = pd.merge(batch, dynamic, on=['time', 'lat', 'lon'], how='inner')

        # Shuffling the dataframe before saving them.
        final = final.sample(frac=1)

        # Loop to calculate the tasmax, tasmin and precipitation_flux in the batch.
        # print("[INFO]. Calculating the Perturbations and additional inputs.")
        # for i in tqdm(range(len(final))):
        #     final.iloc[i, 24] += final.iloc[i, 10]
        #     final.iloc[i, 25] += final.iloc[i, 10]
        #     if final.iloc[i, 9] != np.inf:
        #         final.iloc[i, 26] = (1 + final.iloc[i, 9] / 100) * final.iloc[i, 26]
        # final = final.drop(columns=['index_x', 'index_y', 'spatial_ref', 'W', 'T'])

        final = final.reset_index()
        final.to_feather(os.path.join(input_dir, final_inputs_dir, "input_file_{}.feather".format(file_count)), compression='lz4')

        # Clearing the memory buffer.
        gc.collect()

        print("[INFO]. Batch {} has been processed and saved.".format(file_count))
        file_count += 1
