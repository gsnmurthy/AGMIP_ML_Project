# /**
#  * @file train.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief Train the neural network model to predict yield on crop outputs, soil and climate basis.
#  * @version 2.0
#  * @date 2021-11-12
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries.
import os
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyarrow.ipc as ipc
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import config
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from constants import *


def memory_growth_GPU():
    """Enable memory growth in GPU (if present) for training the model."""
    try:
        physicalDevices = config.experimental.list_physical_devices('GPU')
        config.experimental.set_memory_growth(physicalDevices[0], True)
    except:
        print("[ERR]. Could not enable the memory growth in GPU. Switching to CPU for training.")


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
    """  This is the driver code which initializes all the variable, trains the model and save the outputs. """

    # Calling the function to switch processing to GPU (if present).
    memory_growth_GPU()

    # Initializing variables.
    EPOCHS = 100
    LEARNING_RATE = 1e-9
    BATCH_SIZE = 128
    ES_PATIENCE = 3
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    SEQUENCE = 1
    flag = 1

    for input_file in os.listdir(os.path.join(input_dir, final_inputs_dir)):
        """ Loop to iterate through all the input files in the directory for training.
        """
        batch_num = 1

        # Condition to check if the graph directory for the input_file exists. If not, then create one.
        if not os.path.isdir(os.path.join(output_dir, graphs_dir)):
            os.makedirs(os.path.join(output_dir, graphs_dir))
            print("[INFO]. Directory created successfully.")

        for batch in read_feather_in_chunks(os.path.join(input_dir, final_inputs_dir, input_file)):
            """ Loop to iterate through batches in the input feather files. """

            # Condition to check if the 'saving model' directory for the input_file exists. If not, then create one.
            if not os.path.isdir(os.path.join(output_dir, saved_models_dir, input_file[:-8] + '_S-' + str(SEQUENCE))):
                print("[INFO]. Saving model directory for the input file \'{}\' does not exists. Creating the directory.".format(input_file))
                os.makedirs(os.path.join(output_dir, saved_models_dir, input_file[:-8] + '_S-' + str(SEQUENCE), str(batch_num) + '_batch'))
                print("[INFO]. Directory created successfully.")

            # Loop to calculate the tasmax, tasmin and precipitation_flux in the batch.
            print("[INFO]. Pre-Processing Batch-{} Inputs.".format(batch_num))
            for i in tqdm(range(len(batch))):
                batch.iloc[i, 25] += batch.iloc[i, 11]
                batch.iloc[i, 26] += batch.iloc[i, 11]
                if batch.iloc[i, 10] != np.inf:
                    batch.iloc[i, 27] = (1 + batch.iloc[i, 10] / 100) * batch.iloc[i, 27]

            # Final formatting of the dataframe before traning.
            batch = batch.drop(columns=['index', 'time', 'lat', 'lon', 'index_x', 'index_y', 'spatial_ref', 'W', 'T'])
            batch.gravel = batch.gravel.astype(int)
            batch.clay = batch.clay.astype(int)
            batch.silt = batch.silt.astype(int)
            batch.sand = batch.sand.astype(int)
            batch.awc = batch.awc.astype(int)
            batch.cec_soil = batch.cec_soil.astype(int)
            batch.texture_class = batch.texture_class.astype(int)
            batch.CO2 = batch.CO2.astype(int)
            batch['plant-day'] = batch['plant-day'].astype(int)
            batch['maturity-day'] = batch['maturity-day'].astype(int)

            # Dividing the dataframe in static and dynamic dataframe on the basis of features.
            static_data_input = batch[['plant-day', 'maturity-day', 'CO2', 'N', 'A', 'texture_class', 'soil_ph',
                                        'soil_caco3', 'cec_soil', 'oc', 'awc', 'sand', 'silt', 'clay', 'gravel']]
            static_data_label = batch[['yield_mai']]
            weather_array_1 = batch[['tasmax', 'tasmin', 'pr', 'gdd']]

            # Splitting the static and dynamic dataframe in training and testing set.
            fract = 1 - TEST_SPLIT

            static_X_train = static_data_input[:int(len(static_data_input) * fract)]
            static_X_test = static_data_input[int(len(static_data_input) * fract):]

            static_Y_train = static_data_label[:int(len(static_data_label) * fract)]
            static_Y_test = static_data_label[int(len(static_data_label) * fract):]

            dynamic_X_train = weather_array_1[:int(len(weather_array_1) * fract)]
            dynamic_X_test = weather_array_1[int(len(weather_array_1) * fract):]

            # Scaling static and dynamic data to assist in the training.
            scaler = MinMaxScaler(feature_range=(0.01, 1))
            static_X_train = scaler.fit_transform(static_X_train)
            static_Y_train = scaler.fit_transform(static_Y_train)
            dynamic_X_train = scaler.fit_transform(dynamic_X_train)

            # Clear the memory buffer and deleting un-necessary variables.
            gc.collect()
            del batch, static_data_input, static_data_label, weather_array_1

            # Splitting the static and dynamic dataframe in training and testing set.
            # fract = 1 - TEST_SPLIT

            # static_X_train = scaled_static_data[:int(len(scaled_static_data) * fract)]
            # static_X_test = scaled_static_data[int(len(scaled_static_data) * fract):]

            # static_Y_train = scaled_static_label[:int(len(scaled_static_label) * fract)]
            # static_Y_test = scaled_static_label[int(len(scaled_static_label) * fract):]

            # dynamic_X_train = scaled_dynamic_data[:int(len(scaled_dynamic_data) * fract)]
            # dynamic_X_test = scaled_dynamic_data[int(len(scaled_dynamic_data) * fract):]


            # Clear the memory buffer.
            gc.collect()

            # Defining the neural network for training the model.
            if flag == 1:
                dynamic_input = keras.Input(shape=(dynamic_X_train.shape[1], 1), dtype='float32')
                inner_lstm1 = LSTM(200, return_sequences=True)(dynamic_input)
                inner_lstm2 = LSTM(200, return_sequences=True)(inner_lstm1)
                lstm_out = LSTM(200, return_sequences=False)(inner_lstm2)

                static_input = keras.Input(shape=(static_X_train.shape[1]))
                inner_stat1 = Dense(200, activation='selu')(static_input)
                inner_stat1 = Dense(200, activation='selu')(inner_stat1)
                inner_stat2 = Dense(200, activation='selu')(inner_stat1)

                x = Concatenate()([lstm_out, inner_stat2])

                x = Dense(200, activation='selu')(x)
                x = Dense(200, activation='selu')(x)
                x = Dense(200, activation='selu')(x)

                dynamic_output = Dense(1, activation='selu')(x)

                model = Model(inputs=[dynamic_input, static_input], outputs=[dynamic_output])

                model.compile(loss=keras.metrics.mean_squared_error,
                              optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                              metrics=[keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])

                logs = "./.logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)
                tboard_callback = keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')

            else:
                model = keras.models.load_model(os.path.join(output_dir, saved_models_dir, ".tmp_model"))
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)
                tboard_callback = keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')


            # Training the mode on the dataset.
            history = model.fit(x=[dynamic_X_train, static_X_train], y=static_Y_train, validation_split=VAL_SPLIT, epochs=EPOCHS, callbacks=[tboard_callback, es], batch_size=BATCH_SIZE)

            # Concatinating test_DF to test the model later and save them inside specified folder.
            test_DF = pd.concat([static_X_test, static_Y_test, dynamic_X_test], axis=1)
            test_DF.to_feather(os.path.join(input_dir, test_files_dir, input_file[:-8] + '_S-' + str(SEQUENCE) + '_Batch-{}'.format(batch_num) + "_test_file.feather"))

            if batch_num == 1 and SEQUENCE == 1:
                loss_DF, val_loss_DF = pd.DataFrame(history.history['loss']), pd.DataFrame(history.history['val_loss'])

                loss_DF = loss_DF.rename(columns={0 : 'loss'})
                val_loss_DF = val_loss_DF.rename(columns={0 : 'val_loss'})
                print("loss_DF : ", loss_DF)
                print("val_loss_DF : ", val_loss_DF)
                loss_DF.to_feather(os.path.join(output_dir, graphs_dir, "loss_DF.feather"))
                val_loss_DF.to_feather(os.path.join(output_dir, graphs_dir, "val_loss_DF.feather"))

                gc.collect()
            else:
                prev_loss_DF = pd.read_feather(os.path.join(output_dir, graphs_dir, "loss_DF.feather"))
                prev_val_loss_DF = pd.read_feather(os.path.join(output_dir, graphs_dir, "val_loss_DF.feather"))

                loss_DF = pd.DataFrame(history.history['loss']).rename(columns={0: 'loss'})
                val_loss_DF = pd.DataFrame(history.history['val_loss']).rename(columns={0: 'val_loss'})

                loss_DF = pd.concat([prev_loss_DF, loss_DF], ignore_index=True)
                val_loss_DF = pd.concat([prev_val_loss_DF, val_loss_DF], ignore_index=True)

                loss_DF.to_feather(os.path.join(output_dir, graphs_dir, "loss_DF.feather"))
                val_loss_DF.to_feather(os.path.join(output_dir, graphs_dir, "val_loss_DF.feather"))

            # Plottting the loss graph and saving it in graph directory.
            plt.plot(loss_DF)
            plt.plot(val_loss_DF)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(output_dir, graphs_dir, "Loss_Stats.jpg"))
            plt.clf()

            # Saving the model after each epoch in corresponding directory..
            model.save(os.path.join(output_dir, saved_models_dir, input_file[:-8] + '_S-' + str(SEQUENCE), str(batch_num)+'_batch'))
            model.save(os.path.join(output_dir, saved_models_dir, ".tmp_model"))

            # Clearing the memory buffer and incrementing the variables.
            del loss_DF, val_loss_DF
            gc.collect()

            flag += 1
            batch_num += 1
        SEQUENCE += 1
