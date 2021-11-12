# /**
#  * @file trt_optimizer.py
#  * @author Samay Pashine (samay@iiti.ac.in)
#  * @modified Samay Pashine (samay@iiti.ac.in)
#  * @brief opens the saved model, optimizes it and save it in the specified location.
#  * @version 1.0
#  * @date 2021-11-12
#  * @copyright Copyright (c) 2021
#  */

# Importing necessary libraries
import os
import argparse
from constants import *
from tensorflow.python.compiler.tensorrt import trt_convert as trt

if __name__ == '__main__':
    """ Driver code to process the whole task of optimizing model sequentially. """

    # Argument parser to control things from command-line.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Enter the path of the directory of un-optimized saved model.",
                        type=str)

    parser.add_argument("-o", "--output", help="Enter the path where to save the optimized saved model.",
                        default=os.path.join(output_dir, trt_saved_model),
                        type=str)
    args = parser.parse_args()

    print("Conversion : Started")

    # Configuring the converter.
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.input)

    # COnverting and saving the optimized model at specified location.
    converter.convert()
    converter.save(args.output)

    print("Conversion : Finished")
