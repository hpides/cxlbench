#! /usr/bin/env python3

import argparse
import os
import sys

from plot_generator import PlotGenerator

def dir_path(path):
    """
    Checks if the given directory path is valid.

    :param path: directory path to the results folder
    :return: bool representing if path was valid
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"The path to the results directory is not valid.")

def valid_path(path):
    return path if os.path.isfile(path) else dir_path(path)

if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument("results", type=valid_path, help="path to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    args = parser.parse_args()

    # get the output directory paths
    output_dir_string = "./plots"
    if args.output_dir is not None:
        output_dir_string = args.output_dir

    output_dir = os.path.abspath(output_dir_string)
    results = args.results

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(results):
        sys.exit("Result paths have to be directories.")

    # create plots 
    plotter = PlotGenerator(results, output_dir)
    plotter.process_matrix_jsons()
