"""
This is the main module that has to be executed from the root directory, i.e. "mema-bench". It creates result PNGs and
displays them on an user interface.
"""

# ! ../viz/venv/bin/python

import argparse
import os
import sys

from comparison_plotter import ComparisonPlotter

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

    parser.add_argument("results_first", type=valid_path, help="path to the results directory")
    parser.add_argument("results_second", type=valid_path, help="path to the results directory")
    parser.add_argument("label_first", help="Label of the measurements in the first result directory")
    parser.add_argument("label_second", help="Label of the measurements in the second result directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    args = parser.parse_args()

    # get the output directory paths
    output_dir_string = "./plots"
    if args.output_dir is not None:
        output_dir_string = args.output_dir

    output_dir = os.path.abspath(output_dir_string)
    results_first = args.results_first
    results_second = args.results_second
    label_first = args.label_first
    label_second = args.label_second

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(results_first) or os.path.isfile(results_second):
        sys.exit("Result paths have to be directories.")

    # create plots 
    plotter = ComparisonPlotter(results_first, results_second, label_first, label_second, output_dir)
    plotter.process_matrix_jsons_comparison()
