#! /usr/bin/env python3

import argparse
import os

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
        raise argparse.ArgumentTypeError("The path to the results directory is not valid.")


def valid_path(path):
    return path if os.path.isfile(path) else dir_path(path)


if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument("results", type=valid_path, help="path to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    parser.add_argument("--noplots", action="store_true")
    parser.add_argument("--bars", action="store_true")
    parser.add_argument(
        "--latency_heatmap", action="store_true", help="Generate a heatmap with the average thread's access latency"
    )
    parser.add_argument("--memory_nodes", nargs="+", help="names of the memory nodes")
    args = parser.parse_args()

    results_path = args.results
    if not results_path.startswith("./") and not results_path.startswith("/"):
        results_path = "./" + results_path

    output_dir_string = None

    # get the output directory paths
    if args.output_dir is not None:
        output_dir_string = args.output_dir
    else:
        if os.path.isfile(results_path):
            parts = results_path.rsplit("/", 1)
            assert len(parts)
            output_dir_string = parts[0]
            id = parts[1].split(".", 1)[0]
            output_dir_string = output_dir_string + "/plots/" + id
        else:
            assert os.path.isdir(results_path)
            output_dir_string = results_path + "/plots"

    print("Output directory:", output_dir_string)
    output_dir = os.path.abspath(output_dir_string)
    results = args.results
    no_plots = args.noplots
    latency_heatmap = args.latency_heatmap
    if args.memory_nodes is not None:
        memory_nodes = args.memory_nodes
    else:
        memory_nodes = []

    os.makedirs(output_dir, exist_ok=True)

    # create plots
    plotter = PlotGenerator(results, output_dir, no_plots, latency_heatmap, memory_nodes)
    plotter.process_matrix_jsons()
