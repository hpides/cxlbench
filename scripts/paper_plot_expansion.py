#! /usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import string

from enums.benchmark_keys import BMKeys
from enums.file_names import PLOT_FILE_PREFIX
from matplotlib.lines import Line2D
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
    parser.add_argument(
        "--compare_region_size", action="store_true", help="Generate a throughput heatmap comparing the impact of different region sizes"
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
    compare_region_size = args.compare_region_size
    if args.memory_nodes is not None:
        memory_nodes = args.memory_nodes
    else:
        memory_nodes = []

    os.makedirs(output_dir, exist_ok=True)

    # create plots
    plotter = PlotGenerator(results, output_dir, no_plots, latency_heatmap, compare_region_size, memory_nodes)
    df = plotter.process_matrix_jsons(True)
    df = df[df[BMKeys.NUMA_MEMORY_NODE_WEIGHTS_M0] != "1:0"]
    df[BMKeys.EXEC_MODE] = df[BMKeys.EXEC_MODE].replace({"sequential":"seq", "random":"rnd"})
    df["access_types"] = "CPU & GPU | " + (df[BMKeys.EXEC_MODE] + " " + df[BMKeys.OPERATION]).apply(string.capwords)

    sns.set_style("darkgrid")

    ykey = BMKeys.BANDWIDTH_GB
    xkey = BMKeys.NUMA_MEMORY_NODE_WEIGHTS_M0
    hue_key = "access_types"

    markers = {}
    marker_options = ['o', 's', '^', 'v', 'D', 'x', '+', '*', '.', '|', '_']
    hues = df[hue_key].unique()
    for idx, hue in enumerate(hues):
        markers[hue] = marker_options[idx]

    hue_order = ["CPU & GPU | Seq Read", "CPU & GPU | Rnd Read", "CPU & GPU | Seq Write", "CPU & GPU | Rnd Write"]

    plt.figure(figsize=(4, 3))
    ax = sns.lineplot(data=df, x=xkey, y=ykey, hue=hue_key, style=hue_key, markers=markers, dashes=False, hue_order=hue_order, zorder=2)
    ax.tick_params(axis='x', rotation=90)
    # ax.xaxis.set_tick_params(pad=-3)
    # ax.yaxis.set_tick_params(pad=-3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_ylim(0, None)

    # Labels and title
    plt.xlabel("Interleaving Ratio")
    plt.ylabel("Throughput [GB/s]")
    colors = sns.color_palette()
    # Seq read
    ax.axhline(y=281, linestyle="--", color=colors[0], linewidth=1, zorder=1)
    # Rnd read
    ax.axhline(y=224, linestyle=":", color=colors[1], linewidth=1, zorder=1)
    # Seq write
    ax.axhline(y=236, linestyle="-.", color=colors[2], linewidth=1, zorder=1)
    # Rnd write
    ax.axhline(y=192, linestyle="-", color=colors[3], linewidth=1, zorder=1)

    hline_handles = [
        Line2D([0], [0], color=colors[0], linestyle="--", linewidth=1),
        Line2D([0], [0], color=colors[1], linestyle=":", linewidth=1),
        Line2D([0], [0], color=colors[2], linestyle="-.", linewidth=1),
        Line2D([0], [0], color=colors[3], linestyle="-", linewidth=1),
    ]

    seaborn_handles, seaborn_labels = ax.get_legend_handles_labels()
    all_handles = seaborn_handles + hline_handles
    all_labels = seaborn_labels + ["CPU | Seq Read", "CPU | Rnd Read", "CPU | Seq Write", "CPU | Rnd Write"]

    ax.legend(handles=all_handles, labels = all_labels, loc="upper right", frameon=False)
    sns.move_legend(
        ax,
        "upper center",
        bbox_to_anchor=(0.5, 1.48),
        ncol=2,
        title="Memory Location | Access Type",
        frameon=False,
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.5,
    )

    plt.grid(True)

    # Write pdf
    filename = "{}/{}{}-lineplots.pdf".format(output_dir, PLOT_FILE_PREFIX, "bw-expansion")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
