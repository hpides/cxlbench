#! /usr/bin/env python3

# Paper: Individual Throughput Scaling

import argparse
import glob
import json_util as ju
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import memaplot as mplt
import os
import pandas as pd
import seaborn as sns
import sys

from enums.benchmark_keys import BMKeys
from enums.file_names import PLOT_FILE_PREFIX, FILE_TAG_SUBSTRING
from matplotlib.lines import Line2D  # For creating empty handles

MAX_THREAD_COUNT = 48

# benchmark configuration names
BM_SUPPORTED_CONFIGS = [
    "scale_reads_rnd",
    "scale_writes_rnd",
    "scale_writes_seq",
    "scale_reads_seq",
    "random_reads",
    "random_writes",
    "sequential_reads",
    "sequential_writes",
]

PRINT_DEBUG = False

def print_debug(message):
    if PRINT_DEBUG:
        print(message)


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
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("results", type=valid_path, help="path to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    parser.add_argument("-y", "--y_tick_distance", help="distance between y-ticks")
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
            output_dir_string = output_dir_string + "/plots/"
        else:
            assert os.path.isdir(results_path)
            output_dir_string = results_path + "/plots"

    y_tick_distance = None
    if args.y_tick_distance is not None:
        y_tick_distance = int(args.y_tick_distance)

    print("Output directory:", output_dir_string)
    output_dir = os.path.abspath(output_dir_string)
    results = args.results

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # collect jsons containing matrix arguments
    matrix_jsons = None
    if os.path.isfile(results):
        if not results.endswith(".json"):
            sys.exit("Result path is a single file but is not a .json file.")
        matrix_jsons = [results]
    else:
        matrix_jsons = [path for path in glob.glob(results + "/*.json")]

    # create json file list
    dfs = []
    for path in matrix_jsons:
        # Get the tag from the file name.
        tag = ""
        if FILE_TAG_SUBSTRING in path:
            path_parts = path.split(FILE_TAG_SUBSTRING)
            assert (
                len(path_parts) == 2
            ), "Make sure that the substring {} appears only once in a result file name.".format(FILE_TAG_SUBSTRING)
            tag_part = path_parts[-1]
            assert "-" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            assert "_" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            tag = tag_part.split(".")[0]

        df = pd.read_json(path)
        df[BMKeys.TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    bm_names = df[BMKeys.BM_NAME].unique()
    print("Existing BM groups: {}".format(bm_names))

    # ------------------------------------------------------------------------------------------------------------------
    deny_list_explosion = [
        BMKeys.MATRIX_ARGS,
        BMKeys.THREADS,
        BMKeys.NUMA_TASK_NODES,
        BMKeys.NUMA_MEMORY_NODES,
        BMKeys.NUMA_MEMORY_NODES_M0,
        BMKeys.NUMA_MEMORY_NODES_M1,
        BMKeys.EXPLODED_THREAD_CORES,
    ]

    drop_columns = [
        "index",
        "bm_type",
        "compiler",
        "git_hash",
        "hostname",
        "matrix_args",
        "nt_stores_instruction_set",
        "sub_bm_names",
    ]

    df = df[(df[BMKeys.BM_NAME].isin(BM_SUPPORTED_CONFIGS))]
    df = ju.flatten_nested_json_df(df, deny_list_explosion)
    # Transform GiB/s to GB/s
    df[BMKeys.BANDWIDTH_GB] = df[BMKeys.BANDWIDTH_GiB] * (1024**3 / 1e9)
    df[BMKeys.TAG] = df[BMKeys.EXEC_MODE] + " " + df[BMKeys.OPERATION]
    df[BMKeys.NUMA_MEMORY_NODES_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].apply(mplt.values_as_string)
    numa_node_replacements = {
        "2": "1 Blade",
        "2, 3": "2 Blades",
        "2, 3, 4": "3 Blades",
        "2, 3, 4, 5": "4 Blades",
    }
    df["cxl_config"] = df[BMKeys.NUMA_MEMORY_NODES_M0]
    df["cxl_config"].replace(numa_node_replacements, inplace=True)
    df.to_csv("{}/{}.csv".format(output_dir, "results"))
    df = df.drop(columns=drop_columns, errors="ignore")
    df.to_csv("{}/{}.csv".format(output_dir, "results-reduced"))
    df = df[(df[BMKeys.THREAD_COUNT] <= MAX_THREAD_COUNT)]
    df = df[df[BMKeys.ACCESS_SIZE] >= 64]
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------------
    # create plots
    # plt.figure(figsize=(5, 2.2))
    access_sizes = df[BMKeys.ACCESS_SIZE].unique()
    # assert len(access_sizes) == 1
    # assert access_sizes[0] == 64
    exec_modes = df[BMKeys.EXEC_MODE].unique()
    # assert len(exec_modes) == 1
    # assert exec_modes[0] == "sequential"
    operations = df[BMKeys.OPERATION].unique()
    # assert len(operations) == 1
    # assert operations[0] == "read"

    df = df[df["cxl_config"].isin(["1 Blade", "2 Blades", "3 Blades", "4 Blades"])]
    # df = df.sort_values(by=['cxl_config', BMKeys.ACCESS_SIZE], ascending=False)

    thread_counts = df[BMKeys.THREAD_COUNT].unique()
    thread_counts.sort()

    sns.set_theme(style="ticks")
    hpi_palette = ["#f5a700", "#dc6007", "#b00539"]
    colors = hpi_palette + ["#6b009c", "#006d5b"]

    # Set up the plot grid layout in one row
    num_bm_names = len(bm_names)
    required_bm_names = ["sequential_reads", "random_reads", "sequential_writes", "random_writes"]
    assert all(bm_name in bm_names for bm_name in required_bm_names)

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'text.latex.preamble': r'\usepackage{libertine}'
    })

    fig, axes = plt.subplots(nrows=1, ncols=num_bm_names, figsize=(3 * num_bm_names, 2.3), squeeze=False, sharey=True)
    plt.subplots_adjust(wspace=0)

    # Create a list to store all plots to extract handles and labels for legend
    lines = []
    labels = []

    cxl_config_order = ["4 Blades", "3 Blades", "2 Blades", "1 Blade"]

    for i, bm_name in enumerate(required_bm_names):
        df_filtered = df[df[BMKeys.BM_NAME] == bm_name]

        thread_counts = df_filtered[BMKeys.THREAD_COUNT].unique()
        thread_counts.sort()

        ax = axes[0, i]

        plot = sns.lineplot(
            data=df_filtered,
            x=BMKeys.THREAD_COUNT,
            y=BMKeys.BANDWIDTH_GB,
            hue="cxl_config",
            hue_order=cxl_config_order,
            palette=colors,
            style=BMKeys.ACCESS_SIZE,
            markers=True,
            markersize=5,
            linewidth=1.5,
            dashes=False,
            ax=ax,
            legend=True,
        )

        ax.set_title(f"{bm_name}".replace("_", " ").title(), y=-0.017)
        ax.set_xticks(thread_counts)
        ax.set_xticklabels(thread_counts)
        ax.set_ylabel("Throughput [GB/s]")
        ax.set_xlabel("Thread Count")
        ax.yaxis.grid(True)
        ax.set_xticks(thread_counts[::1])  # show every 2nd x-tick
        ax.set_xticklabels(thread_counts[::1])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # collect handles and labels for the legend from the first subplot
        if i == 0:
            handles, lbls = ax.get_legend_handles_labels()
            lines.extend(handles)
            labels.extend(lbls)
        else:
            ax.tick_params(axis="y", which="both", length=0)

    labels = [
        label.replace(" Blades", "")
        .replace(" Blade", "")
        .replace("cxl_config", "Device Count")
        .replace("access_size", "Access Size")
        for label in labels
    ]

    # Create the merged legend
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.1),
        ncol=10,
        frameon=True,
        fancybox=True,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.5,
        title="",
    )

    for i, bm_name in enumerate(bm_names):
        axes[0, i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(
        "{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "scale_throughput"), bbox_inches="tight", pad_inches=0
    )
