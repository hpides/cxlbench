#! /usr/bin/env python3

# Experiment: Cost of Device Accesses

import argparse
import glob
import json_util as ju
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
import sys

from enums.benchmark_keys import BMKeys
from enums.file_names import FILE_TAG_SUBSTRING, PLOT_FILE_PREFIX


MAX_THREAD_COUNT = 120

# benchmark configuration names
BM_SUPPORTED_CONFIGS = ["split_memory_random_writes", "split_memory_random_reads"]

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
    if args.memory_nodes is not None:
        memory_nodes = args.memory_nodes
    else:
        memory_nodes = []

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
    ]

    drop_columns = [
        "index",
        "bm_type",
        "bm_name",
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
    BMKeys.TAG = "Workload"
    df[BMKeys.TAG] = df[BMKeys.EXEC_MODE] + " " + df[BMKeys.OPERATION]
    BMKeys.PERCENTAGE_SECOND_NODE = "percentage_pages_second_node"
    df[BMKeys.PERCENTAGE_SECOND_NODE] = 100 - df[BMKeys.PERCENTAGE_FIRST_NODE]
    df.to_csv("{}/{}.csv".format(output_dir, "results"))
    df = df.drop(columns=drop_columns, errors="ignore")
    df.to_csv("{}/{}.csv".format(output_dir, "results-reduced"))
    df = df[(df[BMKeys.THREAD_COUNT] <= MAX_THREAD_COUNT)]
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------------
    # create plots
    TAG_RND_READS = "Random Reads"
    TAG_SEQ_READS = "Seq Reads"
    TAG_RND_WRITES = "Random Writes"
    TAG_SEQ_WRITES = "Seq Writes"

    tag_replacements = {
        "random read": TAG_RND_READS,
        "sequential read": TAG_SEQ_READS,
        "random write": TAG_RND_WRITES,
        "sequential write": TAG_SEQ_WRITES,
    }

    df[BMKeys.TAG].replace(tag_replacements, inplace=True)

    # x ticks in steps of 10, 0 to 100.
    page_share_on_device = [x * 10 for x in range(0, 11)]

    sns.set_context("paper")
    sns.set(style="ticks")
    plt.figure(figsize=(5.5, 2.3))
    hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]
    palette = {
        TAG_RND_READS: hpi_palette[0],
        TAG_SEQ_READS: hpi_palette[0],
        TAG_RND_WRITES: hpi_palette[2],
        TAG_SEQ_WRITES: hpi_palette[2],
    }

    markers = {TAG_RND_READS: ".", TAG_SEQ_READS: ".", TAG_RND_WRITES: "X", TAG_SEQ_WRITES: "X"}

    dashes = {TAG_RND_READS: [1, 0], TAG_SEQ_READS: [1, 0], TAG_RND_WRITES: [1, 0], TAG_SEQ_WRITES: [1, 0]}

    hue_order = [TAG_RND_READS, TAG_RND_WRITES]

    lineplot = sns.lineplot(
        data=df,
        x=BMKeys.PERCENTAGE_SECOND_NODE,
        y=BMKeys.BANDWIDTH_GB,
        palette=palette,
        style=BMKeys.TAG,
        dashes=dashes,
        markers=markers,
        hue_order=hue_order,
        hue=BMKeys.TAG,
    )
    lineplot.set_xticks(page_share_on_device)
    lineplot.set_xticklabels(page_share_on_device)
    lineplot.yaxis.grid()
    if y_tick_distance is not None:
        lineplot.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_distance))

    lineplot.legend(title=None)
    sns.move_legend(
        lineplot,
        "lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.5,
    )

    fig = lineplot.get_figure()

    plt.xlabel("Pages on device memory in %")
    plt.ylabel("Throughput in GB/s")

    plt.tight_layout()

    fig.savefig("{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "device_penalty"), bbox_inches="tight", pad_inches=0)
