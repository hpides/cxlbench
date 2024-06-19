#! /usr/bin/env python3

import argparse
import glob
import json_util as ju
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from enums.benchmark_keys import BMKeys
from enums.file_names import FILE_TAG_SUBSTRING, PLOT_FILE_PREFIX

# benchmark configuration names
BM_CONFIG_OS_INTERLEAVING = "interleaved_sequential_reads"
BM_CONFIG_EXPLICIT_PLACEMENT = "parallel_sequential_reads"
BM_SUPPORTED_CONFIGS = [BM_CONFIG_OS_INTERLEAVING, BM_CONFIG_EXPLICIT_PLACEMENT]

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
    parser.add_argument("--bars", action="store_true")
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
    do_barplots = args.bars
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

    # -------------------------------------------------------------------------------------------------------------------

    df_os = df[(df[BMKeys.BM_NAME] == BM_CONFIG_OS_INTERLEAVING)]
    df_os = ju.flatten_nested_json_df(
        df_os,
        [
            BMKeys.MATRIX_ARGS,
            BMKeys.THREADS,
            BMKeys.NUMA_TASK_NODES,
            BMKeys.NUMA_MEMORY_NODES,
            "matrix_args.local_memory_access",
            "matrix_args.device_memory_access",
            "sub_bm_names",
        ],
    )
    df_os.to_csv("{}/interleaving.csv".format(output_dir))

    df_explicit = df[(df[BMKeys.BM_NAME] == BM_CONFIG_EXPLICIT_PLACEMENT)]
    df_explicit = ju.flatten_nested_json_df(
        df_explicit,
        [
            BMKeys.MATRIX_ARGS,
            BMKeys.THREADS,
            BMKeys.NUMA_TASK_NODES,
            BMKeys.NUMA_MEMORY_NODES,
            "matrix_args.local_memory_access",
            "matrix_args.device_memory_access",
            "sub_bm_names",
        ],
    )
    # only keep recors where local memory thread count and device memory thread count are equal
    df_explicit = df_explicit[
        df_explicit["local_memory_access.number_threads"] == df_explicit["device_memory_access.number_threads"]
    ]
    df_explicit.to_csv("{}/explicit-placement.csv".format(output_dir))

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
    df_os = df_os.drop(columns=drop_columns, errors="ignore")
    df_os.to_csv("{}/interleaving-reduced.csv".format(output_dir))

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
    df_explicit = df_explicit.drop(columns=drop_columns, errors="ignore")
    df_explicit.to_csv("{}/explicit-placement-reduced.csv".format(output_dir))

    # ------------------------------------------------------------------------------------------------------------------
    # create plots

    # prepare dataframes
    df_os["tag"] = "os_interleaving"
    df_os["total_threads"] = df_os["number_threads"]
    df_os["combined_bandwidth"] = df_os["bandwidth"]

    df_explicit["bandwidth"] = df_explicit["local_memory_access.results.bandwidth"]
    df_explicit["tag"] = "explicit_placement"
    df_explicit["total_threads"] = (
        df_explicit["local_memory_access.number_threads"] + df_explicit["device_memory_access.number_threads"]
    )
    df_explicit["combined_bandwidth"] = (
        df_explicit["local_memory_access.results.bandwidth"] + df_explicit["device_memory_access.results.bandwidth"]
    )

    df = pd.concat([df_os, df_explicit])

    # create grouped stacked barplot showing bw_0 and bw_1. For interleaving, only bw_0 is used. For explicit placement,
    # bw_0 shows the local memory access bandwidth and bw_1 shows the device memory access bandwidth.
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 2.5))
    hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]

    # Add bandwidth for device memory for explicit placement and os interleaving. For explicit placement, we actually
    # show the total bandwidth but put the local bandwidth in front of it to create a stacked bar.
    df["label1"] = df["tag"]
    df["label1"].replace({"os_interleaving": "OS-Interleaved", "explicit_placement": "Device Memory"}, inplace=True)
    barplot = sns.barplot(x="total_threads", y="combined_bandwidth", data=df, palette=hpi_palette, hue="label1")

    # Add bandwidth for local memory access.
    df["label2"] = df["tag"]
    df["label2"].replace({"os_interleaving": "None", "explicit_placement": "Local Memory"}, inplace=True)

    barplot = sns.barplot(
        x="total_threads",
        y="local_memory_access.results.bandwidth",
        data=df,
        color=hpi_palette[2],
        hue="label2",
    )

    fig = barplot.get_figure()

    plt.xlabel("Total Thread Count")
    plt.ylabel("Throughput [GB/s]")
    plt.yticks(range(0, int(max(df_explicit["combined_bandwidth"])) + 20, 20))

    # Add legend.
    bar_interleaved_access = mpatches.Patch(color=hpi_palette[0], label="Interleaved")
    bar_local_access = mpatches.Patch(color=hpi_palette[1], label="Local Memory")
    bar_device_access = mpatches.Patch(color=hpi_palette[2], label="Device Memory")
    handles = [bar_interleaved_access, bar_local_access, bar_device_access]
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.3),
        loc="upper center",
        borderaxespad=0.0,
        ncol=3,
        handles=handles,
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.5,
        frameon=False,
    )

    plt.tight_layout()

    fig.savefig("{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "explicit_placement"))
