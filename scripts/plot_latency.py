#! /usr/bin/env python3

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


# benchmark configuration names
BM_SUPPORTED_CONFIGS = ["lat_read", "lat_write_cache", "lat_write_none"]
BM_NAME_TITLE = {"lat_read": "Read", "lat_write_cache": "Read, Write", "lat_write_none": "Read, Write, CLWB"}

PRINT_DEBUG = False


def create_plot(df, bench_name, node_names):
    plot_df = df[df[BMKeys.BM_NAME] == bench_name]
    plot_df["op_size"] = plot_df[BMKeys.CUSTOM_OPS].apply(lambda x: x.split(",", 1)[0].rsplit("_", 1)[1])

    LABEL_LOC = "Local"
    LABEL_CXL = "CXL"
    node_names = {0: LABEL_LOC, 1: LABEL_CXL}
    plot_df["node_names"] = plot_df[BMKeys.NUMA_MEMORY_NODES_M0].replace(node_names)

    sns.set(style="ticks")
    plt.figure(figsize=(5.5, 2.3))

    order = plot_df["op_size"].unique()
    hue_order = [LABEL_LOC, LABEL_CXL]
    ax = sns.barplot(
        data=plot_df,
        x="op_size",
        y=BMKeys.LAT_AVG,
        palette="colorblind",
        hue="node_names",
        order=order,
        hue_order=hue_order,
    )
    # plot.set_xticks(thread_counts)
    # plot.set_xticklabels(thread_counts)
    ax.yaxis.grid()
    if y_tick_distance is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_distance))

    ax.legend(title=None)

    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.5,
    )

    # TODO(MW) add error bars, based on
    # https://stackoverflow.com/questions/62820959/use-precalculated-error-bars-with-seaborn-and-barplot

    fig = ax.get_figure()

    plt.xlabel("Access Size")
    plt.ylabel("Latency in ns")
    # plt.title(BM_NAME_TITLE[bench_name], y=1, x=0.1)
    # plt.xticks(rotation=45)

    plt.tight_layout()

    fig.savefig(
        "{}/{}{}-{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "latency", bench_name), bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("results", type=mplt.valid_path, help="path to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    parser.add_argument("-y", "--y_tick_distance", help="distance between y-ticks")
    parser.add_argument("--nodes", nargs="+", help="names of the memory nodes")
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

    if args.nodes is not None:
        node_names = args.nodes
    else:
        node_names = None

    os.makedirs(output_dir, exist_ok=True)

    y_tick_distance = None
    if args.y_tick_distance is not None:
        y_tick_distance = int(args.y_tick_distance)

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
            tag = tag_part.split(".")[0]

        df = pd.read_json(path)
        df[BMKeys.TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    bm_names = df[BMKeys.BM_NAME].unique()
    print("Existing BM groups: {}".format(bm_names))

    # -------------------------------------------------------------------------------------------------------------------

    df = df[(df[BMKeys.BM_NAME].isin(BM_SUPPORTED_CONFIGS))]
    df = ju.flatten_nested_json_df(
        df,
        [
            BMKeys.MATRIX_ARGS,
            BMKeys.THREADS,
            BMKeys.NUMA_TASK_NODES,
            BMKeys.NUMA_MEMORY_NODES_M0,
            BMKeys.NUMA_MEMORY_NODES_M1,
        ],
    )
    df.to_csv("{}/data.csv".format(output_dir))
    df[BMKeys.NUMA_MEMORY_NODES_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].apply(mplt.get_single_list_value)
    drop_columns = [
        "index",
        "bm_type",
        "compiler",
        "git_hash",
        "hostname",
        "matrix_args",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")
    df.to_csv("{}/data-reduced.csv".format(output_dir))

    # ------------------------------------------------------------------------------------------------------------------
    # create plots

    for bench_name in BM_SUPPORTED_CONFIGS:
        create_plot(df, bench_name, node_names)
