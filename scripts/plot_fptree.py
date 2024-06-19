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

from scripts.enums.file_names import PLOT_FILE_PREFIX, FILE_TAG_SUBSTRING


# benchmark configuration names
BM_REPLACE_OLD_NAMES = {
    "hybrid_tree_index_lookup_local_baseline": "hybrid_tree_index_lookup",
    "hybrid_tree_index_update_local_baseline": "hybrid_tree_index_update",
}

BM_SUPPORTED_CONFIGS = ["hybrid_tree_index_lookup", "hybrid_tree_index_update"]

PRINT_DEBUG = False


def main():
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

    os.makedirs(output_dir, exist_ok=True)

    y_tick_distance = None
    if args.y_tick_distance is not None:
        y_tick_distance = float(args.y_tick_distance)

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
        df[mplt.KEY_TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    df[mplt.KEY_BM_NAME].replace(BM_REPLACE_OLD_NAMES, inplace=True)
    bm_names = df[mplt.KEY_BM_NAME].unique()
    print("Existing BM groups: {}".format(bm_names))

    # -------------------------------------------------------------------------------------------------------------------

    df = df[(df[mplt.KEY_BM_NAME].isin(BM_SUPPORTED_CONFIGS))]
    df = ju.flatten_nested_json_df(
        df,
        [
            mplt.KEY_MATRIX_ARGS,
            mplt.KEY_THREADS,
            mplt.KEY_NUMA_TASK_NODES,
            mplt.KEY_M0_NUMA_MEMORY_NODES,
            mplt.KEY_M1_NUMA_MEMORY_NODES,
        ],
    )
    df.to_csv("{}/data.csv".format(output_dir))
    df[mplt.KEY_M0_NUMA_MEMORY_NODES] = df[mplt.KEY_M0_NUMA_MEMORY_NODES].apply(mplt.values_as_string)
    df[mplt.KEY_M1_NUMA_MEMORY_NODES] = df[mplt.KEY_M1_NUMA_MEMORY_NODES].apply(mplt.values_as_string)
    drop_columns = [
        "index",
        "bm_type",
        "compiler",
        "git_hash",
        "hostname",
        "matrix_args",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")

    df["M_ops"] = df[mplt.KEY_OPS_PER_SECOND] / 10**6
    df["inner_node_size"] = df[mplt.KEY_CUSTOM_OPS].apply(lambda x: x.split(",", 1)[0].rsplit("_", 1)[1])
    df["leaf_node_size"] = df["inner_node_size"].astype(int) / 2
    df["inner_leaf_sizes"] = df["inner_node_size"] + "/" + df["leaf_node_size"].astype(int).astype(str)
    bm_name_replacement = {"hybrid_tree_index_lookup": "Lookup", "hybrid_tree_index_update": "Update"}
    df["bm_name_short"] = df[mplt.KEY_BM_NAME].replace(bm_name_replacement)
    df["workload"] = df["bm_name_short"] + "\n" + df["inner_leaf_sizes"]

    df.to_csv("{}/data-reduced.csv".format(output_dir))

    thread_counts = df[mplt.KEY_THREAD_COUNT].unique()
    for thread_count in thread_counts:
        create_plot(df, thread_count, y_tick_distance, output_dir)


def create_plot(df, thread_count, y_tick_distance, output_dir):
    df = df[df[mplt.KEY_THREAD_COUNT] == thread_count]
    # create plots

    # LABEL_LOC = "Local"
    # LABEL_CXL = "CXL"
    # node_names = {0: LABEL_LOC, 1: LABEL_CXL}
    df["config"] = df[mplt.KEY_M0_NUMA_MEMORY_NODES].astype(str) + df[mplt.KEY_M1_NUMA_MEMORY_NODES].astype(str)
    config_names = {"00": "Local", "01": "Hybrid", "11": "CXL"}
    df["config"].replace(config_names, inplace=True)

    sns.set(style="ticks")
    plt.figure(figsize=(5.5, 2.3))

    # hue_order = [LABEL_LOC, LABEL_CXL]
    ax = sns.barplot(
        data=df,
        x="workload",
        y="M_ops",
        palette="colorblind",
        hue="config",
        # hue_order=hue_order
    )
    plt.xticks(rotation=0)
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

    plt.xlabel("Workload with Inner/Leaf Node Size (Byte)")
    plt.ylabel("Million Ops/s")
    # plt.title(BM_NAME_TITLE[bench_name], y=1, x=0.1)
    # plt.xticks(rotation=45)

    plt.tight_layout()

    fig.savefig(
        "{}/{}{}-T{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "fptree", thread_count),
        bbox_inches="tight",
        pad_inches=0,
    )


if __name__ == "__main__":
    main()
