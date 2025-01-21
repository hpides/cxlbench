#! /usr/bin/env python3

import argparse
import glob
import numpy as np
from builtins import len, str, int, list, any, print, float

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

BM_SUPPORTED_CONFIGS = ["tree_index_lookup", "tree_index_update"]
KEY_LOCATION_M0 = "m0_memory_location"
KEY_LOCATION_M1 = "m1_memory_location"
SAPRAP_MEM_REPLACEMENT = {"0, 1, 2, 3": "CPU0", "4, 5, 6, 7": "CPU1", "8": "CXL"}
EMR_MEM_REPLACEMENT = {"0": "CPU0", "1": "CPU1", "2": "CXL"}


def is_saprap(df):
    saprap_mem_options = list(SAPRAP_MEM_REPLACEMENT.keys())
    for key in [BMKeys.NUMA_MEMORY_NODES_M0, BMKeys.NUMA_MEMORY_NODES_M0]:
        if any(config not in saprap_mem_options for config in df[key].unique()):
            return False
    return True


def is_emr(df):
    emr_mem_options = list(EMR_MEM_REPLACEMENT.keys())
    for key in [BMKeys.NUMA_MEMORY_NODES_M0, BMKeys.NUMA_MEMORY_NODES_M0]:
        if any(config not in emr_mem_options for config in df[key].unique()):
            return False
    return True


def add_memory_location(df):
    replacements = None
    if is_saprap(df):
        replacements = SAPRAP_MEM_REPLACEMENT
    elif is_emr(df):
        replacements = EMR_MEM_REPLACEMENT

    if replacements:
        df[KEY_LOCATION_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].replace(replacements)
        df[KEY_LOCATION_M1] = df[BMKeys.NUMA_MEMORY_NODES_M1].replace(replacements)
    else:
        df[KEY_LOCATION_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0]
        df[KEY_LOCATION_M1] = df[BMKeys.NUMA_MEMORY_NODES_M1]
    return df


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
        _tag = ""
        if FILE_TAG_SUBSTRING in path:
            path_parts = path.split(FILE_TAG_SUBSTRING)
            assert (
                len(path_parts) == 2
            ), "Make sure that the substring {} appears only once in a result file name.".format(FILE_TAG_SUBSTRING)
            tag_part = path_parts[-1]
            assert "-" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            _tag = tag_part.split(".")[0]

        df = pd.read_json(path)
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
            BMKeys.EXPLODED_THREAD_CORES,
        ],
    )
    df[BMKeys.NUMA_MEMORY_NODES_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].apply(mplt.values_as_string)
    df[BMKeys.NUMA_MEMORY_NODES_M1] = df[BMKeys.NUMA_MEMORY_NODES_M1].apply(mplt.values_as_string)
    df.to_csv("{}/data.csv".format(output_dir))
    df = add_memory_location(df)

    drop_columns = [
        "index",
        "bm_type",
        "compiler",
        "git_hash",
        "hostname",
        "matrix_args",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")

    df["M_ops"] = df[BMKeys.OPS_PER_SECOND] / 10**6
    df["inner_node_size"] = df[BMKeys.CUSTOM_OPS].apply(lambda x: x.split(",", 1)[0].rsplit("_", 1)[1])
    df["leaf_node_size"] = df["inner_node_size"].astype(int) / 2
    df["inner_leaf_sizes"] = df["inner_node_size"] + "/" + df["leaf_node_size"].astype(int).astype(str)
    bm_name_replacement = {"tree_index_lookup": "Lookup", "tree_index_update": "Update"}
    df["bm_name_short"] = df[BMKeys.BM_NAME].replace(bm_name_replacement)
    df["workload"] = df["bm_name_short"] + "\n" + df["inner_leaf_sizes"]

    df.to_csv("{}/data-reduced.csv".format(output_dir))

    node_mapping = {"0": "CPU0", "8": "CPU1", "255": "GPU", "CPU0": "CPU", "CPU1": "GPU"}
    # Define the configuration column
    df["Memory Tiers"] = (
        "Inner: "
        + df[KEY_LOCATION_M0].apply(lambda x: node_mapping[x])
        + ", Leaf: "
        + df[KEY_LOCATION_M1].apply(lambda x: node_mapping[x])
    )
    df = df.rename({"M_ops": "Million Ops/s", "number_threads": "Number of Threads"}, axis="columns")
    # Get unique workloads
    unique_workloads = df["workload"].unique()

    # Define the number of rows and columns for the subplots
    num_cols = len(unique_workloads)
    sns.set_style("darkgrid")
    sns.set(font_scale=1.5)

    # Iterate over each workload
    g = sns.relplot(
        data=df,
        x="Number of Threads",
        y="Million Ops/s",
        hue="Memory Tiers",
        style="Memory Tiers",
        markers=True,
        col="workload",
        kind="line",
        col_wrap=num_cols,
    )
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.4, 0.9),
        ncol=3,
        title=None,
        frameon=False,
    )
    g.set(xticks=np.arange(18, step=2))
    # g.set(xticks=np.arange(80, step=8))
    g.axes[0].set_title("Lookup")
    g.axes[1].set_title("Update")
    # Save the figure
    g.savefig(
        "{}/{}{}-lineplots.pdf".format(output_dir, PLOT_FILE_PREFIX, "fptree"),
    )


if __name__ == "__main__":
    main()
