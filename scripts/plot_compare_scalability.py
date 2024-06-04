"""
This module plots multiple results into one plot
(throughput x thread count) for a given list of
access sizes.
"""

import argparse
import glob
import itertools
import logging

import os
import sys
from enum import StrEnum

import pandas as pd
from matplotlib import pyplot as plt

from scripts import json_util
from scripts.compare import valid_path
from scripts.plot_generator import FILE_TAG_SUBSTRING, FLUSH_INSTR_NONE

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout,
)


class Keys(StrEnum):
    ACCESS_SIZE = "access_size"
    BANDWIDTH_GiB = "bandwidth"
    BANDWIDTH_GB = "bandwidth_gb"
    BM_GROUP = "bm_name"
    BM_TYPE = "bm_type"
    CHUNK_SIZE = "min_io_chunk_size"
    CUSTOM_OPS = "custom_operations"
    EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"
    EXPLODED_NUMA_TASK_NODES = "benchmarks.config.numa_task_nodes"
    LAT_AVG = "latency.avg"
    MATRIX_ARGS = "matrix_args"
    MEMORY_REGION_SIZE = "memory_region_size"
    NUMA_TASK_NODES = "numa_task_nodes"
    NUMA_MEMORY_NODES = "numa_memory_nodes"
    OPERATION = "operation"
    OPERATION_COUNT = "number_operations"
    PARTITION_COUNT = "number_partitions"
    RANDOM_DISTRIBUTION = "random_distribution"
    RUN_TIME = "run_time"
    TAG = "tag"
    THREAD_COUNT = "number_threads"
    THREADS = "threads"
    THREADS_LEVELED = "benchmarks.results.threads"
    FLUSH_INSTRUCTION = "flush_instruction"


class BM_Groups(StrEnum):
    RANDOM_WRITES = ("random_writes",)
    RANDOM_READS = ("random_reads",)
    SEQUENTIAL_WRITES = ("sequential_writes",)
    SEQUENTIAL_READS = ("sequential_reads",)
    OPERATION_LATENCY = "operation_latency"

    def get_title(self) -> object:
        if self == self.RANDOM_WRITES:
            return "Random Writes"
        elif self == self.RANDOM_READS:
            return "Random Reads"
        elif self == self.SEQUENTIAL_WRITES:
            return "Sequential Writes"
        elif self == self.SEQUENTIAL_READS:
            return "Sequential Reads"
        elif self == self.OPERATION_LATENCY:
            return "Operation Latency"


def format_path(results_path):
    if not results_path.startswith("./") and not results_path.startswith("/"):
        return "./" + results_path
    return results_path


def parse_matrix_jsons(results):
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
            ), "Make sure that the substring {} appears only once in a result file name.".format(
                FILE_TAG_SUBSTRING
            )
            tag_part = path_parts[-1]
            assert (
                "-" not in tag_part
            ), "Make sure that the tag is the last part of the name before the file extension."
            assert (
                "_" not in tag_part
            ), "Make sure that the tag is the last part of the name before the file extension."
            tag = tag_part.split(".")[0]

        df = pd.read_json(path)
        df[Keys.TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    bm_names = df[Keys.BM_GROUP].unique()
    print("Existing BM groups: {}".format(bm_names))
    selected_bm_groups = [
        BM_Groups.SEQUENTIAL_WRITES.value,
        BM_Groups.RANDOM_WRITES.value,
        BM_Groups.RANDOM_READS.value,
        BM_Groups.SEQUENTIAL_READS.value,
    ]
    print("Supported BM groups: {}".format(selected_bm_groups))

    df = df[
        (df[Keys.BM_GROUP].isin(selected_bm_groups)) & (df[Keys.BM_TYPE] == "single")
    ]
    df = json_util.flatten_nested_json_df(
        df,
        [
            Keys.MATRIX_ARGS,
            Keys.THREADS_LEVELED,
            Keys.EXPLODED_NUMA_MEMORY_NODES,
            Keys.EXPLODED_NUMA_TASK_NODES,
        ],
    )

    # If only latency benchnarks have been performed, the dataframe does note have a KEY_ACCESS_SIZE column so it
    # must be added.
    if Keys.ACCESS_SIZE not in df.columns:
        df[Keys.ACCESS_SIZE] = -1
    df[Keys.ACCESS_SIZE] = df[Keys.ACCESS_SIZE].fillna(-1)
    df[Keys.ACCESS_SIZE] = df[Keys.ACCESS_SIZE].astype(int)

    # For read benchmarks, an additional flush instruction will never be performed. As 'none' is also one of the
    # valid flush instructions, we set the corresponding value to 'none'. If only read benchnarks have been
    # performed, the dataframe does note have a KEY_FLUSH_INSTRUCTION column so it must be added.
    if Keys.FLUSH_INSTRUCTION not in df.columns:
        df[Keys.FLUSH_INSTRUCTION] = FLUSH_INSTR_NONE
    df[Keys.FLUSH_INSTRUCTION] = df[Keys.FLUSH_INSTRUCTION].fillna(FLUSH_INSTR_NONE)
    if Keys.BANDWIDTH_GiB in df.columns:
        df[Keys.BANDWIDTH_GB] = df[Keys.BANDWIDTH_GiB] * (1024**3 / 1e9)
    return df


def flatten_df(df):
    df[Keys.NUMA_MEMORY_NODES] = df[Keys.NUMA_MEMORY_NODES].transform(
        lambda x: ",".join(str(i) for i in x)
    )
    df[Keys.NUMA_TASK_NODES] = df[Keys.NUMA_TASK_NODES].transform(
        lambda x: ",".join(str(i) for i in x)
    )
    return df


def create_scalability_plot(plot_data, title, output_file, access_sizes):
    plt.figure(figsize=(10, 6))

    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "."]
    ax = plt.gca()
    for access_size in access_sizes:
        color = next(ax._get_lines.prop_cycler)["color"]
        for i, name in enumerate(plot_data.keys()):
            df = plot_data[name]
            df = df[(df[Keys.ACCESS_SIZE] == access_size)]

            plt.plot(
                df[Keys.THREAD_COUNT].tolist(),
                df[Keys.BANDWIDTH_GB].tolist(),
                marker=markers[i],
                linestyle=linestyles[i],
                color=color,
                label=f"{name} {access_size} byte",
            )

    first_df = next(iter(plot_data.values()))
    plt.xticks(first_df[Keys.THREAD_COUNT].tolist())
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Number of Threads")
    plt.ylabel("Bandwidth in GB/s")
    plt.savefig(output_file)
    plt.close()


def get_plot_filename(tag, flush_type, bm_group, numa_task_node, custom=None):
    filename_tag = ""
    if tag != "":
        filename_tag = f"{tag}_"
    filename = f"{filename_tag}{flush_type}_{bm_group}_{numa_task_node}"

    if custom is not None:
        filename = f"{filename}_{custom}"

    filename = f"{filename}.pdf"
    return filename


def create_scalability_plots(dfs, names, output_dir, access_sizes):
    df = dfs[0]  # let first result decide on fields
    bm_groups = df[Keys.BM_GROUP].unique()
    partition_counts = df[Keys.PARTITION_COUNT].unique()
    flush_types = df[Keys.FLUSH_INSTRUCTION].unique()
    tags = df[Keys.TAG].unique()
    numa_task_nodes = df[Keys.NUMA_TASK_NODES].unique()

    for tag, flush_type, partition_count, bm_group, numa_task_node in itertools.product(
        tags, flush_types, partition_counts, bm_groups, numa_task_nodes
    ):
        plot_data = {}
        for i, d in enumerate(dfs):
            df_sub = d[
                (d[Keys.BM_GROUP] == bm_group)
                & (d[Keys.PARTITION_COUNT] == partition_count)
                & (d[Keys.FLUSH_INSTRUCTION] == flush_type)
                & (d[Keys.TAG] == tag)
                & (d[Keys.NUMA_TASK_NODES] == numa_task_node)
            ]

            # Since we check for certain flush instructions, the data frame is empty for read and
            # operation latency benchmark results if the flush instruction is not `none`.
            if flush_type != FLUSH_INSTR_NONE and (
                "read" in bm_group or bm_group == "operation_latency"
            ):
                assert (
                    df_sub.empty
                ), "Flush instruction must be none for read and latency benchmarks."

            if df_sub.empty:
                continue

            plot_data[names[i]] = df_sub

        filename = get_plot_filename(tag, flush_type, bm_group, numa_task_node)
        plot_output_dir = os.path.join(output_dir, filename)
        bm_group_title = BM_Groups(bm_group).get_title()

        title = f"Scalability {bm_group_title}"
        create_scalability_plot(plot_data, title, plot_output_dir, access_sizes)


def get_output_dir():
    # get the output directory paths
    if args.output_dir is not None:
        output_dir_string = args.output_dir
    else:
        results_path = results_paths[0]
        if os.path.isfile(results_path):
            parts = results_path.rsplit("/", 1)
            assert len(parts)
            output_dir_string = parts[0]
            id = parts[1].split(".", 1)[0]
            output_dir_string = output_dir_string + "/plots/" + id
        else:
            assert os.path.isdir(results_path)
            output_dir_string = results_path + "/plots/scalability"

    print("Output directory:", output_dir_string)
    return os.path.abspath(output_dir_string)


if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results",
        nargs="+",
        type=valid_path,
        help="Paths to the results directories that should be plotted.",
    )
    parser.add_argument(
        "-n",
        "--names",
        nargs="+",
        help="Names of the graphs ordered by results order",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--access_sizes",
        nargs="+",
        help="Access sizes to plot",
        required=True,
        type=int,
    )
    parser.add_argument("-o", "--output_dir", help="path to the output directory")

    args = parser.parse_args()

    results_paths = args.results
    results_paths = list(map(format_path, results_paths))
    access_sizes = args.access_sizes

    names = args.names
    if names is None:
        names = [path.split("/")[-1] for path in results_paths]

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    keep_columns = [
        Keys.FLUSH_INSTRUCTION,
        Keys.BM_GROUP,
        Keys.TAG,
        Keys.NUMA_TASK_NODES,
        Keys.PARTITION_COUNT,
        Keys.THREAD_COUNT,
        Keys.BANDWIDTH_GB,
        Keys.ACCESS_SIZE,
    ]

    dfs = []
    for i, results in enumerate(results_paths):
        logging.info(f"Parsing {results_paths}")
        df = parse_matrix_jsons(results)
        df = flatten_df(df)
        df = df[keep_columns]
        dfs.append(df)

    create_scalability_plots(dfs, names, output_dir, access_sizes)
    logging.info(f"Plots have been written to {output_dir}.")
