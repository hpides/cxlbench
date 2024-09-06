"""
This module plots multiple results into one plot
(throughput x thread count) for a given list of
access sizes.
"""

import argparse
import itertools
import logging

import os
import sys

from matplotlib import pyplot as plt

from compare import valid_path
from enums.benchmark_keys import BMKeys
from enums.benchmark_groups import BMGroups
from json_util import parse_matrix_jsons
from cxlbenchplot import FLUSH_INSTR_NONE

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout,
)


def format_path(results_path):
    if not results_path.startswith("./") and not results_path.startswith("/"):
        return "./" + results_path
    return results_path


def create_scalability_plot(plot_data, title, output_file, access_sizes):
    plt.figure(figsize=(10, 6))

    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "."]
    ax = plt.gca()
    for access_size in access_sizes:
        color = next(ax._get_lines.prop_cycler)["color"]
        for i, name in enumerate(plot_data.keys()):
            df = plot_data[name]
            df = df[(df[BMKeys.ACCESS_SIZE] == access_size)]

            plt.plot(
                df[BMKeys.THREAD_COUNT].tolist(),
                df[BMKeys.BANDWIDTH_GB].tolist(),
                marker=markers[i],
                linestyle=linestyles[i],
                color=color,
                label=f"{name} {access_size} byte",
            )

    first_df = next(iter(plot_data.values()))
    plt.xticks(first_df[BMKeys.THREAD_COUNT].tolist())
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
    bm_groups = df[BMKeys.BM_GROUP].unique()
    partition_counts = df[BMKeys.PARTITION_COUNT].unique()
    flush_types = df[BMKeys.FLUSH_INSTRUCTION].unique()
    tags = df[BMKeys.TAG].unique()
    numa_task_nodes = df[BMKeys.NUMA_TASK_NODES].unique()

    for tag, flush_type, partition_count, bm_group, numa_task_node in itertools.product(
        tags, flush_types, partition_counts, bm_groups, numa_task_nodes
    ):
        plot_data = {}
        for i, d in enumerate(dfs):
            df_sub = d[
                (d[BMKeys.BM_GROUP] == bm_group)
                & (d[BMKeys.PARTITION_COUNT] == partition_count)
                & (d[BMKeys.FLUSH_INSTRUCTION] == flush_type)
                & (d[BMKeys.TAG] == tag)
                & (d[BMKeys.NUMA_TASK_NODES] == numa_task_node)
            ]

            # Since we check for certain flush instructions, the data frame is empty for read and
            # operation latency benchmark results if the flush instruction is not `none`.
            if flush_type != FLUSH_INSTR_NONE and ("read" in bm_group or bm_group == "operation_latency"):
                assert df_sub.empty, "Flush instruction must be none for read and latency benchmarks."

            if df_sub.empty:
                continue

            plot_data[names[i]] = df_sub

        filename = get_plot_filename(tag, flush_type, bm_group, numa_task_node)
        plot_output_dir = os.path.join(output_dir, filename)
        bm_group_title = BMGroups(bm_group).get_title()

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
        BMKeys.FLUSH_INSTRUCTION,
        BMKeys.BM_GROUP,
        BMKeys.TAG,
        BMKeys.NUMA_TASK_NODES,
        BMKeys.PARTITION_COUNT,
        BMKeys.THREAD_COUNT,
        BMKeys.BANDWIDTH_GB,
        BMKeys.ACCESS_SIZE,
    ]

    supported_bm_groups = [
        BMGroups.SEQUENTIAL_WRITES,
        BMGroups.RANDOM_WRITES,
        BMGroups.RANDOM_READS,
        BMGroups.SEQUENTIAL_READS,
    ]

    dfs = []
    for i, results in enumerate(results_paths):
        logging.info(f"Parsing {results_paths}")
        df = parse_matrix_jsons(results, supported_bm_groups)
        df = df[keep_columns]
        dfs.append(df)

    create_scalability_plots(dfs, names, output_dir, access_sizes)
    logging.info(f"Plots have been written to {output_dir}.")
