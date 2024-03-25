#! /usr/bin/env python3

# Paper: Combined Throughput Scaling

import argparse
import glob
import json_util as ju
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
import sys

KEY_ACCESS_SIZE = "access_size"
KEY_BANDWIDTH_GiB = "bandwidth"
KEY_BANDWIDTH_GB = "bandwidth_gb"
KEY_BM_NAME = "bm_name"
KEY_BM_TYPE = "bm_type"
KEY_CHUNK_SIZE = "min_io_chunk_size"
KEY_CUSTOM_OPS = "custom_operations"
KEY_EXEC_MODE = "exec_mode"
KEY_EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"
KEY_EXPLODED_NUMA_TASK_NODES = "benchmarks.config.numa_task_nodes"
KEY_LAT_AVG = "latency.avg"
KEY_MATRIX_ARGS = "matrix_args"
KEY_MEMORY_REGION_SIZE = "memory_region_size"
KEY_NUMA_TASK_NODES = "numa_task_nodes"
KEY_NUMA_MEMORY_NODES = "numa_memory_nodes"
KEY_OPERATION = "operation"
KEY_OPERATION_COUNT = "number_operations"
KEY_PARTITION_COUNT = "number_partitions"
KEY_RANDOM_DISTRIBUTION = "random_distribution"
KEY_RUN_TIME = "run_time"
KEY_SUB_BM_NAMES = "sub_bm_names"
KEY_TAG = "tag"
KEY_THREAD_COUNT = "number_threads"
KEY_THREADS = "threads"
KEY_THREADS_LEVELED = "benchmarks.results.threads"
KEY_FLUSH_INSTRUCTION = "flush_instruction"
FLUSH_INSTR_NONE = "none"

DATA_FILE_PREFIX = "data_"
PLOT_FILE_PREFIX = "plot_"
FILE_TAG_SUBSTRING = "TAG_"
MAX_THREAD_COUNT = 40

# benchmark configuration names
BM_CONFIG_PARALLEL_LOCAL_DEVICE = "seq_reads_local_device"
BM_CONFIG_PARALLEL_LOCAL_ONLY = "seq_reads_local1_local2"

BM_SUPPORTED_CONFIGS = [BM_CONFIG_PARALLEL_LOCAL_DEVICE, BM_CONFIG_PARALLEL_LOCAL_ONLY]

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

    print("Output directory:", output_dir_string)
    output_dir = os.path.abspath(output_dir_string)
    results = args.results
    if args.memory_nodes is not None:
        memory_nodes = args.memory_nodes
    else:
        memory_nodes = []

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
            assert "_" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            tag = tag_part.split(".")[0]

        df = pd.read_json(path)
        df[KEY_TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    bm_names = df[KEY_BM_NAME].unique()
    print("Existing BM groups: {}".format(bm_names))

    # ------------------------------------------------------------------------------------------------------------------
    deny_list_explosion = [
        KEY_MATRIX_ARGS,
        KEY_THREADS,
        KEY_NUMA_TASK_NODES,
        KEY_NUMA_MEMORY_NODES,
        "matrix_args.local",
        "matrix_args.device",
        "matrix_args.local1",
        "matrix_args.local2",
        "sub_bm_names",
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

    df = df[(df[KEY_BM_NAME].isin(BM_SUPPORTED_CONFIGS))]

    # parallel local device
    df_local_dev = df[(df[KEY_BM_NAME] == BM_CONFIG_PARALLEL_LOCAL_DEVICE)]
    df_local_dev = ju.flatten_nested_json_df(df_local_dev, deny_list_explosion)
    df_local_dev.to_csv("{}/{}.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_DEVICE))
    df_local_dev = df_local_dev.drop(columns=drop_columns, errors="ignore")
    df_local_dev.to_csv("{}/{}-reduced.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_DEVICE))

    # parallel local only
    df_local = df[(df[KEY_BM_NAME] == BM_CONFIG_PARALLEL_LOCAL_ONLY)]
    df_local = ju.flatten_nested_json_df(df_local, deny_list_explosion)
    df_local.to_csv("{}/{}.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_ONLY))
    df_local = df_local.drop(columns=drop_columns, errors="ignore")
    df_local.to_csv("{}/{}-reduced.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_ONLY))

    # ------------------------------------------------------------------------------------------------------------------
    # create plots

    TAG_LOCAL_DEV = "local_device"
    TAG_LOCAL_LOCAL = "local_local"

    # We plot combined_bandwidth first and add workload_1_bandwidth afterwards. workload_1_bandwidth is drawn over
    # combined_bandwidth. combined_bandwidth is visualized as the delta of combined_bandwidth - workload_1_bandwidth.
    # Thus, the label for the combined_bandwidth should be a description of workload 2.

    # Prepare dataframes
    # - parallel
    # worklaods: local device
    df_local_dev["tag"] = TAG_LOCAL_DEV
    df_local_dev["workload_1_bandwidth"] = df_local_dev["local.results.bandwidth"]
    df_local_dev["workload_2_bandwidth"] = df_local_dev["device.results.bandwidth"]
    df_local_dev["workload_1_thread_count"] = df_local_dev["local.number_threads"]
    df_local_dev["workload_2_thread_count"] = df_local_dev["device.number_threads"]

    # Workloads: local local
    df_local["tag"] = TAG_LOCAL_LOCAL
    df_local["workload_1_bandwidth"] = df_local["local1.results.bandwidth"]
    df_local["workload_2_bandwidth"] = df_local["local2.results.bandwidth"]
    df_local["workload_1_thread_count"] = df_local["local1.number_threads"]
    df_local["workload_2_thread_count"] = df_local["local2.number_threads"]

    # Calculate combined bandwidth
    df = pd.concat([df_local_dev, df_local])
    df["combined_bandwidth"] = df["workload_1_bandwidth"] + df["workload_2_bandwidth"]
    assert len(df["workload_1_thread_count"].unique()) == 1

    # Transform GiB/s to GB/s
    df["combined_bandwidth_gb"] = df["combined_bandwidth"] * (1024**3 / 1e9)
    df["workload_1_bandwidth_gb"] = df["workload_1_bandwidth"] * (1024**3 / 1e9)
    df["workload_2_bandwidth_gb"] = df["workload_2_bandwidth"] * (1024**3 / 1e9)

    sns.set(style="ticks")
    # hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]
    SEPARATOR = ": "

    LABEL_LOC_DEV_W1 = "A{}Local (W1)".format(SEPARATOR)
    LABEL_LOC_DEV_W2 = "A{}Device (W2)".format(SEPARATOR)
    LABEL_LOC_LOC_W1 = "B{}Local (W1)".format(SEPARATOR)
    LABEL_LOC_LOC_W2 = "B{}Local (W2)".format(SEPARATOR)

    # Plot combined bandwidth
    df["combined_bandwidth_label"] = df["tag"]
    combined_bandwidth_tag_replacements = {
        TAG_LOCAL_DEV: LABEL_LOC_DEV_W2,
        TAG_LOCAL_LOCAL: LABEL_LOC_LOC_W2,
    }
    df["combined_bandwidth_label"].replace(combined_bandwidth_tag_replacements, inplace=True)

    plt.figure(figsize=(5.5, 2.8))

    barplot = sns.barplot(
        x="workload_2_thread_count",
        y="combined_bandwidth",
        data=df,
        # palette=hpi_palette,
        palette="colorblind",
        hue="combined_bandwidth_label",
    )

    # Plot workload 1 bandwidth
    df["workload_1_bandwidth_label"] = df["tag"]
    workload_1_bandwidth_tag_replacements = {
        TAG_LOCAL_DEV: LABEL_LOC_DEV_W1,
        TAG_LOCAL_LOCAL: LABEL_LOC_LOC_W1,
    }
    df["workload_1_bandwidth_label"].replace(workload_1_bandwidth_tag_replacements, inplace=True)

    barplot = sns.barplot(
        # we plot workload_1_bandwidth over combined_bandwidth. However, we still use workload 2's thread count since
        # w1's thread count is assumed to be fixed.
        x="workload_2_thread_count",
        y="workload_1_bandwidth",
        data=df,
        #   color=hpi_palette[2],
        palette="dark",
        hue="workload_1_bandwidth_label",
    )

    barplot.yaxis.grid()
    if y_tick_distance is not None:
        barplot.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_distance))

    fig = barplot.get_figure()

    plt.xlabel("Thread Count (W2)")
    plt.ylabel("Throughput [GB/s]")

    legend = plt.legend()

    # Two legends
    # legend_handles = [[None, None], [None, None]]
    # legend_indexes = {
    #     LABEL_LOC_DEV_W1: 0,
    #     LABEL_LOC_DEV_W2: 0,
    #     LABEL_LOC_LOC_W1: 1,
    #     LABEL_LOC_LOC_W2: 1,
    # }
    # handle_indexes = {
    #     LABEL_LOC_DEV_W1: 0,
    #     LABEL_LOC_DEV_W2: 1,
    #     LABEL_LOC_LOC_W1: 0,
    #     LABEL_LOC_LOC_W2: 1,
    # }

    # one legend
    handles = [None, None, None, None]
    handle_indexes = {
        LABEL_LOC_DEV_W1: 0,
        LABEL_LOC_DEV_W2: 1,
        LABEL_LOC_LOC_W1: 2,
        LABEL_LOC_LOC_W2: 3,
    }

    for handle, label in zip(legend.legend_handles, legend.get_texts()):
        if label.get_text() in handle_indexes:
            handles[handle_indexes[label.get_text()]] = mpatches.Patch(
                color=handle.get_facecolor(), label=label.get_text()
            )
            continue

        if not label.get_text().startswith("None"):
            handles.append(mpatches.Patch(color=handle.get_facecolor(), label=label.get_text()))

    legend = plt.legend(
        # legend above plot
        # loc="upper center",
        # bbox_to_anchor=(0.5, 1.35),
        # ncol=2,
        # legend right outside plot
        # legend on the right
        loc="upper left",
        bbox_to_anchor=(0.98, 1),
        ncol=1,
        handles=handles,
        handlelength=0.8,
        columnspacing=1,
        handletextpad=0.3,
        frameon=False,
    )

    plt.tight_layout()

    fig.savefig(
        "{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "combined_throughput_scaling"),
        bbox_inches="tight",
        pad_inches=0,
    )
