#! /usr/bin/env python3

# Paper: Combined Throughput Scaling

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


MAX_THREAD_COUNT = 40

# benchmark configuration names
BM_CONFIG_PARALLEL_LOCAL_DEVICE = "seq_reads_local_device"
BM_CONFIG_PARALLEL_LOCAL_REMOTE_SOCKET = "seq_reads_local_remote_socket"
BM_CONFIG_PARALLEL_LOCAL_ONLY = "seq_reads_local1_local2"

BM_SUPPORTED_CONFIGS = [
    BM_CONFIG_PARALLEL_LOCAL_DEVICE,
    BM_CONFIG_PARALLEL_LOCAL_REMOTE_SOCKET,
    BM_CONFIG_PARALLEL_LOCAL_ONLY,
]


def dir_path(path):
    """
    Checks if the given directory path is valid.

    :param path: directory path to the results folder
    :return: bool representing if path was valid.
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
        BMKeys.THREAD_CORES,
        "matrix_args.local",
        "matrix_args.device",
        "matrix_args.local1",
        "matrix_args.local2",
        "matrix_args.remote_socket",
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

    df = df[(df[BMKeys.BM_NAME].isin(BM_SUPPORTED_CONFIGS))]

    # parallel local device
    df_local_dev = df[(df[BMKeys.BM_NAME] == BM_CONFIG_PARALLEL_LOCAL_DEVICE)]
    df_local_dev = ju.flatten_nested_json_df(df_local_dev, deny_list_explosion)
    df_local_dev.to_csv("{}/{}.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_DEVICE))
    df_local_dev = df_local_dev.drop(columns=drop_columns, errors="ignore")
    df_local_dev.to_csv("{}/{}-reduced.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_DEVICE))

    # parallel local only
    df_local = df[(df[BMKeys.BM_NAME] == BM_CONFIG_PARALLEL_LOCAL_ONLY)]
    df_local = ju.flatten_nested_json_df(df_local, deny_list_explosion)
    df_local.to_csv("{}/{}.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_ONLY))
    df_local = df_local.drop(columns=drop_columns, errors="ignore")
    df_local.to_csv("{}/{}-reduced.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_ONLY))

    # parallel local remote socket
    df_local_remote_socket = df[(df[BMKeys.BM_NAME] == BM_CONFIG_PARALLEL_LOCAL_REMOTE_SOCKET)]
    df_local_remote_socket = ju.flatten_nested_json_df(df_local_remote_socket, deny_list_explosion)
    df_local_remote_socket.to_csv("{}/{}.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_REMOTE_SOCKET))
    df_local_remote_socket = df_local_remote_socket.drop(columns=drop_columns, errors="ignore")
    df_local_remote_socket.to_csv("{}/{}-reduced.csv".format(output_dir, BM_CONFIG_PARALLEL_LOCAL_REMOTE_SOCKET))

    # ------------------------------------------------------------------------------------------------------------------
    # create plots

    TAG_LOCAL_DEV = "local_device"
    TAG_LOCAL_LOCAL = "local_local"
    TAG_LOCAL_REMOTE_SOCKET = "local_remote_socket"

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

    # Workloads: local remote socket
    df_local_remote_socket["tag"] = TAG_LOCAL_REMOTE_SOCKET
    df_local_remote_socket["workload_1_bandwidth"] = df_local_remote_socket["local.results.bandwidth"]
    df_local_remote_socket["workload_2_bandwidth"] = df_local_remote_socket["remote_socket.results.bandwidth"]
    df_local_remote_socket["workload_1_thread_count"] = df_local_remote_socket["local.number_threads"]
    df_local_remote_socket["workload_2_thread_count"] = df_local_remote_socket["remote_socket.number_threads"]

    # Calculate combined bandwidth
    df = pd.concat([df_local_dev, df_local, df_local_remote_socket])
    df["combined_bandwidth"] = df["workload_1_bandwidth"] + df["workload_2_bandwidth"]
    df.to_csv("{}/{}.csv".format(output_dir, "manual_check"))

    w1_thread_counts = df["workload_1_thread_count"].unique()

    # Transform GiB/s to GB/s
    df["combined_bandwidth_gb"] = df["combined_bandwidth"] * (1024**3 / 1e9)
    df["workload_1_bandwidth_gb"] = df["workload_1_bandwidth"] * (1024**3 / 1e9)
    df["workload_2_bandwidth_gb"] = df["workload_2_bandwidth"] * (1024**3 / 1e9)

    sns.set_theme(style="ticks")
    SEPARATOR = ": "

    LABEL_LOC_DEV_W1 = "C{}Local (W1)".format(SEPARATOR)
    LABEL_LOC_DEV_W2 = "C{}Device (W2)".format(SEPARATOR)
    LABEL_LOC_LOC_W1 = "A{}Local (W1)".format(SEPARATOR)
    LABEL_LOC_LOC_W2 = "A{}Local (W2)".format(SEPARATOR)
    LABEL_LOC_REMOTE_SOCKET_W1 = "B{}Local (W1)".format(SEPARATOR)
    LABEL_LOC_REMOTE_SOCKET_W2 = "B{}Remote Socket (W2)".format(SEPARATOR)

    # Prepare subplots
    hatches = ["/", "\\", "+"]
    fig, axes = plt.subplots(1, len(w1_thread_counts), figsize=(2 * len(w1_thread_counts), 1.7), sharey=True)

    for ax, thread_count in zip(axes, w1_thread_counts):
        df_plot = df[df["workload_1_thread_count"] == thread_count]

        # Plot combined bandwidth
        df_plot["combined_bandwidth_label"] = df_plot["tag"]
        combined_bandwidth_tag_replacements = {
            TAG_LOCAL_DEV: LABEL_LOC_DEV_W2,
            TAG_LOCAL_LOCAL: LABEL_LOC_LOC_W2,
            TAG_LOCAL_REMOTE_SOCKET: LABEL_LOC_REMOTE_SOCKET_W2,
        }
        df_plot["combined_bandwidth_label"].replace(combined_bandwidth_tag_replacements, inplace=True)

        hue_order = [LABEL_LOC_LOC_W2, LABEL_LOC_REMOTE_SOCKET_W2, LABEL_LOC_DEV_W2]

        sns.barplot(
            x="workload_2_thread_count",
            y="combined_bandwidth",
            data=df_plot,
            palette="colorblind",
            hue="combined_bandwidth_label",
            hue_order=hue_order,
            ax=ax,
        )

        # Plot workload 1 bandwidth
        df_plot["workload_1_bandwidth_label"] = df_plot["tag"]
        workload_1_bandwidth_tag_replacements = {
            TAG_LOCAL_DEV: LABEL_LOC_DEV_W1,
            TAG_LOCAL_LOCAL: LABEL_LOC_LOC_W1,
            TAG_LOCAL_REMOTE_SOCKET: LABEL_LOC_REMOTE_SOCKET_W1,
        }
        df_plot["workload_1_bandwidth_label"].replace(workload_1_bandwidth_tag_replacements, inplace=True)

        hue_order_w1 = [LABEL_LOC_LOC_W1, LABEL_LOC_REMOTE_SOCKET_W1, LABEL_LOC_DEV_W1]

        sns.barplot(
            # we plot workload_1_bandwidth over combined_bandwidth. However, we still use workload 2's thread count since
            # w1's thread count is assumed to be fixed.
            x="workload_2_thread_count",
            y="workload_1_bandwidth",
            data=df_plot,
            palette="dark",
            hue="workload_1_bandwidth_label",
            hue_order=hue_order_w1,
            ax=ax,
        )

        ax.yaxis.grid()
        if y_tick_distance is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_distance))
            # Add y-axis label at every 2nd tick
            label_distance = 2 * y_tick_distance
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % label_distance == 0 else "")
            )

        ax.set_title(f"#Threads W1: {thread_count}")
        if ax != axes[0]:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove legend for each subplot
        ax.get_legend().remove()

    handles, labels = ax.get_legend_handles_labels()

    # Define the order of the handles based on the new desired order
    ordered_labels = [
        LABEL_LOC_LOC_W1,
        LABEL_LOC_LOC_W2,
        LABEL_LOC_REMOTE_SOCKET_W1,
        LABEL_LOC_REMOTE_SOCKET_W2,
        LABEL_LOC_DEV_W1,
        LABEL_LOC_DEV_W2,
    ]

    ordered_handles = [handles[labels.index(label)] for label in ordered_labels]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=6,
        frameon=False,
        handlelength=0.8,  # reduce handle size
        handletextpad=0.3,  # reduce space between handle and label
    )

    fig.text(0.55, 0.03, "Thread Count (W2)", ha="center")
    fig.text(0.0, 0.5, "Throughput [GB/s]", va="center", rotation="vertical")

    plt.tight_layout()

    fig.savefig(
        "{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, "combined_throughput_scaling"),
        bbox_inches="tight",
        pad_inches=0,
    )
