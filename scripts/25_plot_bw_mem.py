#! /usr/bin/env python3

import argparse
import cxlbenchplot as mplt
import glob
import json_util as ju
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
import sys

from enums.benchmark_keys import BMKeys
from enums.file_names import FILE_TAG_SUBSTRING, PLOT_FILE_PREFIX

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


def get_load_config(path):
    if "emr-bw-cxl-dax_none" in path:
        return "EMR w/o GNR load"
    elif "emr-bw-cxl-dax_25-gnr" in path:
        return "EMR w/ GNR load"
    else:
        return ""

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("results", nargs="+", help="paths to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    args = parser.parse_args()

    results_paths = args.results
    for results_path in results_paths:
        if not results_path.startswith("./") and not results_path.startswith("/"):
            results_path = "./" + results_path

    output_dir_string = None

    # get the output directory paths
    if args.output_dir is not None:
        output_dir_string = args.output_dir
    else:
        if os.path.isfile(results_paths[0]):
            parts = results_paths[0].rsplit("/", 1)
            assert len(parts)
            output_dir_string = parts[0]
            output_dir_string = output_dir_string + "/plots/"
        else:
            assert os.path.isdir(results_paths[0])
            output_dir_string = results_paths[0] + "/plots/"
    output_dir = os.path.abspath(output_dir_string)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    file_paths = []
    for results_path in results_paths:
        if os.path.isfile(results_path):
            if not results_path.endswith(".json"):
                sys.exit("Result path is a single file but is not a .json file.")
            file_paths.append(results_path)
        else:
            for path in glob.glob(results_path + "/*.json"):
                file_paths.append(path)

    print("File paths", file_paths)
    # ------------------------------------------------------------------------------------------------------------------

    dfs = []
    for path in file_paths:
        load_config = get_load_config(path)
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
        df["load_config"] = load_config
        dfs.append(df)

    df = pd.concat(dfs)

    # -------------------------------------------------------------------------------------------------------------------

    assert not df.empty, "DataFrame is empty"
    deny_explosion_list = [
        BMKeys.MATRIX_ARGS,
        BMKeys.THREADS_LEVELED,
        BMKeys.EXPLODED_NUMA_MEMORY_NODES_M0,
        BMKeys.EXPLODED_NUMA_MEMORY_NODES_M1,
        BMKeys.EXPLODED_NUMA_TASK_NODES,
        BMKeys.EXPLODED_NUMA_MEMORY_NODES,
        BMKeys.EXPLODED_THREAD_CORES,
    ]
    df = ju.flatten_nested_json_df(df, deny_explosion_list)
    if BMKeys.BANDWIDTH_GiB in df.columns:
        df[BMKeys.BANDWIDTH_GB] = df[BMKeys.BANDWIDTH_GiB] * (1024**3 / 1e9)
    df.to_csv("{}/data.csv".format(output_dir))
    df[BMKeys.NUMA_MEMORY_NODES_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].apply(mplt.values_as_string)
    drop_columns = [
        "index",
        "bm_type",
        "compiler",
        "git_hash",
        "hostname",
        "matrix_args",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")
    operations = df[BMKeys.OPERATION].unique()
    modes = df[BMKeys.EXEC_MODE].unique()
    print("Existing operations: {}".format(operations))
    print("Existing modes: {}".format(modes))
    df["mode_op"] = df[BMKeys.EXEC_MODE].str.capitalize() + "\n" + df[BMKeys.OPERATION]
    df.to_csv("{}/data-reduced.csv".format(output_dir))
    print(df)

    # ------------------------------------------------------------------------------------------------------------------
    # create plots

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 150)
    # print(df[[BMKeys.THREAD_COUNT, BMKeys.BANDWIDTH_GB, BMKeys.CACHE_INSTRUCTION, BMKeys.BM_NAME]])
    # exit()

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{libertine}"})

    hpi_col = [
        "#f5a700",
        "#dc6007",
        "#b00539",
        "#6b009c",
        "#006d5b",
        "#0073e6",
        "#e6007a",
        "#00C800",
        "#FFD500",
        "#0033A0",
    ]


# --------------------------------------------------------------------------------------------------------
style_key = BMKeys.CACHE_INSTRUCTION
style_key = "load_config"

def plot(df):
    print("columns:", df["mode_op"].unique())
    col_num = 4
    g = sns.relplot(
        data=df,
        x=BMKeys.THREAD_COUNT,
        y=BMKeys.BANDWIDTH_GB,
        hue=style_key,
        style=style_key,
        markers=True,
        markersize=4,
        col="mode_op",
        col_order=['Sequential\nread', 'Random\nread', 'Sequential\nstream-read', 'Random\nstream-read',
          'Sequential\nwrite', 'Random\nwrite', 'Sequential\nstream-write', 'Random\nstream-write'],
        kind="line",
        col_wrap=col_num,
        height=1.5,
        aspect=0.3,
        facet_kws={"sharex": False, "sharey": True},
        palette=hpi_col
    )
    thread_counts = df[BMKeys.THREAD_COUNT].unique()
    thread_counts.sort()
    g.set_ylabels("Throughput [GB/s]")
    g.set_xlabels("\# Threads")
    major_locator = 5
    minor_locator = 2.5
    for ax in g.axes.flat:
        ymax_list = [line.get_ydata().max() for line in ax.lines if len(line.get_ydata()) > 0]
        max_y = max(ymax_list) if ymax_list else 0
        print(max_y)
        if max_y < 5:
            major_locator = 1
            minor_locator = 0.5
        ax.set_xticks(thread_counts)
        ax.set_xticklabels(thread_counts)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_locator))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_locator))
        ax.grid(axis="both", which="major", alpha=0.7, zorder=1)
        ax.grid(axis="both", which="minor", alpha=0.2, zorder=1)
        ax.set_ylim(bottom=0)
        title = ax.get_title()
        clean_title = title.split(" = ")[-1]
        ax.set_title(clean_title)

    # if subplot height is too small, seaborn / matplotlib override the sharey=True. Manually ensure that only most left subplots has
    # y axis lables.
    for ax_id, ax in enumerate(g.axes.flat):
        ax.set_xlabel("\# Threads")
        # ax.tick_params(labelbottom=True)
        if ax_id % col_num != 0:
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_ylabel("")
        else:
            ax.yaxis.set_tick_params(labelleft=True)

    g.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.2, hspace=0.8)

    sns.move_legend(
        g,
        title="",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.23),
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=1,
        frameon=True,
        borderpad=0.2,
    )

    region_size_gib = df["region_size_GiB"].unique()
    assert len(region_size_gib) == 1
    region_size_gib = region_size_gib[0]
    # plt.tight_layout(pad=0)

    g.savefig(
        f"{output_dir}/{region_size_gib}-bw-lines.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )


# --------------------------------------------------------------------------------------------------------

df = df[df[BMKeys.THREAD_COUNT] <= 96]
df = df.sort_values(by=[BMKeys.THREAD_COUNT], ascending=[True])
df[BMKeys.THREAD_COUNT] = df[BMKeys.THREAD_COUNT].astype(str)
df["region_size_GiB"] = df["m0_region_size"] / 1024**3
region_sizes_gib = df["region_size_GiB"].unique()
for size in region_sizes_gib:
    df_plot = df[df["region_size_GiB"] == size]
    plot(df_plot)
