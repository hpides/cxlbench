#! /usr/bin/env python3

import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os
import seaborn as sns
import sys


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
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument("json_results", type=valid_path, help="path to the json_results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
    parser.add_argument("-y", "--y_tick_distance", help="distance between y-ticks")
    args = parser.parse_args()

    results_path = args.json_results
    if not results_path.startswith("./") and not results_path.startswith("/"):
        results_path = "./" + results_path

    output_dir_string = None

    # Get the output directory paths
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
    json_results = args.json_results
    os.makedirs(output_dir, exist_ok=True)

    y_tick_distance = None
    if args.y_tick_distance is not None:
        y_tick_distance = int(args.y_tick_distance)

    # Get json files
    json_file_paths = None
    if os.path.isfile(json_results):
        if not json_results.endswith(".json"):
            sys.exit("Result path is a single file but is not a .json file.")
        json_file_paths = [json_results]
    else:
        json_file_paths = [path for path in glob.glob(json_results + "/*.json")]

    # Jsons to DF
    dfs = []
    for file_path in json_file_paths:
        df = pd.read_json(file_path)
        meta_fields = [
            "thread_count",
            "size_factor_per_thread",
            "op_count",
        ]
        df = pd.json_normalize(df["results"], ["durations"], meta_fields)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        df.rename(columns={0: "duration_ns"}, inplace=True)
        df["file"] = file_path
        dfs.append(df)
    df = pd.concat(dfs)
    df["op_latency_ns"] = df["duration_ns"] / df["op_count"]
    df["duration_ms"] = df["duration_ns"] / 1e6
    df["Access offset in Byte"] = df["size_factor_per_thread"] * 8
    df.to_csv("{}/{}.csv".format(output_dir, "false_sharing"))

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(2.5, 4))
    barplot = sns.barplot(
        x="thread_count",
        y="op_latency_ns",
        data=df,
        # palette=hpi_palette,
        palette="colorblind",
        hue="Access offset in Byte",
    )

    if y_tick_distance is not None:
        barplot.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_distance))

    sns.move_legend(
        barplot,
        "lower center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.5,
    )
    plt.xlabel("Thread Count")
    plt.ylabel("Average operation latency (ns)")
    # plt.yscale('log')
    plt.tight_layout()
    fig = barplot.get_figure()
    fig.savefig(
        "{}/{}.pdf".format(output_dir, "false_sharing"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    sys.exit()
