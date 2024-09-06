#! /usr/bin/env python3

# basis: plot_latency_percentiles.py

import argparse
import glob
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys


def get_load_config(path):
    if "emr-lat-cxl-dax_none" in path:
        return "EMR w/o GNR load"
    elif "emr-lat-cxl-dax_25-gnr" in path:
        return "EMR w/ GNR load"
    else:
        return ""


if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
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

rows = []

# Parse json
for path in file_paths:
    load_config = get_load_config(path)
    with open(path, "r") as file:
        results = json.load(file)
        for result in results:
            assert "benchmarks" in result
            for benchmark in result["benchmarks"]:
                config = benchmark["config"]
                # print(config)
                latencies = benchmark["results"]["latencies"]
                if isinstance(latencies[0], list):
                    latencies = latencies[0]
                for latency in latencies:
                    memory_interface = "unknown"
                    memory_source = "unknown"
                    if len(config["m0_numa_nodes"]) > 0:
                        assert len(config["m0_numa_nodes"]) == 1
                        memory_interface = "NUMA"
                        memory_source = str(config["m0_numa_nodes"][0])
                    if len(config["m0_device_path"]) > 0:
                        assert memory_interface == "unknown"
                        memory_interface = "DAX"
                        memory_source = config["m0_device_path"]
                    if len(config["numa_task_nodes"]) > 0:
                        pin_desc = "numa"
                        assert len(config["numa_task_nodes"]) == 1
                        thread_pin_target = config["numa_task_nodes"][0]
                    elif len(config["thread_cores"]) > 0:
                        assert len(config["thread_cores"]) == 1
                        pin_desc = "core"
                        thread_pin_target = config["thread_cores"][0]
                    assert "access_size" in config
                    rows.append(
                        [
                            config["exec_mode"],
                            config["operation"],
                            config["cache_instruction"],
                            config["access_size"],
                            memory_interface,
                            memory_source,
                            pin_desc,
                            thread_pin_target,
                            config["m0_region_size"] / 1024**3,
                            latency,
                            load_config,
                        ]
                    )
df = pd.DataFrame(
    rows,
    columns=[
        "exec_mode",
        "op",
        "cache_instr",
        "access_size",
        "memory_interface",
        "memory_source",
        "thread_pinning_type",
        "thread_pin_target",
        "region_size_GiB",
        "sample_latency_ns",
        "load_config",
    ],
)
df["exec_mode"] = df["exec_mode"].replace({"sequential_latency": "Seq'", "random_latency": "Random", "latency": ""})
df["op"] = df["op"].replace({"compare-and-swap": "CAS", "fetch-and-add": "FAA"})
print(df["exec_mode"].unique())
# df["op"] = df["op"].replace({"read": "Read", "write": "Write"})
# df["access_pattern"] = df["exec_mode"] + " " + df["access_size"].astype(str) + "B\n" + df["op"] + "\n" + df["cache_instr"]
df["access_pattern"] = df["exec_mode"] + "\n" + df["op"] + " " + df["cache_instr"]
df["access_pattern"] = df["access_pattern"].str.replace("write-back", "(clwb)", regex=False)
df["access_pattern"] = df["access_pattern"].str.replace("flush-opt", "(flush opt)", regex=False)
df["access_pattern"] = df["access_pattern"].str.replace("none", "", regex=False)
df["placement"] = df["memory_interface"] + " " + df["memory_source"]
df["placement"] = df["placement"].replace({"unknown unknown": "CXL"})
df.to_csv(f"{output_dir_string}data.csv")

# Plot
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{libertine}"})

hpi_palette = [
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
def plot(df, suffix):
    df = df.sort_values(by=["access_pattern"])
    access_patterns = df["access_pattern"].unique()
    placements = df["placement"].unique()
    load_configs = df["load_config"].unique()
    percentiles = [25, 50, 75, 90, 95, 99]
    # percentiles.append(99.9)
    # percentiles = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    labels = [str(p) if p in [25, 50, 75, 90, 95, 99] else "" for p in percentiles]
    markers = ["o", "X", "D", "s", "*", "P"]
    linestyles = ["-", "--", ":"]

    patterns_count = len(access_patterns)
    if patterns_count < 3:
      column_width = 1
    else:
      column_width = 0.85
    row_count = 1
    col_count = math.ceil(patterns_count / row_count)
    fig, axes = plt.subplots(
        row_count, col_count, figsize=(col_count * column_width, row_count * 1.8), sharey=True, sharex=False
    )
    axes = axes.flatten()

    for ax, pattern in zip(axes, access_patterns):
        print(f"###### {pattern}")
        for p_idx, placement in enumerate(placements):
            for l_idx, load_config in enumerate(load_configs):
                style_idx = p_idx * len(placements) + l_idx
                subset = df[
                    (df["access_pattern"] == pattern)
                    & (df["placement"] == placement)
                    & (df["load_config"] == load_config)
                ]
                print(subset)
                if subset.empty:
                    continue
                latencies = np.sort(subset["sample_latency_ns"])
                percentile_values = np.percentile(latencies, percentiles)
                # label = placement + " " + load_config
                label = load_config
                percentile_strings = [f"{p}%" for p in percentiles]
                sns.lineplot(
                    ax=ax,
                    x=percentile_strings,
                    y=percentile_values,
                    markersize=3,
                    color=hpi_palette[style_idx],
                    marker=markers[style_idx],
                    label=label,
                    zorder=2,
                    linewidth=1.2,
                )
                average = np.mean(latencies)
                ax.axhline(
                    average,
                    color=hpi_palette[len(placements) + style_idx],
                    linestyle=linestyles[style_idx],
                    label=f"AVG {label}",
                    linewidth=1,
                    zorder=1,
                )

                for artist in ax.lines:
                    artist.set_markeredgewidth(0.3)  # Thinner white border
                    artist.set_markeredgecolor("white")

                print(placement)
                print(f"avg: {average}")
                for i, val in enumerate(percentiles):
                    print(f"{percentiles[i]}%: {percentile_values[i]}")

        # TODO(MW) comment in again?
        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
        # # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        # # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        # # ax.axvline(x=1, color='gray', linewidth=1, linestyle='dashed', alpha=0.7, zorder=1)
        ax.grid(axis="y", which="minor", linestyle="-", alpha=0.2)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
        # ax.set_yscale('log')
        # custom_ticks = [10, 100, 1000]
        # ax.set_yticks(custom_ticks)
        # ax.set_yticklabels([str(t) for t in custom_ticks])
        # ax.grid(axis='y', which='minor', linestyle='-', alpha=0.2)
        # ax.grid(axis='y', which='major', linestyle='-', alpha=0.6)
        ax.set_title(pattern, pad=4, fontsize=10)
        ax.tick_params(axis="x", labelrotation=90)
        # plt.ticklabels(labels)

        if ax == axes[0]:
            # ax.set_ylabel("Latency [Cycles]")
            ax.set_ylabel("Latency [ns]")
        else:
            ax.set_ylabel(
                "",
            )
        ax.get_legend().remove()
        plt.ylim(0, 1500)
        # plt.ylim(0)

    for ax in axes[patterns_count:]:
        ax.set_visible(False)

    x_center = 0.57
    fig.supxlabel("Percentile", fontsize=10, y=-0.02, x=x_center)
    # plt.subplots_adjust(wspace=-1.5)
    handles, labels = axes[0].get_legend_handles_labels()

    ordered_labels = labels
    ordered_handles = handles
    # desired_order = ['CPU', 'Remote CPU', 'CXL', 'AVG CPU', 'AVG Remote CPU', 'AVG CXL']
    # label_to_handle = dict(zip(labels, handles))
    # ordered_labels = [label for label in desired_order if label in label_to_handle]
    # ordered_handles = [label_to_handle[label] for label in ordered_labels]

    def replace_second_space(label):
        parts = label.split(" ", 2)
        if len(parts) > 2:
            return parts[0] + " " + parts[1] + "\n" + parts[2]
        elif len(parts) > 1:
            return parts[0] + "\n" + parts[1]
        else:
            return label

    anchor = (0.56, 1.27)
    legend_columns = 2
    labelspacing=0.1
    if patterns_count < 3:
        anchor = (1.28, 0.95)
        legend_columns = 1
        labelspacing=0.8
        ordered_labels = [replace_second_space(label) for label in ordered_labels]

    fig.legend(
        labels=ordered_labels,
        handles=ordered_handles,
        title="",
        ncol=legend_columns,
        loc="upper center",
        frameon=True,
        handlelength=1,
        handletextpad=0.2,
        labelspacing=labelspacing,
        borderpad=0.2,
        bbox_to_anchor=anchor,
    )

    region_size_gib = df["region_size_GiB"].unique()
    assert len(region_size_gib) == 1
    region_size_gib = region_size_gib[0]

    plt.tight_layout(pad=0)
    plt.savefig(f"{output_dir_string}{region_size_gib}-percentiles-lat{suffix}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close("all")


# --------------------------------------------------------------------------------------------------------

# df = df[df["exec_mode"] == "Random"]
# df = df[df["cache_instr"].isin(["write-back", "none"])]
# df = df[df["exec_mode"].isin(["Rnd"])]
region_sizes_gib = df["region_size_GiB"].unique()

for size in region_sizes_gib:
    df_plot = df[df["region_size_GiB"] == size]
    read_write_df = df_plot[~df_plot["op"].isin(["FAA", "CAS"])]
    plot(read_write_df, "-read-write")

    atomics_df = df_plot[df_plot["op"].isin(["FAA", "CAS"])]
    plot(atomics_df, "-atomics")
