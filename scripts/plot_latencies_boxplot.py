#! /usr/bin/env python3

import argparse
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument("results", help="path to the results directory")
    parser.add_argument("-o", "--output_dir", help="path to the output directory")
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
            output_dir_string = results_path + "/plots/"
    output_dir = os.path.abspath(output_dir_string)
    os.makedirs(output_dir, exist_ok=True)

if os.path.isfile(results_path):
    if not results_path.endswith(".json"):
        sys.exit("Result path is a single file but is not a .json file.")
    file_paths = [results_path]
else:
    file_paths = [path for path in glob.glob(results_path + "/*.json")]

rows = []

# Parse json
for path in file_paths:
    with open(path, "r") as file:
        data = json.load(file)[0]
        assert "benchmarks" in data
        for benchmark in data["benchmarks"]:
            config = benchmark["config"]
            latencies = benchmark["results"]["latencies"]
            if isinstance(latencies[0], list):
                latencies = latencies[0]
            for latency in latencies:
                assert len(config["m0_numa_nodes"]) == 1
                if len(config["numa_task_nodes"]) > 0:
                    pin_desc = "numa"
                    assert len(config["numa_task_nodes"]) == 1
                    thread_pin_target = config["numa_task_nodes"][0]
                elif len(config["thread_cores"]) > 0:
                    assert len(config["thread_cores"]) == 1
                    pin_desc = "core"
                    thread_pin_target = config["thread_cores"][0]
                rows.append(
                    [
                        config["exec_mode"],
                        config["operation"],
                        config["m0_numa_nodes"][0],
                        pin_desc,
                        thread_pin_target,
                        config["m0_region_size"] / 1024**3,
                        latency,
                    ]
                )
df = pd.DataFrame(
    rows,
    columns=[
        "exec_mode",
        "op",
        "memory_node",
        "thread_pinning_type",
        "thread_pin_target",
        "region_size_GiB",
        "sample_latency_ns",
    ],
)
df["exec_mode"] = df["exec_mode"].replace({"sequential_latency": "Seq", "random_latency": "Rnd"})
df["op"] = df["op"].replace({"read": "Read", "write": "Write"})
df["access_pattern"] = df["exec_mode"] + " " + df["op"]
df["placement"] = df["memory_node"].replace({0: "Remote CPU", 1: "CPU", 2: "CXL"})

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

### Boxplot
# plt.figure(figsize=(4, 2.3))
# flierprops = dict(marker='o', markersize=0.5, markerfacecolor='black', linestyle='none')
# ax = sns.boxplot(x='access_pattern', y='sample_latency_ns', hue='placement',
#         hue_order=["CPU","Remote CPU","CXL"], data=df, palette=hpi_palette,
#         flierprops=flierprops)
# ax.set_xlabel("")
# ax.set_ylabel("Latency [cycles]")
# plt.ylim(1,3000)
# ax.set_yscale('log')
# # plt.ylim(1)
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
# # ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
# # ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
# ax.yaxis.grid(which="both")
# plt.xticks(rotation=90)
# legend = ax.legend(
#     title="",
#     loc="upper center",
#     bbox_to_anchor=(1.3, 0.8),
#     ncol=1, frameon=True,
#     columnspacing=2.5,
#     handlelength=0.8,
#     handletextpad=0.3,
#     labelspacing=0.3,)

# region_size_gib = df['region_size_GiB'].unique()
# assert(len(region_size_gib) == 1)
# region_size_gib = region_size_gib[0]
# plt.tight_layout()
# plt.savefig(f"{output_dir_string}boxplot-{region_size_gib}GiB-tail_lat.pdf", bbox_inches="tight", pad_inches=0)
# plt.close("all")
# exit(1)

access_patterns = df["access_pattern"].unique()
placements = ["CPU", "Remote CPU", "CXL"]
percentiles = [25, 50, 75, 90, 95, 99]
# percentiles = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
labels = [str(p) if p in [25, 50, 75, 90, 95, 99] else "" for p in percentiles]
markers = ["o", "X", "D", "s", "*", "P"]
linestyles = ["-", "--", ":"]

fig, axes = plt.subplots(1, len(access_patterns), figsize=(3.9, 2.2), sharey=True, sharex=True)

for ax, pattern in zip(axes, access_patterns):
    print(f"###### {pattern}")
    for idx, placement in enumerate(placements):
        subset = df[(df["access_pattern"] == pattern) & (df["placement"] == placement)]
        if subset.empty:
            continue
        latencies = np.sort(subset["sample_latency_ns"])
        percentile_values = np.percentile(latencies, percentiles)
        label = placement
        percentile_strings = [f"{p}%" for p in percentiles]
        sns.lineplot(
            ax=ax,
            x=percentile_strings,
            y=percentile_values,
            markersize=3,
            color=hpi_palette[idx],
            marker=markers[idx],
            label=label,
            zorder=2,
            linewidth=1.2,
        )
        average = np.mean(latencies)
        ax.axhline(
            average,
            color=hpi_palette[len(placements) + idx],
            linestyle=linestyles[idx],
            label=f"AVG {placement}",
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

    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    # ax.axvline(x=1, color='gray', linewidth=1, linestyle='dashed', alpha=0.7, zorder=1)
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
    # plt.ylim(0,5500)
    # plt.ylim(0)

x_center = 0.565
fig.supxlabel("Percentile", fontsize=10, y=-0.02, x=x_center)
# plt.subplots_adjust(wspace=-1.5)
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    labels=labels,
    handles=handles,
    # title="Memory Location",
    title="",
    ncol=3,
    loc="upper center",
    frameon=True,
    handlelength=1,
    handletextpad=0.2,
    labelspacing=0.1,
    borderpad=0.2,
    # bbox_to_anchor=(x_center, 1.3)
    bbox_to_anchor=(x_center, 1.25),
)

region_size_gib = df["region_size_GiB"].unique()
assert len(region_size_gib) == 1
region_size_gib = region_size_gib[0]

plt.tight_layout(pad=0)
plt.savefig(f"{output_dir_string}{region_size_gib}-percentiles-lat.pdf", bbox_inches="tight", pad_inches=0)
plt.close("all")
