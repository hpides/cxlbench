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

def get_system_id(path):
    if "emr" in path:
        return "EMR"
    elif "gnr" in path:
        return "GNR"
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
    with open(path, 'r') as file:
        system_id = get_system_id(path)
        results = json.load(file)
        for result in results:
            assert 'benchmarks' in result
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
                        assert(len(config["numa_task_nodes"]) == 1)
                        thread_pin_target = config["numa_task_nodes"][0]
                    elif len(config["thread_cores"])>0:
                        assert(len(config["thread_cores"]) == 1)
                        pin_desc = "core"
                        thread_pin_target = config["thread_cores"][0]
                    assert "access_size" in config
                    rows.append([config["exec_mode"], config["operation"], config["cache_instruction"], config["access_size"], memory_interface, memory_source, pin_desc, thread_pin_target,
                        config["m0_region_size"]/1024**3, latency, system_id])
df = pd.DataFrame(rows, columns=['exec_mode', 'op', 'cache_instr', 'access_size', 'memory_interface', 'memory_source', 'thread_pinning_type', 'thread_pin_target', 'region_size_GiB', 'sample_latency_ns', 'system'])
df["exec_mode"] = df["exec_mode"].replace({"sequential_latency": "Seq", "random_latency": "Rnd"})
# df["op"] = df["op"].replace({"read": "Read", "write": "Write"})
# df["access_pattern"] = df["exec_mode"] + " " + df["access_size"].astype(str) + "B\n" + df["op"] + "\n" + df["cache_instr"]
df["access_pattern"] = df["exec_mode"] + " " + df["op"] + "\n" + df["cache_instr"]
df["access_pattern"] = df["access_pattern"].str.replace("write-back", "(write back)", regex=False)
df["access_pattern"] = df["access_pattern"].str.replace("flush-opt", "(flush opt)", regex=False)
df["access_pattern"] = df["access_pattern"].str.replace("none", "", regex=False)
df["placement"] = df['memory_interface'] + " " + df['memory_source']
df["placement"] = df["placement"].replace({"unknown unknown": "CXL"})
df["latency_s"] = df["sample_latency_ns"] / 10**9
df = df.sort_values(by=["region_size_GiB"], ascending=[True])
df["region_size_GiB"] = df["region_size_GiB"].astype(int)
# df["region_size_GiB"] = df["region_size_GiB"].astype(str)
df.to_csv(f"{output_dir_string}data.csv")
df["op"] = df["op"].str.replace("memory-", "", regex=False)
df["op"] = df["op"].str.replace("map-", "map ", regex=False)
df["op"] = df["op"].str.replace("unmap-", "unmap ", regex=False)
df.loc[df["placement"].str.contains("DAX", na=False), "placement"] = "CXL (devdax)"
df.loc[df["placement"].str.contains("NUMA", na=False), "placement"] = "Local (anonym')"
df["placement-op"] = df["placement"] + " " + df["op"]

# Plot
plt.rcParams.update({
  'text.usetex': True,
  'font.family': 'serif',
  'text.latex.preamble': r'\usepackage{libertine}'
})

hpi_palette = ["#f5a700","#dc6007","#b00539", "#6b009c", "#006d5b", "#0073e6", "#e6007a", "#00C800", "#FFD500", "#0033A0" ]

# --------------------------------------------------------------------------------------------------------
def plot(df):
    print(df)
    region_sizes = df['region_size_GiB'].astype(int).unique()
    region_sizes.sort()

    systems = df['system'].unique()
    num_systems = len(systems)

    fig, axes = plt.subplots(
        nrows=1, ncols=num_systems,
        figsize=(2 * num_systems, 2),  # Scale width per subplot
        sharey=True
    )

    if num_systems == 1:
        axes = [axes]  # Make iterable if only one subplot

    ymax = df['latency_s'].max()
    for i, system in enumerate(systems):
        ax = axes[i]
        sub_df = df[df['system'] == system]

        sns.lineplot(
            data=sub_df, y='latency_s', x='region_size_GiB',
            hue='placement-op', style='placement-op', markersize=3,
            palette=hpi_palette, markers=True, dashes=False, ax=ax
        )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("Latency [s]" if i == 0 else "")
        ax.set_xlabel("Region size [GiB]")
        ax.set_xticks(region_sizes)
        ax.set_xticklabels(region_sizes)
        ax.set_title(system)
        ax.grid(axis='both', which='major', alpha=0.7, zorder=1)
        ax.grid(axis='both', which='minor', alpha=0.2, zorder=1)

        ax.set_xlim(left=12)
        ax.set_ylim(bottom=-2)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

        ax.get_legend().remove()

        for line in ax.lines:
            line.set_markeredgewidth(0.5)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles=handles,
        labels=labels,
        title="",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.27),
        columnspacing=0.5,
        handlelength=0.8,
        handletextpad=0.4,
        labelspacing=0.5,
        borderpad=0.2
    )


    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/memory-ops-latency-lines.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
# --------------------------------------------------------------------------------------------------------

plot(df)
# ops = df['op'].unique()

# for op in ops:
#   df_plot = df[df['op'] == op]
#   plot(df_plot)
