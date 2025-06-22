#!/usr/bin/env python3

import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
import util

supported_selectivity_per_mill = [1000,600,200,1]
supported_selectivity_per_mill = [1000,1]

hpi_palette = ["#f5a700","#dc6007","#b00539"]
colors = hpi_palette + ["#6b009c", "#006d5b"]

value_size_bytes = 4

parser = argparse.ArgumentParser()

parser.add_argument("results", help="path to the results directory")
args = parser.parse_args()

results_path = args.results
if not results_path.startswith("./") and not results_path.startswith("/"):
    results_path = "./" + results_path

if os.path.isdir(results_path):
    output_dir = results_path
else:
    output_dir = results_path.rsplit("/",1)[0]

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{libertine}'
})

df = util.load_benchmarks_from_json_files(results_path)
df.to_csv("{}/{}.csv".format(output_dir, "scan_results"))

# df = df[df["run_type"] == "aggregate"]
# df = df[df["aggregate_name"].isin(["mean","stddev"])]
df = df[df["aggregate_name"].isin(["mean"])]
df = df.drop(columns=["name"])
df["placement"] = df["run_name"].apply(lambda x: x.split("/")[1])

placement_names = {
    "AllLocal" : "CPU",
    "TableCXL1Blade" : "ColumnsCXL-1",
    "TableCXL4Blades" : "ColumnsCXL-4",
    "AllCXL1Blade" : "CXL-1",
    "AllCXL4Blades" : "CXL-4",
    "E1_AllLocal" : "CPU",
    "E1_ColumnsCXL1Blade" : "ColumnsCXL-1",
    "E1_ColumnsCXL4Blades" : "ColumnsCXL-4",
    "E1_AllCXL1Blade" : "CXL-1",
    "E1_AllCXL4Blades" : "CXL-4",
}
df["placement"] = df["placement"].replace(placement_names)

df["per_mill"] = df["run_name"].apply(lambda x: int(x.split("/")[2]))
df = df[df["per_mill"].isin(supported_selectivity_per_mill)]
df["selectivity_percent"] = df["per_mill"].apply(
    lambda x: "{:.1f} \%".format(float(x) / 10) if x % 10 != 0 else "{} \%".format(int(float(x) / 10))
)
assert (df["time_unit"] == "us").all(), "Not all values in 'time_unit' column are 'us'"
df["real_time_s"] = df["real_time"] / 10e6
df["throughput_ps"] = df["scanned_values"] / df["real_time_s"]
df["throughput_Mps"] = df["throughput_ps"] / 10e6
df["throughput_Gps"] = df["throughput_ps"] / 10e9
df["throughput_GBps"] = df["throughput_Gps"] * value_size_bytes
df.to_csv(f"{output_dir}/scan_agg_results.csv", index=False)

selectivities = df["selectivity_percent"].unique()
placements = df["placement"].unique()
# print(selectivities)

max_columns = 4
num_columns = min(len(selectivities), max_columns)
num_rows = math.ceil(len(selectivities) / max_columns)

fig, axes = plt.subplots(num_rows, num_columns, figsize=(1.6 * num_columns, 19 * num_rows), sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.08)

# Flatten axes if there's more than one row
if num_rows > 1:
    axes = axes.flatten()

# y_key = "throughput_Gps"
# y_label = 'Throughput (G values/s)'

y_key = "throughput_GBps"
y_label = "Throughput [GB/s]"

y_steps = 20
y_min = 0
y_max = df[y_key].max() + y_steps

markers = ['o', 's', 'D', '^', 'X']

for col, selectivity in enumerate(selectivities):
    df_sub = df[df["selectivity_percent"] == selectivities[col]]

    sns.lineplot(
        data=df_sub,
        x="threads", y=y_key, hue="placement", ax=axes[col],
        palette=colors, markers=markers, style='placement', style_order=placements, hue_order=placements,
        markersize=4, dashes='', linewidth=1
    )

    axes[col].set_title("Selectivity: {}%".format(selectivity))
    axes[col].set_xlabel('')
    axes[col].set_ylabel(y_label if col % num_columns == 0 else "")

    xticks = [1] + [x for x in df["threads"].unique() if x % 8 == 0]
    axes[col].set_xticks(xticks)

    axes[col].set_ylim(y_min, y_max)
    axes[col].yaxis.set_major_locator(ticker.MultipleLocator(4.0))
    axes[col].yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    # axes[col].grid(True, axis='y')
    axes[col].grid(axis='y', which='minor', linestyle='-', alpha=0.2)
    axes[col].grid(axis='y', which='major', linestyle='-', alpha=0.6)

    for line in axes[col].lines:
        line.set_alpha(1)
        line.set_markeredgewidth(0.4)
        line.set_markeredgecolor('white')

for col in range(len(selectivities), num_rows * num_columns):
    fig.delaxes(axes[col])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.91, 0.5), borderaxespad=0, frameon=True,
    handlelength=1.2,
    columnspacing=0.8,
    handletextpad=0.5,)

for col in range(len(selectivities)):
    axes[col].get_legend().remove()

fig.supxlabel('Threads', y=-0.1)

fig.savefig("{}/{}-detailed.pdf".format(output_dir, "throughput"), bbox_inches="tight", pad_inches=0)