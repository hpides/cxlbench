#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import util

l1_metrics = ["Retiring", "Front-End Bound", "Bad Speculation", "Back-End Bound"] # pipeline slots
backend_metrics = ["Memory Bound", "Core Bound"] # pipeline slots
memory_metrics = ["L1 Bound", "L2 Bound", "L3 Bound", "DRAM Bound", "Store Bound"] # Clockticks
dram_metrics = ["Memory Bandwidth", "Memory Latency"]
legend_ncol = [2,1,3,1]
legend_titles = ["","","",""]

names = ["l1", "backend", "memory", "dram"]
metric_sets = [l1_metrics, backend_metrics, memory_metrics, dram_metrics]
metric_selection = [True, False, False, True]
y_labels = ["Share of $\mu$Ops [\%]","Share of $\mu$Ops [\%]","Normalized\nshare of $\mu$Ops [\%]", "Normalized\nshare of $\mu$Ops [\%]"]

minimum_level = 2 
max_date = "3000"
max_date += "z"

hpi_palette = ["#f5a700","#dc6007","#b00539"]
palette = hpi_palette + ["#6b009c", "#006d5b"]

def parse_id(df):
    df['Placement'] = df['id'].str.extract(r'_(All\w+)T')
    df['Threads'] = df['id'].str.extract(r'T(\d+)S')
    df['Selectivity'] = (df['id'].str.extract(r'S(\d+)$')).astype(int)
    df["Selectivity"] = df["Selectivity"].apply(
            lambda x: "{:.1f} \%".format(float(x) / 10) if x % 10 != 0 else "{} \%".format(int(float(x) / 10))
        )
    return df

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

scalefactor = 3
fontsize = plt.rcParams['font.size'] * scalefactor
plt.rcParams.update({"font.size": fontsize})
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["legend.title_fontsize"] = fontsize
plt.rcParams["hatch.linewidth"] = plt.rcParams["hatch.linewidth"] * (scalefactor-1)

df_all = util.load_all_csv(results_path)
df_all = parse_id(df_all)
selectivities = df_all['Selectivity'].unique()

subplot_count = sum(metric_selection)
fig, axes = plt.subplots(1, subplot_count, figsize=(scalefactor * len(selectivities) * 1.2 * subplot_count, 14 * scalefactor))
plt.subplots_adjust(wspace=0.5)

ax_idx = 0
for metric_set_index, metrics in enumerate(metric_sets):
    if not metric_selection[metric_set_index]:
        continue
    ax = axes[ax_idx]
    name = names[metric_set_index]
    if name == "memory":
        filter_metrics = metrics.copy() + ["Memory Bound"]
        df = df_all[(df_all["Metric Name"].isin(filter_metrics)) & (df_all["Hierarchy Level"] >= minimum_level) & (df_all["id"] <= max_date)]
        df["Metric Value"] = df["Metric Value"].astype(float)
        # print(df)
        results = []
        for id in df["id"].unique():
            subset = df[df["id"] == id]
            memory_bound = subset[subset["Metric Name"] == "Memory Bound"]["Metric Value"].values[0]
            metrics_total = subset[subset["Metric Name"].isin(metrics)]["Metric Value"].sum()
            
            for metric in metrics:
                metric_value = subset[subset["Metric Name"] == metric]["Metric Value"].values[0]
                normalized_value = (metric_value * memory_bound) / metrics_total if metrics_total != 0 else 0
                results.append({
                    "id": id,
                    "Metric Name": metric,
                    "Metric Value": normalized_value
                })

        df = pd.DataFrame(results)
        df = parse_id(df)
    elif name == "dram":
        filter_metrics = metrics.copy() + memory_metrics + ["Memory Bound"]
        df = df_all[(df_all["Metric Name"].isin(filter_metrics)) & (df_all["Hierarchy Level"] >= minimum_level) & (df_all["id"] <= max_date)]
        df["Metric Value"] = df["Metric Value"].astype(float)
        results = []
        for id in df["id"].unique():
            subset = df[df["id"] == id]
            # Normalize DRAM Bound
            memory_bound = subset[subset["Metric Name"] == "Memory Bound"]["Metric Value"].values[0]
            memory_metrics_total = subset[subset["Metric Name"].isin(memory_metrics)]["Metric Value"].sum()
            dram_bound = subset[subset["Metric Name"] == "DRAM Bound"]["Metric Value"].values[0]
            normalized_dram_bound = (dram_bound * memory_bound) / memory_metrics_total if memory_metrics_total != 0 else 0
            # Normalize Mem Bandwidth and Mem Latency
            dram_metrics_total = subset[subset["Metric Name"].isin(dram_metrics)]["Metric Value"].sum()
            for metric in dram_metrics:
                metric_value = subset[subset["Metric Name"] == metric]["Metric Value"].values[0]
                normalized_value = (metric_value * normalized_dram_bound) / dram_metrics_total if dram_metrics_total != 0 else 0
                results.append({
                    "id": id,
                    "Metric Name": metric,
                    "Metric Value": normalized_value
                })

        df = pd.DataFrame(results)
        df = parse_id(df)
    else:
        df = df_all[(df_all["Metric Name"].isin(metrics)) & (df_all["Hierarchy Level"] >= minimum_level) & (df_all["id"] <= max_date)]
        df["Metric Value"] = df["Metric Value"].astype(float)

    print(df)
    df.reset_index(drop=True, inplace=True)
    df["Placement"] = df["Placement"].str.replace("AllCXL1Blade", "CXL-1", regex=True)
    df["Placement"] = df["Placement"].str.replace("AllCXL4Blades", "CXL-4", regex=True)
    df["Placement"] = df["Placement"].str.replace("AllLocal", "CPU", regex=True)
    # Create and melt the DataFrame
    dfp = df.pivot(index=['Selectivity', 'Placement'], columns=['Metric Name'], values='Metric Value')
    # Fix column order
    if name == "backend":
        dfp = dfp[["Memory Bound", "Core Bound"]]
    print(dfp.to_string())

    # Add solid grey boxes behind every second group of bars
    start = 0
    for i, cat in enumerate(selectivities):
        num_bars = len(dfp.loc[cat])
        if i % 2 == 1:  # Apply shading to every second selectivity group (1-based index)
            ax.axvspan(start - 0.5, start + num_bars - 0.5, color='lightgrey', alpha=0.5)
        start += num_bars

    dfp.plot(kind='bar', stacked=True, ax=ax, color=palette)
    ax.set_xlabel("")
    ax.set_ylabel(y_labels[metric_set_index])
    ax.set_xticklabels([imp for cat, imp in dfp.index], rotation=90)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(axis='y', which='minor', linestyle='-', alpha=0.2)
    ax.grid(axis='y', which='major', linestyle='-', alpha=0.6)
    ax.set_ylim(bottom=0,top=100)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=legend_ncol[metric_set_index], title=legend_titles[metric_set_index], borderaxespad=-5.2, frameon=True,
        handlelength=.9,
        columnspacing=0.5,
        handletextpad=0.3,)

    selectivity_indices = [dfp.index.get_loc((cat, imp)) for cat in selectivities for imp in dfp.loc[cat].index if imp == 'CPU']

    for index, cat in zip(selectivity_indices, selectivities):
        print(index, cat)
        if index == 0:
            ax.text(index - 0.9, 107, "Selectivity", ha='center', fontsize=fontsize, color='black')
        ax.text(index + 1, 107, cat, ha='center', fontsize=fontsize, color='black')
    ax_idx += 1

# plt.tight_layout()
plt.savefig(os.path.join(output_dir,f'breakdown-detailed.pdf'), format='pdf', bbox_inches="tight", pad_inches=0)
