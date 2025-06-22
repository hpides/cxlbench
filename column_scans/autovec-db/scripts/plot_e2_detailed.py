#!/usr/bin/env python3

import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns
import util

value_size_bytes = 4

hpi_palette = ["#f5a700","#dc6007","#b00539"]
palette = hpi_palette + ["#6b009c", "#006d5b"]

def main():
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
    df["name"] = df["run_name"].apply(lambda x: x.split("/")[1])
    df["placement"] = df["name"].apply(lambda x: x.split("_")[1])
    df["share_cxl"] = df["placement"].apply(lambda x: int(x.split("CXL")[1]))
    df["device_count"] = df["name"].apply(lambda x: x.split("_")[2].replace("Blade",""))
    df["per_mill"] = df["run_name"].apply(lambda x: int(x.split("/")[2]))
    df["selectivity_percent"] = df["per_mill"].apply(
        lambda x: "{:.1f} \%".format(float(x) / 10) if x % 10 != 0 else "{} \%".format(int(float(x) / 10))
    )
    assert (df["time_unit"] == "us").all(), "Not all values in 'time_unit' column are 'us'"
    df["real_time_s"] = df["real_time"] / 10e6
    df["throughput_ps"] = df["scanned_values"] / df["real_time_s"]
    df["throughput_Mps"] = df["throughput_ps"] / 10e6
    df["throughput_Gps"] = df["throughput_ps"] / 10e9
    df["throughput_GBps"] = df["throughput_Gps"] * value_size_bytes
    df["hue"] = df["device_count"] + " Blade(s), " + df["selectivity_percent"] + " Selectivity"
    df.to_csv(f"{output_dir}/scan_agg_results.csv", index=False)

    # === Plotting
    
    unique_device_counts = df["device_count"].unique()
    markers = {device: marker for device, marker in zip(unique_device_counts, ['o', 'x', 'D', 's', '^', 'v', '<', '>', 'P', '*'])}
    colors = {device: marker for device, marker in zip(unique_device_counts, palette[::2])}

    # Create two subplots: one row, two columns
    # fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=False)
    fig, axes = plt.subplots(1, 3, figsize=(5, 16), sharey=False)
    fig.subplots_adjust(wspace=0.0)

    # Get unique selectivity values
    # selectivities = df["selectivity_percent"].unique()
    thread_configs = df["threads"].unique()

    for i, thread_config in enumerate(thread_configs):
        # Filter the dataframe for the current selectivity
        df_selectivity = df[df["threads"] == thread_config]

        # Plot the data on the i-th axis (subplot)
        # for device, marker in markers.items():
        #     df_device = df_selectivity[df_selectivity["device_count"] == device]
        #     sns.lineplot(
        #         data=df_device,
        #         x="share_cxl", y="throughput_Mps", label=f"{device} devices",
        #         color=colors[device], ax=axes[i], marker=marker
        #     )

        sns.lineplot(
            data=df_selectivity,
            x="share_cxl", y="throughput_GBps", hue="device_count",
            palette=colors, ax=axes[i], style="selectivity_percent", markers=True, dashes=False
        )
        
        # Set titles and labels for each subplot
        axes[i].set_title(f"{thread_config} Threads")
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Throughput (GB/s)' if i == 0 else '')

        # Add gridlines for each subplot
        # axes[i].grid(True, axis='y')

        # Set x-axis ticks
        xticks = [x for x in df["share_cxl"].unique()[::2]]
        axes[i].set_xticks(xticks)
        max_y_value = df_selectivity["throughput_GBps"].max()
        axes[i].set_ylim(0, max_y_value * 1.1)
        for line in axes[i].get_lines():
            line.set_markersize(4)

        axes[0].yaxis.set_major_locator(ticker.MultipleLocator(10))
        axes[1].yaxis.set_major_locator(ticker.MultipleLocator(10)) 
        axes[2].yaxis.set_major_locator(ticker.MultipleLocator(10)) 

        # Set minor y-axis ticks at 0.2 intervals without labels
        axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(2))
        axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(2))
        axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(2))

        # Turn off labels for the minor ticks
        axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
        axes[1].yaxis.set_minor_formatter(ticker.NullFormatter())
        axes[2].yaxis.set_minor_formatter(ticker.NullFormatter())

        axes[0].grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)
        axes[1].grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)
        axes[2].grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)

        # Keep major gridlines as well
        axes[0].grid(True, which='major', axis='y')
        axes[1].grid(True, which='major', axis='y')
        axes[2].grid(True, which='major', axis='y')

    # Create a unified legend outside of the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # labels = [label.replace("device_count", "\# Blades").replace("threads", "\# Threads") for label in labels]
    # Remove labels for manual placement.
    labels = [label.replace("device_count", "").replace("selectivity_percent", "") for label in labels]

    # Legend above.
    # fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.1),
    #     columnspacing=0.5,
    #     handlelength=1.2,
    #     handletextpad=0.5,)
    # Legend next to plot.
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.96, 0.6), frameon=False, ncol=1,
        handlelength=1.,
        columnspacing=0.8,
        handletextpad=0.5,)
    
    # Manually adjust the positions of the first and fourth labels
    fig.text(0.98, 0.75, "\# Blades", ha='left', va='center')  # Move first label to the left
    fig.text(0.98, 0.56, "\# Selectivity", ha='left', va='center')  # Move fourth label to the left

    # Remove individual legends from each subplot
    for col in range(len(thread_configs)):
        axes[col].get_legend().remove()

    # Set the overall title for the figure
    # Set shared xlabel
    fig.supxlabel('Share of columns on CXL memory (\%)', y=0.07)

    # Adjust the layout to make space for the overall title
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(f"{output_dir}/throughput_cxl_shares-d.pdf", bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    main()
