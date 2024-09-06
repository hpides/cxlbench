#! /usr/bin/env python3

import argparse
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import seaborn as sns
import sys


class LatencyData:
    def __init__(
        self,
        avg,
        lower_quartile,
        upper_quartile,
        max,
        median,
        min,
        p90,
        p95,
        p99,
        p999,
        p9999,
        std_dev,
        mnodes,
        label,
        region_size,
    ):
        self.avg = avg
        self.min = min
        self.max = max
        self.p25 = lower_quartile
        self.p50 = median
        self.p75 = upper_quartile
        self.p90 = p90
        self.p95 = p95
        self.p99 = p99
        self.p999 = p999
        self.p9999 = p9999
        self.std_dev = std_dev
        self.memory_nodes = mnodes
        self.label = label
        self.region_size = region_size

    def percentiles(self):
        return (
            [25, 50, 75, 90, 95, 99, 99.9, 99.99],
            [self.p25, self.p50, self.p75, self.p90, self.p95, self.p99, self.p999, self.p9999],
        )

    def print(self):
        print(
            "avg",
            self.avg,
            "lq",
            self.lower_quartile,
            "uq",
            self.upper_quartile,
            "max",
            self.max,
            "min",
            self.min,
            "p90",
            self.p90,
            "p95",
            self.p95,
            "p99",
            self.p99,
            "p999",
            self.p999,
            "p9999",
            self.p9999,
            "std dev",
            self.std_dev,
        )


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

bm_lat_data = []

# Parse json
for path in file_paths:
    with open(path, "r") as file:
        data = json.load(file)[0]
        assert "benchmarks" in data
        for benchmark in data["benchmarks"]:
            config = benchmark["config"]
            jlat = benchmark["results"]["latency"]
            print(config["m0_numa_nodes"], config["custom_operations"])
            bm_lat_data.append(
                LatencyData(
                    jlat["avg"],
                    jlat["lower_quartile"],
                    jlat["upper_quartile"],
                    jlat["max"],
                    jlat["median"],
                    jlat["min"],
                    jlat["percentile_90"],
                    jlat["percentile_95"],
                    jlat["percentile_99"],
                    jlat["percentile_999"],
                    jlat["percentile_9999"],
                    jlat["std_dev"],
                    config["m0_numa_nodes"],
                    config["custom_operations"],
                    config["m0_region_size"],
                )
            )

# Plot
for idx, lat_data in enumerate(bm_lat_data):
    [percentiles, latencies] = lat_data.percentiles()
    percentiles_str = [str(p) for p in percentiles]

    plt.figure(figsize=(4, 2.5))
    ax = sns.scatterplot(x=percentiles_str, y=latencies, marker="o", linestyle="-", color="b")
    ax.axhline(lat_data.avg, c="blue", label="Mean")
    ax.axhline(lat_data.std_dev, c="red", linestyle="dashed", label="Standard Deviation")
    # ax.axhline(lat_data.p50, c="red", linestyle="dashed", label="Median")
    plt.ylim(0, None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Latency [ns]")
    ax.set_title(f"Tail Latency Percentile Trend {lat_data.label} {lat_data.memory_nodes}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    region_size_gib = lat_data.region_size / 1024**3
    plt.savefig(f"{output_dir_string}{lat_data.label}-{lat_data.memory_nodes}-{region_size_gib}GiB-tail_lat.pdf")
    plt.close("all")
