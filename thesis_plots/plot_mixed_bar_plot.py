import json
import pandas as pd
import seaborn as sns
from builtins import open, list, enumerate, round, int
from math import floor
from matplotlib import pyplot as plt

data = {
    "gh200": {
        "nvlink": {
            "sequential": {
                "sup_title": "Mixed Sequential Access Throughput",
                "output_path": "./gh200_mixed_seq_bw.pdf",
                "results": [
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_gh200_50_50/build/2024-07-20T11-28-28-247850482/workloads-2024-07-20-11-28-50.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_gh200_75_25/build/2024-07-20T11-26-15-170978470/workloads-2024-07-20-11-26-37.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_gh200_25_75/build/2024-07-20T11-24-01-693914032/workloads-2024-07-20-11-24-24.json",
                ],
            },
            "random": {
                "sup_title": "Mixed Random Access Throughput",
                "output_path": "./gh200_mixed_ran_bw.pdf",
                "results": [
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_r_gh200_50_50/build/2024-07-20T12-37-17-023228268/workloads-2024-07-20-12-37-42.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_r_gh200_75_25/build/2024-07-20T12-42-00-503225722/workloads-2024-07-20-12-42-22.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_r_gh200_25_75/build/2024-07-20T12-39-46-640221177/workloads-2024-07-20-12-40-09.json",
                ],
            },
        }
    },
    "power9": {
        "nvlink": {
            "sequential": {
                "sup_title": "Mixed Sequential Access Throughput",
                "output_path": "./power9_mixed_seq_bw.pdf",
                "results": [
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_50_50/build/2024-07-21T16-18-43-583300551/workloads-2024-07-21-16-25-07.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_75_25/build/2024-07-21T16-45-44-610668320/workloads-2024-07-21-16-50-06.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_25_75/build/2024-07-21T16-40-06-712593663/workloads-2024-07-21-16-44-29.json",
                ],
            },
            "random": {
                "sup_title": "Mixed Random Access Throughput",
                "output_path": "./power9_mixed_r_bw.pdf",
                "results": [
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_r_50_50/build/2024-07-21T17-01-25-519279956/workloads-2024-07-21-17-05-47.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_r_75_25/build/2024-07-22T11-02-49-725005968/workloads-2024-07-22-11-07-11.json",
                    "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_power9_r_25_75/build/2024-07-21T16-55-45-207478548/workloads-2024-07-21-17-00-08.json",
                ],
            },
        }
    },
}


def parse_data(result_path):
    with open(result_path) as f:
        data: list = json.load(f)

    gb_conversion_factor = 2**30 / 10**9
    read_bandwidths = [
        e["benchmarks"][0]["results"]["reads"]["results"]["bandwidth"] * gb_conversion_factor for e in data
    ]
    read_thread_numbers = [e["benchmarks"][0]["config"]["reads"]["number_threads"] for e in data]
    write_bandwidths = [
        e["benchmarks"][0]["results"]["writes"]["results"]["bandwidth"] * gb_conversion_factor for e in data
    ]
    write_thread_numbers = [e["benchmarks"][0]["config"]["writes"]["number_threads"] for e in data]

    if result_path in [
        "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_gh200_50_50/build/2024-07-20T11-28-28-247850482/workloads-2024-07-20-11-28-50.json",
        "/Users/felixwerner/Desktop/mema-bench/results/mixed_bw_r_gh200_50_50/build/2024-07-20T12-37-17-023228268/workloads-2024-07-20-12-37-42.json",
    ]:
        read_bandwidths.pop(0)
        write_bandwidths.pop(0)
        write_thread_numbers.pop(0)
        read_thread_numbers.pop(0)

    total_bandwidths = [read_bandwidths[i] + write_bandwidths[i] for i, e in enumerate(read_bandwidths)]
    threads = [f"{read_thread_numbers[i]}:{write_thread_numbers[i]}" for i, e in enumerate(read_thread_numbers)]

    df = pd.DataFrame(
        data={
            "Number of Threads (read:write)": threads,
            "read": [round(e, 1) for e in read_bandwidths],
            "total": [round(e, 1) for e in total_bandwidths],
            "write": [round(e, 1) for e in write_bandwidths],
        }
    )

    return df


if __name__ == "__main__":
    data = data["gh200"]["nvlink"]["sequential"]
    result_paths = data["results"]
    titles = ["50% Reads, 50% Writes", "75% Reads, 25% Writes", "25% Reads, 75% Writes"]
    output_path = data["output_path"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 5), sharey=True)
    # fig.suptitle(data["sup_title"])
    hatches = ["///", "xx", "\\\\\\"]
    for i, ax in enumerate(axes.flatten()):
        df = parse_data(result_paths[i])
        dfm = pd.melt(
            df, id_vars="Number of Threads (read:write)", var_name="Operation", value_name="Throughput in GB/s"
        )
        ax.set_title(titles[i], fontsize=10)
        ax.title.set_position([0.1, -0.2])
        bar = sns.barplot(x="Number of Threads (read:write)", y="Throughput in GB/s", hue="Operation", data=dfm, ax=ax)
        if "power9" in output_path:
            yticks = [0, 5, 10, 15]
            if i == 0:
                patch_iterator = 8
            else:
                patch_iterator = 4
        else:
            yticks = [0, 50, 100, 150]
            patch_iterator = 9
        l = 0
        for j, b in enumerate(bar.patches):
            h = floor(j / patch_iterator)
            if h > 2:
                h = l
                l += 1
            b.set_hatch(hatches[h])
        ax.legend()
        ax.set_yticks(yticks)
        ax.spines[["top", "right"]].set_visible(False)
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )
        if i != 0:
            ax.get_legend().set_visible(False)
        if i != 1:
            ax.set(ylabel=None)
        if i != 2:
            ax.set(xlabel=None)
        for c in ax.containers:
            ax.bar_label(c, fontsize=6)

    fig.tight_layout()

    fig.savefig(output_path)
