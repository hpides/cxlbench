import glob
import json_util as ju
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

KEY_ACCESS_SIZE = "access_size"
KEY_BANDWIDTH = "bandwidth"
KEY_BM_GROUP = "bm_name"
KEY_BM_SUB_NAMES = "sub_bm_names"
KEY_BM_TYPE = "bm_type"
KEY_CUSTOM_OPS = "custom_operations"
KEY_EXEC_TIME = "execution_time"
KEY_LABEL = "label"
KEY_LAT_AVG = "latency.avg"
KEY_MATRIX_ARGS = "matrix_args"
KEY_OP_COUNT = "number_operations"
KEY_THREAD_COUNT = "number_threads"
KEY_THREADS = "threads"
KEY_THREADS_LEVELED = "benchmarks.results.threads"

PLOT_FILE_PREFIX = "plot_"


class ComparisonPlotter:
    """
    This class calls the methods of the plotter classes, according go the given JSON.
    """

    def __init__(self, results_first, results_second, label_first, label_second, output_dir):
        self.results_first = results_first
        self.results_second = results_second
        self.label_first = label_first
        self.label_second = label_second
        self.output_dir = output_dir

    # mainly used for legacy versions of json files. With newer versions, we want to be able to differentiate between
    # different setups, e.g., even if multiple json fils only contain DRAM measurements, the DRAM memory regions might
    # be located on different machines, devices, and NUMA nodes.
    def process_matrix_jsons_comparison(self):
        # collect jsons containing matrix arguments
        if os.path.isfile(self.results_first) or os.path.isfile(self.results_second):
            sys.exit("Result path has to be a directory.")

        # create json file lists for both runs
        matrix_jsons_first = [path for path in glob.glob(self.results_first + "/*.json")]
        matrix_jsons_second = [path for path in glob.glob(self.results_second + "/*.json")]

        # write label into config files
        for path in matrix_jsons_first:
            if not ju.has_label(path, self.label_first):
                print("adding label")
                ju.add_label(path, self.label_first)
            else:
                print("labels are already present")
            # ju.pretty_print(path)
        for path in matrix_jsons_second:
            if not ju.has_label(path, self.label_second):
                print("adding label")
                ju.add_label(path, self.label_second)
            else:
                print("labels are already present")
            # ju.pretty_print(path)

        # compare sequential/random reads/writes
        dfs = []
        for path in matrix_jsons_first:
            df = pd.read_json(path)
            dfs.append(df)

        for path in matrix_jsons_second:
            df = pd.read_json(path)
            dfs.append(df)

        df = pd.concat(dfs)
        bm_names = df[KEY_BM_GROUP].unique()
        print("BM groups: {}".format(bm_names))
        selected_bm_groups = [
            "random_writes",
            "random_reads",
            "sequential_writes",
            "sequential_reads",
            "operation_latency",
        ]
        print("Selected BM groups: {}".format(selected_bm_groups))
        df = df[(df[KEY_BM_GROUP].isin(selected_bm_groups)) & (df[KEY_BM_TYPE] == "single")]
        df = df.drop(columns=[KEY_BM_TYPE, KEY_BM_SUB_NAMES])
        df = ju.flatten_nested_json_df(df, [KEY_MATRIX_ARGS, KEY_THREADS_LEVELED])
        df = df.replace("cxl", "CXL-attached")
        df = df.replace("dram", "CPU-attached")
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].fillna(-1)
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].astype(int)
        df.to_csv("{}/flattened_df.csv".format(self.output_dir))

        bm_groups = df["bm_name"].unique()
        drop_columns = [
            "index",
            "matrix_args",
            "exec_mode",
            "memory_type",
            "numa_pattern",
            "number_partitions",
            "operation",
            "threads",
            "persist_instruction",
            "prefault_file",
            "random_distribution",
        ]
        for column in df.columns:
            if column in ["index", KEY_MATRIX_ARGS, KEY_THREADS]:
                continue
            print("{}: {}".format(column, df[column].unique()))

        print("folumns to be dropped: {}".format(drop_columns))

        df = df.drop(columns=drop_columns)
        df.to_csv("{}/flattened_reduced_df.csv".format(self.output_dir))

        print(bm_groups)

        for bm_group in bm_groups:
            df_sub = df[df[KEY_BM_GROUP] == bm_group]
            self.create_plot(df_sub)

        sys.exit("Exit")

    def create_plot(self, df):
        assert KEY_BM_GROUP in df.columns
        assert len(df[KEY_BM_GROUP].unique()) == 1
        bm_group = df[KEY_BM_GROUP].unique()[0]
        bandwidth_plot_group = ["sequential_reads", "random_reads", "sequential_writes", "random_writes"]
        latency_plot_group = ["operation_latency"]
        if bm_group in bandwidth_plot_group:
            # Plot 1 (x: thread count, y: throughput)
            print("Creating barplot (# threads) for BM group {}".format(bm_group))
            self.create_barplot(
                df,
                KEY_THREAD_COUNT,
                KEY_BANDWIDTH,
                "Number of Threads",
                "Throughput in GB/s",
                KEY_LABEL,
                "Memory Type",
                "{}{}_threads.pdf".format(PLOT_FILE_PREFIX, bm_group),
            )
            # Plot 2 (x: access size, y: throughput)
            df = df[df[KEY_THREAD_COUNT] == df[KEY_THREAD_COUNT].iloc[0]]
            print("Creating barplot (access sizes) for BM group {}".format(bm_group))
            self.create_barplot(
                df,
                KEY_ACCESS_SIZE,
                KEY_BANDWIDTH,
                "Access Size in Byte",
                "Throughput in GB/s",
                KEY_LABEL,
                "Memory Type",
                "{}{}_access_sizes.pdf".format(PLOT_FILE_PREFIX, bm_group),
            )
        elif bm_group in latency_plot_group:
            thread_counts = df[KEY_THREAD_COUNT].unique()
            for thread_count in thread_counts:
                print(
                    "Creating barplot (latency per operations) for BM group {} and thread count {}".format(
                        bm_group, thread_count
                    )
                )
                df_thread = df[df[KEY_THREAD_COUNT] == thread_count]
                self.create_barplot(
                    df_thread,
                    KEY_CUSTOM_OPS,
                    KEY_LAT_AVG,
                    "Operations",
                    "Latency in ns",
                    KEY_LABEL,
                    "Memory Type",
                    "{}{}_latency_custom_ops_{}_threads.pdf".format(PLOT_FILE_PREFIX, bm_group, thread_count),
                    True,
                )
            print("Generating ploits for latency plot group needs to be implemented.")
        else:
            sys.exit("Benchmark group '{}' is not known.".format(bm_group))

    def create_barplot(self, data, x, y, x_label, y_label, hue, legend_title, filename, rotation_x_labels=False):
        hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]
        palette = [hpi_palette[0], hpi_palette[2]]

        barplot = sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None, palette=palette, linewidth=2, edgecolor="k")
        barplot.set_xlabel(x_label)
        barplot.set_ylabel(y_label)
        barplot.legend(title=legend_title)

        # Set hatches
        x_distinct_val_count = len(data[x].unique())
        hatches = ["//", "\\\\", "-", "+", "x", "\\", "*", "o", "//", "/", "\\"]
        for patch_idx, bar in enumerate(barplot.patches):
            # Set a different hatch for each bar
            hatch_idx = int(patch_idx / x_distinct_val_count)
            bar.set_hatch(hatches[hatch_idx])
        # Update legend so that hatches are also visible
        barplot.legend()

        if rotation_x_labels:
            plt.xticks(rotation=15)

        plt.tight_layout()
        fig = barplot.get_figure()
        fig.savefig("{}/{}".format(self.output_dir, filename))
        plt.close(fig)
