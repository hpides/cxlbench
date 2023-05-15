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
KEY_BM_TYPE = "bm_type"
KEY_CUSTOM_OPS = "custom_operations"
KEY_LAT_AVG = "latency.avg"
KEY_MATRIX_ARGS = "matrix_args"
KEY_NUMA_MEMORY_NODES = "numa_memory_nodes"
KEY_EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"
KEY_PARTITION_COUNT = "number_partitions"
KEY_THREAD_COUNT = "number_threads"
KEY_THREADS = "threads"
KEY_THREADS_LEVELED = "benchmarks.results.threads"

# For plots showing the access size in Bytes on the X axis and the bandwidth on the y axis, this constat specifies the number of threads used.
PLOT_BW_PER_ACCESS_SIZE_THREAD_COUNT = 1
PLOT_FILE_PREFIX = "plot_"


class PlotGenerator:
    """
    This class calls the methods of the plotter classes, according go the given JSON.
    """

    def __init__(self, results, output_dir, no_plots):
        self.results = results
        self.output_dir = output_dir
        self.no_plots = no_plots

    # mainly used for legacy versions of json files. With newer versions, we want to be able to differentiate between
    # different setups, e.g., even if multiple json fils only contain DRAM measurements, the DRAM memory regions might
    # be located on different machines, devices, and NUMA nodes.
    def process_matrix_jsons(self):
        # collect jsons containing matrix arguments
        matrix_jsons = None
        if os.path.isfile(self.results):
            if not self.results.endswith(".json"):
                sys.exit("Result path is a single file but is not a .json file.")
            matrix_jsons = [self.results]
        else:
            matrix_jsons = [path for path in glob.glob(self.results + "/*.json")]

        # create json file list

        dfs = []
        for path in matrix_jsons:
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
        df = ju.flatten_nested_json_df(df, [KEY_MATRIX_ARGS, KEY_THREADS_LEVELED, KEY_EXPLODED_NUMA_MEMORY_NODES])
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].fillna(-1)
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].astype(int)
        df.to_csv("{}/flattened_df.csv".format(self.output_dir))

        drop_columns = [
            "index",
            "bm_type",
            "matrix_args",
            "exec_mode",
            "memory_type",
            "operation",
            "threads",
            "prefault_file",
        ]

        if "persist_instruction" in df.columns:
            drop_columns.append("persist_instruction")

        if "random_distribution" in df.columns:
            drop_columns.append("random_distribution")

        for column in df.columns:
            if column in ["index", KEY_MATRIX_ARGS, KEY_THREADS]:
                continue
            print("{}: {}".format(column, df[column].explode().unique()))

        print("columns to be dropped: {}".format(drop_columns))

        # For now, we assume that memory was allocated on a single numa node.
        assert (df[KEY_NUMA_MEMORY_NODES].str.len() == 1).all()
        df[KEY_NUMA_MEMORY_NODES] = df[KEY_NUMA_MEMORY_NODES].transform(lambda x: x[0])
        df = df.drop(columns=drop_columns)
        df.to_csv("{}/flattened_reduced_df.csv".format(self.output_dir))
        if self.no_plots:
            sys.exit("Exiting without generating plots. CSV were stored.")

        bm_groups = df[KEY_BM_GROUP].unique()
        partition_counts = df[KEY_PARTITION_COUNT].unique()

        for partition_count in partition_counts:
            for bm_group in bm_groups:
                df_sub = df[(df[KEY_BM_GROUP] == bm_group) & (df[KEY_PARTITION_COUNT] == partition_count)]
                self.create_plot(df_sub)

        sys.exit("Exit")

    def create_plot(self, df):
        assert KEY_BM_GROUP in df.columns
        assert len(df[KEY_BM_GROUP].unique()) == 1
        bm_group = df[KEY_BM_GROUP].unique()[0]
        assert KEY_PARTITION_COUNT in df.columns
        assert len(df[KEY_PARTITION_COUNT].unique()) == 1
        partition_count = df[KEY_PARTITION_COUNT].unique()[0]
        bandwidth_plot_group = ["sequential_reads", "random_reads", "sequential_writes", "random_writes"]
        latency_plot_group = ["operation_latency"]
        plot_title = bm_group.replace("_", " ").title()
        legend_title = "Memory Node"
        if bm_group in bandwidth_plot_group:
            # Plot 1 (x: thread count, y: throughput)
            print("Creating barplot (# threads) for BM group {}".format(bm_group))
            filename = "{}{}_threads_{}_partitions.pdf".format(PLOT_FILE_PREFIX, bm_group, partition_count)
            self.create_barplot(
                df,
                KEY_THREAD_COUNT,
                KEY_BANDWIDTH,
                "Number of Threads",
                "Throughput in GB/s",
                KEY_NUMA_MEMORY_NODES,
                plot_title,
                legend_title,
                filename,
            )
            # Plot 2 (x: access size, y: throughput)
            df = df[df[KEY_THREAD_COUNT] == PLOT_BW_PER_ACCESS_SIZE_THREAD_COUNT]
            print("Creating barplot (access sizes) for BM group {}".format(bm_group))
            filename = "{}{}_access_sizes_{}_partitions.pdf".format(PLOT_FILE_PREFIX, bm_group, partition_count)
            self.create_barplot(
                df,
                KEY_ACCESS_SIZE,
                KEY_BANDWIDTH,
                "Access Size in Byte",
                "Throughput in GB/s",
                KEY_NUMA_MEMORY_NODES,
                plot_title,
                legend_title,
                filename,
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
                filename = "{}{}_latency_custom_ops_{}_threads_{}_partitions.pdf".format(
                    PLOT_FILE_PREFIX, bm_group, thread_count, partition_count
                )
                self.create_barplot(
                    df_thread,
                    KEY_CUSTOM_OPS,
                    KEY_LAT_AVG,
                    "Operations",
                    "Latency in ns",
                    KEY_NUMA_MEMORY_NODES,
                    plot_title,
                    legend_title,
                    filename,
                    True,
                )
            print("Generating ploits for latency plot group needs to be implemented.")
        else:
            sys.exit("Benchmark group '{}' is not known.".format(bm_group))

    def create_barplot(self, data, x, y, x_label, y_label, hue, title, legend_title, filename, rotation_x_labels=False):
        hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]
        palette = [hpi_palette[0], hpi_palette[2]]

        barplot = sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None, palette=palette, linewidth=2, edgecolor="k")
        barplot.set_xlabel(x_label)
        barplot.set_ylabel(y_label)
        barplot.set_title(title)

        # Set hatches
        x_distinct_val_count = len(data[x].unique())
        hatches = ["//", "\\\\", "-", "+", "x", "\\", "*", "o", "//", "/", "\\"]
        for patch_idx, bar in enumerate(barplot.patches):
            # Set a different hatch for each bar
            hatch_idx = int(patch_idx / x_distinct_val_count)
            bar.set_hatch(hatches[hatch_idx])
        # Update legend so that hatches are also visible
        barplot.legend(title=legend_title)

        if rotation_x_labels:
            plt.xticks(rotation=90)

        plt.tight_layout()
        fig = barplot.get_figure()
        fig.savefig("{}/{}".format(self.output_dir, filename))
        plt.close(fig)
