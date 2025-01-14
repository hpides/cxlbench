# The documentation of this file was improved with GitHub copilot.
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.ticker as ticker
import numpy as np
import sys

from enums.benchmark_groups import BMGroups
from enums.benchmark_keys import BMKeys
from enums.file_names import DATA_FILE_PREFIX, PLOT_FILE_PREFIX
from json_util import parse_matrix_jsons
from memaplot import FLUSH_INSTR_NONE
from heatmaps.bandwidth_heatmap import BandwidthHeatmap
from heatmaps.latency_heatmap import LatencyHeatmap


def assert_has_one_value(df, attribute_name):
    assert attribute_name in df.columns, "{} is not in present as a column in the data frame.".format(attribute_name)
    distinct_value_count = len(df[attribute_name].unique())
    #assert distinct_value_count == 1, "{} has {} distinct values but 1 is expected.\n{}".format(
    #    attribute_name, distinct_value_count, df
    #)


def assert_has_multiple_values(df, attribute_name):
    assert attribute_name in df.columns, "{} is not in present as a column in the data frame.".format(attribute_name)
    distinct_value_count = len(df[attribute_name].unique())
    assert distinct_value_count > 1, "{} has {} distinct values but more than 1 are expected.\n{}".format(
        attribute_name, distinct_value_count, df
    )


def get_single_distinct_value(attribute_name, df):
    assert_has_one_value(df, attribute_name)
    return df[attribute_name].unique()[0]


def assert_config_columns_one_value(df, exclude_columns):
    config_columns = [
        BMKeys.BM_GROUP,
        BMKeys.TAG,
        BMKeys.CHUNK_SIZE,
        BMKeys.THREAD_COUNT,
        BMKeys.PARTITION_COUNT,
        BMKeys.ACCESS_SIZE,
        BMKeys.OPERATION,
        BMKeys.OPERATION_COUNT,
        BMKeys.FLUSH_INSTRUCTION,
        BMKeys.MEMORY_REGION_SIZE,
        BMKeys.RUN_TIME,
        BMKeys.RANDOM_DISTRIBUTION,
    ]
    for column in config_columns:
        if column in exclude_columns or column not in df.columns:
            continue
        else:
            assert_has_one_value(df, column)


class PlotGenerator:
    """
    This class calls the methods of the plotter classes, according go the given JSON.
    """

    def __init__(self, results, output_dir, no_plots, latency_heatmap, memory_nodes):
        self.results = results
        self.output_dir = output_dir
        self.no_plots = no_plots
        self.latency_heatmap = latency_heatmap
        self.memory_nodes = memory_nodes

    def add_avg_access_latency(self, df):
        df[BMKeys.TOTAL_ACCESSES] = (df[BMKeys.ACCESSED_BYTES] / df[BMKeys.ACCESS_SIZE]).astype(int)
        total_execution_time = df[BMKeys.THREADS].transform(
            lambda threads_data: sum([x["execution_time"] for x in threads_data])
        )
        total_execution_time = total_execution_time * 1000000000  # seconds to nanoseconds
        df[BMKeys.AVG_ACCESS_LATENCY] = (total_execution_time / df[BMKeys.TOTAL_ACCESSES]).round(0).astype(int)

        return df

    # mainly used for legacy versions of json files. With newer versions, we want to be able to differentiate between
    # different setups, e.g., even if multiple json fils only contain DRAM measurements, the DRAM memory regions might
    # be located on different machines, devices, and NUMA nodes.
    def process_matrix_jsons(self):
        supported_bm_groups = [
            BMGroups.SEQUENTIAL_READS,
            BMGroups.RANDOM_READS,
            BMGroups.SEQUENTIAL_WRITES,
            BMGroups.RANDOM_WRITES,
            BMGroups.OPERATION_LATENCY,
        ]
        df = parse_matrix_jsons(self.results, supported_bm_groups)
        df.to_csv("{}/flattened_df.csv".format(self.output_dir))
        if self.latency_heatmap:
            df = self.add_avg_access_latency(df)

        drop_columns = [
            "index",
            "bm_type",
            "matrix_args",
            "exec_mode",
            "memory_type",
            "threads",
            "prefault_memory",
        ]

        df = df.drop(columns=drop_columns, errors="ignore")
        df.to_csv("{}/flattened_reduced_df.csv".format(self.output_dir))
        if self.no_plots:
            sys.exit("Exiting without generating plots. CSV were stored.")

        bm_groups = df[BMKeys.BM_GROUP].unique()
        partition_counts = df[BMKeys.PARTITION_COUNT].unique()
        flush_types = df[BMKeys.FLUSH_INSTRUCTION].unique()
        tags = df[BMKeys.TAG].unique()
        numa_task_nodes = df[BMKeys.NUMA_TASK_NODES].unique()

        for tag, flush_type, partition_count, bm_group in itertools.product(
            tags, flush_types, partition_counts, bm_groups
        ):
            df_sub = df[
                (df[BMKeys.BM_GROUP] == bm_group)
                & (df[BMKeys.PARTITION_COUNT] == partition_count)
                & (df[BMKeys.FLUSH_INSTRUCTION] == flush_type)
                & (df[BMKeys.TAG] == tag)
                #& (df[BMKeys.NUMA_TASK_NODES] == numa_task_node)
            ]

            # Since we check for certain flush instructions, the data frame is empty for read and
            # operation latency benchmark results if the flush instruction is not `none`.
            if flush_type != FLUSH_INSTR_NONE and ("read" in bm_group or bm_group == "operation_latency"):
                assert df_sub.empty, "Flush instruction must be none for read and latency benchmarks."

            if df_sub.empty:
                continue

            if tag == "B" and flush_type == "nocache" and bm_group == "random_writes":
                # Comment in to filter for a specific thread count.
                # plot_df = df_sub[df_sub[BMKeys.THREAD_COUNT] == 8]
                self.create_paper_plot_throughput_for_threadcount(df_sub, "cache_random_write_8threads")
            self.create_plot(df_sub)

        sys.exit("Exit")

    def create_plot(self, df):
        bm_group = get_single_distinct_value(BMKeys.BM_GROUP, df)
        # Assert that only one partition is used.
        get_single_distinct_value(BMKeys.PARTITION_COUNT, df)
        flush_type = get_single_distinct_value(BMKeys.FLUSH_INSTRUCTION, df)
        tag = get_single_distinct_value(BMKeys.TAG, df)
        numa_task_node = get_single_distinct_value(BMKeys.NUMA_TASK_NODES, df)
        plot_title_template = "{} [Flush: {}] {}\n <custom>".format(tag, flush_type, bm_group.replace("_", " ").title())
        legend_title = "Memory Node"
        filename_tag = ""
        if tag != "":
            filename_tag = "{}_".format(tag)
        filename_template = "{tag}{flush_type}_{bm_group}_task_node_{task_node}_<custom>".format(
            bm_group=bm_group,
            flush_type=flush_type,
            tag=filename_tag,
            task_node=numa_task_node,
        )
        filename = filename_template.replace("_<custom>", "")
        df.to_csv("{}/{}{}.csv".format(self.output_dir, DATA_FILE_PREFIX, filename))
        if BMKeys.BANDWIDTH_GB in df.columns:
            # Plot heatmap (x: thread count, y: access size)
            key_memory_nodes = BMKeys.NUMA_MEMORY_NODES_M0
            if BMKeys.NUMA_MEMORY_NODES in df.columns:
                # legacy measurements
                key_memory_nodes = BMKeys.NUMA_MEMORY_NODES

            numa_memory_nodes = df[key_memory_nodes].unique()
            for memory_node in numa_memory_nodes:
                flush_type = get_single_distinct_value(BMKeys.FLUSH_INSTRUCTION, df)
                print(
                    "Creating heatmap for BM group {}, {}, Mem Node {}, Task Node {}".format(
                        bm_group, flush_type, memory_node, numa_task_node
                    )
                )
                df_sub = df[df[key_memory_nodes] == memory_node]
                plot_title = plot_title_template.replace(
                    "<custom>", "task node: {} mem node: {}".format(numa_task_node, memory_node)
                )

                custom_suffix = f"heatmap_memory_node_{memory_node}"
                filename = filename_template.replace("<custom>", custom_suffix)
                df_sub.to_csv("{}/{}{}.csv".format(self.output_dir, DATA_FILE_PREFIX, filename))

                if self.latency_heatmap:
                    heatmap = LatencyHeatmap(df_sub, plot_title, self.output_dir, filename)
                else:
                    heatmap = BandwidthHeatmap(df_sub, plot_title, self.output_dir, filename)

                heatmap.create()

        elif BMKeys.LAT_AVG in df.columns:
            # Todo: per custom instruction, show threads
            thread_counts = df[BMKeys.THREAD_COUNT].unique()
            thread_counts_count = len(thread_counts)

            for index in range(thread_counts_count):
                fig, axes = plt.subplots(1, 1)
                axes = np.reshape(axes, (1, -1))
                thread_count = thread_counts[index]
                print(
                    "Creating barplot (latency per operations) for BM group {} and thread count {}".format(
                        bm_group, thread_count
                    )
                )
                df_thread = df[df[BMKeys.THREAD_COUNT] == thread_count]
                assert_config_columns_one_value(df_thread, [])
                filename = filename_template.replace("<custom>", "latency_custom_ops_{}_threads".format(thread_count))
                plot_title = plot_title_template.replace(
                    "<custom>", "Latency Custom Ops {} Threads".format(thread_count)
                )

                self.create_barplot(
                    df_thread,
                    BMKeys.CUSTOM_OPS,
                    BMKeys.LAT_AVG,
                    "Operations",
                    "Latency in ns",
                    BMKeys.NUMA_MEMORY_NODES,
                    plot_title,
                    legend_title,
                    self.memory_nodes,
                    axes,
                    0,
                    True,
                )

                fig.set_size_inches(
                    (
                        len(df[df[BMKeys.THREAD_COUNT] == thread_counts[0]][BMKeys.CUSTOM_OPS].unique())
                        + len(df[df[BMKeys.THREAD_COUNT] == thread_counts[0]][BMKeys.NUMA_MEMORY_NODES].unique())
                    )
                    * 0.8,
                    10,
                )
                fig.tight_layout()
                fig.savefig("{}/{}{}.pdf".format(self.output_dir, PLOT_FILE_PREFIX, filename))
                plt.close("all")
            print("Generating ploits for latency plot group needs to be implemented.")

        else:
            sys.exit("Benchmark group '{}' is not known.".format(bm_group))

    def create_paper_plot_throughput_for_threadcount(self, df, filename):
        assert_config_columns_one_value(df, [BMKeys.ACCESS_SIZE])
        df[BMKeys.NUMA_MEMORY_NODES] = df[BMKeys.NUMA_MEMORY_NODES].replace({0: "Local"})
        df[BMKeys.NUMA_MEMORY_NODES] = df[BMKeys.NUMA_MEMORY_NODES].replace({1: "UPI 1-hop remote"})
        df[BMKeys.NUMA_MEMORY_NODES] = df[BMKeys.NUMA_MEMORY_NODES].replace({2: "CXL remote"})

        # colorblind color palette:
        # https://github.com/rasbt/mlxtend/issues/347
        # https://seaborn.pydata.org/tutorial/color_palettes.html#qualitative-color-palettes
        palette = sns.color_palette("colorblind", 3).as_hex()
        x = BMKeys.ACCESS_SIZE
        x_label = "Access size (Byte)"
        y = BMKeys.BANDWIDTH_GB
        y_label = "Throughput (GB/s)"
        hue = BMKeys.NUMA_MEMORY_NODES
        legend_title = None

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params, font_scale=1.3)
        plt.figure(figsize=(7, 2.8))
        barplot = sns.barplot(data=df, x=x, y=y, hue=hue, errorbar=None, palette=palette, linewidth=2, edgecolor="k")
        barplot.set_xlabel(x_label)
        barplot.set_ylabel(y_label)
        barplot.set_title("")

        # Set hatches.
        x_distinct_val_count = len(df[x].unique())
        hatches = ["//", "\\\\", "xx", "++", "--", "\\", "*", "o", "//", "/", "\\"]
        for patch_idx, bar in enumerate(barplot.patches):
            # Set a different hatch for each bar.
            hatch_idx = int(patch_idx / x_distinct_val_count)
            bar.set_hatch(hatches[hatch_idx])
        # Update legend so that hatches are also visible.
        barplot.legend(title=legend_title)
        sns.move_legend(
            barplot,
            "lower center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
            frameon=False,
        )

        barplot.yaxis.set_major_locator(ticker.MultipleLocator(2))
        barplot.yaxis.set_major_formatter(ticker.ScalarFormatter())

        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.grid(axis="y", color="k", linestyle=":")
        fig = barplot.get_figure()
        fig.savefig(
            "{}/{}_paper_{}.pdf".format(self.output_dir, PLOT_FILE_PREFIX, filename),
            bbox_inches="tight",
            pad_inches=0.015,
        )
        plt.close(fig)

    def create_barplot(
        self, data, x, y, x_label, y_label, hue, title, legend_title, memory_nodes, axes, index, rotation_x_labels=False
    ):
        hpi_palette = [(0.9609, 0.6563, 0), (0.8633, 0.3789, 0.0313), (0.6914, 0.0234, 0.2265)]
        palette = [hpi_palette[0], hpi_palette[1], hpi_palette[2]]

        x_index = index % 3
        y_index = math.floor(index / 3)

        x_count = len(data[x].unique())
        hue_count = len(data[hue].unique())
        fig_size_x = (x_count + hue_count) * 1.2
        plt.figure(figsize=(fig_size_x, 7))
        barplot = sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            errorbar=None,
            palette=palette,
            linewidth=2,
            edgecolor="k",
            width=0.4,
            ax=axes[y_index][x_index],
        )
        barplot.margins(y=0.16)
        barplot.set_xlabel(x_label)
        barplot.set_ylabel(y_label)
        barplot.set_title(title, pad=50, fontsize=6)

        barplot.grid(axis="y", color="k", linestyle=":")

        # Set hatches
        x_distinct_val_count = len(data[x].unique())
        hatches = ["//", "\\\\", "xx", "++", "--", "\\", "*", "o", "//", "/", "\\"]
        for patch_idx, bar in enumerate(barplot.patches):
            # Set a different hatch for each bar
            hatch_idx = int(patch_idx / x_distinct_val_count)
            bar.set_hatch(hatches[hatch_idx])

        # Update legend so that hatches are also visible
        barplot.legend(title=legend_title)

        if len(memory_nodes) > 0:
            assert len(memory_nodes) == len(
                data[hue].unique()
            ), "{} memory nodes given but {} memory nodes in the data frame.".format(
                len(memory_nodes), len(data[hue].unique())
            )
            barplot.legend(labels=memory_nodes)

        sns.move_legend(
            barplot,
            "lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=True,
        )

        # Add bar lables.
        for container_id in barplot.containers:
            barplot.bar_label(container_id, rotation=90, padding=4, fmt="%.1f")

        if rotation_x_labels:
            barplot.tick_params(axis="x", labelrotation=90)

        plt.close()
