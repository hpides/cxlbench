# The documentation of this file was improved with GitHub copilot.

import glob
import json_util as ju
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import math
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import numpy as np
import sys

KEY_ACCESS_SIZE = "access_size"
KEY_BANDWIDTH_GiB = "bandwidth"
KEY_BANDWIDTH_GB = "bandwidth_gb"
KEY_BM_GROUP = "bm_name"
KEY_BM_TYPE = "bm_type"
KEY_CHUNK_SIZE = "min_io_chunk_size"
KEY_CUSTOM_OPS = "custom_operations"
KEY_EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"
KEY_EXPLODED_NUMA_TASK_NODES = "benchmarks.config.numa_task_nodes"
KEY_LAT_AVG = "latency.avg"
KEY_MATRIX_ARGS = "matrix_args"
KEY_MEMORY_REGION_SIZE = "memory_region_size"
KEY_NUMA_TASK_NODES = "numa_task_nodes"
KEY_NUMA_MEMORY_NODES = "numa_memory_nodes"
KEY_OPERATION = "operation"
KEY_OPERATION_COUNT = "number_operations"
KEY_PARTITION_COUNT = "number_partitions"
KEY_RANDOM_DISTRIBUTION = "random_distribution"
KEY_RUN_TIME = "run_time"
KEY_TAG = "tag"
KEY_THREAD_COUNT = "number_threads"
KEY_THREADS = "threads"
KEY_THREADS_LEVELED = "benchmarks.results.threads"
KEY_FLUSH_INSTRUCTION = "flush_instruction"
FLUSH_INSTR_NONE = "none"

DATA_FILE_PREFIX = "data_"
PLOT_FILE_PREFIX = "plot_"
FILE_TAG_SUBSTRING = "TAG_"


def assert_has_one_value(df, attribute_name):
    assert attribute_name in df.columns, "{} is not in present as a column in the data frame.".format(attribute_name)
    distinct_value_count = len(df[attribute_name].unique())
    assert distinct_value_count == 1, "{} has {} distinct values but 1 is expected.\n{}".format(
        attribute_name, distinct_value_count, df
    )


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
        KEY_BM_GROUP,
        KEY_TAG,
        KEY_CHUNK_SIZE,
        KEY_THREAD_COUNT,
        KEY_PARTITION_COUNT,
        KEY_ACCESS_SIZE,
        KEY_OPERATION,
        KEY_OPERATION_COUNT,
        KEY_FLUSH_INSTRUCTION,
        KEY_MEMORY_REGION_SIZE,
        KEY_RUN_TIME,
        KEY_RANDOM_DISTRIBUTION,
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

    def __init__(self, results, output_dir, no_plots, do_barplots, memory_nodes):
        self.results = results
        self.output_dir = output_dir
        self.no_plots = no_plots
        self.do_barplots = do_barplots
        self.memory_nodes = memory_nodes

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
            # Get the tag from the file name.
            tag = ""
            if FILE_TAG_SUBSTRING in path:
                path_parts = path.split(FILE_TAG_SUBSTRING)
                assert (
                    len(path_parts) == 2
                ), "Make sure that the substring {} appears only once in a result file name.".format(FILE_TAG_SUBSTRING)
                tag_part = path_parts[-1]
                assert (
                    "-" not in tag_part
                ), "Make sure that the tag is the last part of the name before the file extension."
                assert (
                    "_" not in tag_part
                ), "Make sure that the tag is the last part of the name before the file extension."
                tag = tag_part.split(".")[0]

            df = pd.read_json(path)
            df[KEY_TAG] = tag
            dfs.append(df)

        df = pd.concat(dfs)
        bm_names = df[KEY_BM_GROUP].unique()
        print("Existing BM groups: {}".format(bm_names))
        selected_bm_groups = [
            "random_writes",
            "random_reads",
            "sequential_writes",
            "sequential_reads",
            "operation_latency",
        ]
        print("Supported BM groups: {}".format(selected_bm_groups))

        df = df[(df[KEY_BM_GROUP].isin(selected_bm_groups)) & (df[KEY_BM_TYPE] == "single")]
        df = ju.flatten_nested_json_df(
            df,
            [
                KEY_MATRIX_ARGS,
                KEY_THREADS_LEVELED,
                KEY_EXPLODED_NUMA_MEMORY_NODES,
                KEY_EXPLODED_NUMA_TASK_NODES,
            ],
        )

        # If only latency benchnarks have been performed, the dataframe does note have a KEY_ACCESS_SIZE column so it
        # must be added.
        if KEY_ACCESS_SIZE not in df.columns:
            df[KEY_ACCESS_SIZE] = -1
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].fillna(-1)
        df[KEY_ACCESS_SIZE] = df[KEY_ACCESS_SIZE].astype(int)

        # For read benchmarks, an additional flush instruction will never be performed. As 'none' is also one of the
        # valid flush instructions, we set the corresponding value to 'none'. If only read benchnarks have been
        # performed, the dataframe does note have a KEY_FLUSH_INSTRUCTION column so it must be added.
        if KEY_FLUSH_INSTRUCTION not in df.columns:
            df[KEY_FLUSH_INSTRUCTION] = FLUSH_INSTR_NONE
        df[KEY_FLUSH_INSTRUCTION] = df[KEY_FLUSH_INSTRUCTION].fillna(FLUSH_INSTR_NONE)
        if KEY_BANDWIDTH_GiB in df.columns:
            df[KEY_BANDWIDTH_GB] = df[KEY_BANDWIDTH_GiB] * (1024**3 / 1e9)

        df.to_csv("{}/flattened_df.csv".format(self.output_dir))

        drop_columns = [
            "index",
            "bm_type",
            "matrix_args",
            "exec_mode",
            "memory_type",
            "threads",
            "prefault_memory",
        ]

        # (comment in for debug purposes)
        # for column in df.columns:
        #     if column in ["index", KEY_MATRIX_ARGS, KEY_THREADS]:
        #         continue
        #     print("{}: {}".format(column, df[column].explode().unique()))
        # print("columns to be dropped: {}".format(drop_columns))

        df[KEY_NUMA_MEMORY_NODES] = df[KEY_NUMA_MEMORY_NODES].transform(lambda x: ",".join(str(i) for i in x))
        df[KEY_NUMA_TASK_NODES] = df[KEY_NUMA_TASK_NODES].transform(lambda x: ",".join(str(i) for i in x))
        df = df.drop(columns=drop_columns, errors="ignore")
        df.to_csv("{}/flattened_reduced_df.csv".format(self.output_dir))
        if self.no_plots:
            sys.exit("Exiting without generating plots. CSV were stored.")

        bm_groups = df[KEY_BM_GROUP].unique()
        partition_counts = df[KEY_PARTITION_COUNT].unique()
        flush_types = df[KEY_FLUSH_INSTRUCTION].unique()
        tags = df[KEY_TAG].unique()
        numa_task_nodes = df[KEY_NUMA_TASK_NODES].unique()

        for tag in tags:
            for flush_type in flush_types:
                for partition_count in partition_counts:
                    for bm_group in bm_groups:
                        for numa_task_node in numa_task_nodes:
                            # (comment in for debug purposes)
                            # print(tag, flush_type, partition_count, bm_group)
                            df_sub = df[
                                (df[KEY_BM_GROUP] == bm_group)
                                & (df[KEY_PARTITION_COUNT] == partition_count)
                                & (df[KEY_FLUSH_INSTRUCTION] == flush_type)
                                & (df[KEY_TAG] == tag)
                                & (df[KEY_NUMA_TASK_NODES] == numa_task_node)
                            ]

                            # (comment in for debug purposes)
                            # print("DF for", tag, bm_group, partition_count, flush_type, tag)
                            # print(df_sub.to_string())

                            # Since we check for certain flush instructions, the data frame is empty for read and
                            # operation latency benchmark results if the flush instruction is not `none`.
                            if flush_type != FLUSH_INSTR_NONE and (
                                "read" in bm_group or bm_group == "operation_latency"
                            ):
                                assert df_sub.empty, "Flush instruction must be none for read and latency benchmarks."

                            if df_sub.empty:
                                continue

                            if tag == "B" and flush_type == "nocache" and bm_group == "random_writes":
                                # Comment in to filter for a specific thread count.
                                # plot_df = df_sub[df_sub[KEY_THREAD_COUNT] == 8]
                                self.create_paper_plot_throughput_for_threadcount(df_sub, "cache_random_write_8threads")
                            self.create_plot(df_sub)

        sys.exit("Exit")

    def create_plot(self, df):
        bm_group = get_single_distinct_value(KEY_BM_GROUP, df)
        # Assert that only one partition is used.
        get_single_distinct_value(KEY_PARTITION_COUNT, df)
        flush_type = get_single_distinct_value(KEY_FLUSH_INSTRUCTION, df)
        tag = get_single_distinct_value(KEY_TAG, df)
        numa_task_node = get_single_distinct_value(KEY_NUMA_TASK_NODES, df)
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
        if KEY_BANDWIDTH_GB in df.columns:
            if self.do_barplots:
                # Plot 1 (x: thread count, y: throughput, for each access size)
                access_sizes = df[KEY_ACCESS_SIZE].unique()
                access_sizes_count = len(access_sizes)

                row_count = math.ceil(access_sizes_count / 3)
                col_count = min(3, access_sizes_count)

                fig, axes = plt.subplots(row_count, col_count)
                if access_sizes_count <= 3:
                    axes = np.reshape(axes, (1, access_sizes_count))
                else:
                    if access_sizes_count % 3 == 1:
                        axes[-1, -1].axis("off")
                        axes[-1, -2].axis("off")
                    if access_sizes_count % 3 == 2:
                        axes[-1, -1].axis("off")

                for index in range(access_sizes_count):
                    access_size = access_sizes[index]
                    plot_df = df[df[KEY_ACCESS_SIZE] == access_size]
                    assert_config_columns_one_value(plot_df, [KEY_THREAD_COUNT])
                    print("Creating barplot (# threads) for BM group {}, {}B".format(bm_group, access_size))
                    plot_title = plot_title_template.replace("<custom>", "{}B".format(access_size))

                    self.create_barplot(
                        plot_df,
                        KEY_THREAD_COUNT,
                        KEY_BANDWIDTH_GB,
                        "Number of Threads",
                        "Throughput in GB/s",
                        KEY_NUMA_MEMORY_NODES,
                        plot_title,
                        legend_title,
                        self.memory_nodes,
                        axes,
                        index,
                    )

                fig.set_size_inches(
                    min(3, len(access_sizes))
                    * (
                        len(df[df[KEY_ACCESS_SIZE] == access_sizes[0]][KEY_THREAD_COUNT].unique())
                        + len(df[df[KEY_ACCESS_SIZE] == access_sizes[0]][KEY_NUMA_MEMORY_NODES].unique())
                    )
                    * 0.8,
                    math.ceil(len(access_sizes) / 3) * 5,
                )
                fig.tight_layout()

                filename = "{}".format(filename_template.replace("<custom>", "access_size"))
                fig.savefig("{}/{}{}.pdf".format(self.output_dir, PLOT_FILE_PREFIX, filename))
                plt.close("all")

                # Plot 2 (x: access size, y: throughput)
                thread_counts = df[KEY_THREAD_COUNT].unique()
                thread_counts_count = len(thread_counts)

                row_count = math.ceil(thread_counts_count / 3)
                col_count = min(3, thread_counts_count)

                fig, axes = plt.subplots(row_count, col_count)
                if thread_counts_count <= 3:
                    axes = np.reshape(axes, (1, thread_counts_count))
                else:
                    if thread_counts_count % 3 == 1:
                        axes[-1, -1].axis("off")
                        axes[-1, -2].axis("off")
                    if thread_counts_count % 3 == 2:
                        axes[-1, -1].axis("off")

                for index in range(thread_counts_count):
                    thread_count = thread_counts[index]
                    plot_df = df[df[KEY_THREAD_COUNT] == thread_count]
                    assert_config_columns_one_value(plot_df, [KEY_ACCESS_SIZE])
                    print("Creating barplot (access sizes) for BM group {}, {} threads".format(bm_group, thread_count))
                    filename = filename_template.replace("<custom>", "{}_threads".format(thread_count))
                    plot_title = plot_title_template.replace("<custom>", "{} Threads".format(thread_count))

                    self.create_barplot(
                        plot_df,
                        KEY_ACCESS_SIZE,
                        KEY_BANDWIDTH_GB,
                        "Access Size in Byte",
                        "Throughput in GB/s",
                        KEY_NUMA_MEMORY_NODES,
                        plot_title,
                        legend_title,
                        self.memory_nodes,
                        axes,
                        index,
                    )

                filename = filename_template.replace("<custom>", "threads")
                fig.set_size_inches(
                    min(3, len(thread_counts))
                    * (
                        len(df[df[KEY_THREAD_COUNT] == thread_counts[0]][KEY_ACCESS_SIZE].unique())
                        + len(df[df[KEY_THREAD_COUNT] == thread_counts[0]][KEY_NUMA_MEMORY_NODES].unique())
                    )
                    * 0.8,
                    math.ceil(len(thread_counts) / 3) * 5,
                )
                fig.tight_layout()
                fig.savefig("{}/{}{}.pdf".format(self.output_dir, PLOT_FILE_PREFIX, filename))
                plt.close("all")

            # Plot 3: heatmap (x: thread count, y: access size)
            numa_memory_nodes = df[KEY_NUMA_MEMORY_NODES].unique()
            for memory_node in numa_memory_nodes:
                flush_type = get_single_distinct_value(KEY_FLUSH_INSTRUCTION, df)
                print(
                    "Creating heatmap for BM group {}, {}, Mem Node {}, Task Node {}".format(
                        bm_group, flush_type, memory_node, numa_task_node
                    )
                )
                df_sub = df[df[KEY_NUMA_MEMORY_NODES] == memory_node]
                plot_title = plot_title_template.replace(
                    "<custom>", "task node: {} mem node: {}".format(numa_task_node, memory_node)
                )

                filename = filename_template.replace("<custom>", "heatmap_memory_node_{}".format(memory_node))
                df_sub.to_csv("{}/{}{}.csv".format(self.output_dir, DATA_FILE_PREFIX, filename))

                self.create_heatmap(df_sub, plot_title, filename)
        elif KEY_LAT_AVG in df.columns:
            # Todo: per custom instruction, show threads
            thread_counts = df[KEY_THREAD_COUNT].unique()
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
                df_thread = df[df[KEY_THREAD_COUNT] == thread_count]
                assert_config_columns_one_value(df_thread, [])
                filename = filename_template.replace("<custom>", "latency_custom_ops_{}_threads".format(thread_count))
                plot_title = plot_title_template.replace(
                    "<custom>", "Latency Custom Ops {} Threads".format(thread_count)
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
                    self.memory_nodes,
                    axes,
                    0,
                    True,
                )

                fig.set_size_inches(
                    (
                        len(df[df[KEY_THREAD_COUNT] == thread_counts[0]][KEY_CUSTOM_OPS].unique())
                        + len(df[df[KEY_THREAD_COUNT] == thread_counts[0]][KEY_NUMA_MEMORY_NODES].unique())
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
        assert_config_columns_one_value(df, [KEY_ACCESS_SIZE])
        df[KEY_NUMA_MEMORY_NODES] = df[KEY_NUMA_MEMORY_NODES].replace({0: "Local"})
        df[KEY_NUMA_MEMORY_NODES] = df[KEY_NUMA_MEMORY_NODES].replace({1: "UPI 1-hop remote"})
        df[KEY_NUMA_MEMORY_NODES] = df[KEY_NUMA_MEMORY_NODES].replace({2: "CXL remote"})

        # colorblind color palette:
        # https://github.com/rasbt/mlxtend/issues/347
        # https://seaborn.pydata.org/tutorial/color_palettes.html#qualitative-color-palettes
        palette = sns.color_palette("colorblind", 3).as_hex()
        x = KEY_ACCESS_SIZE
        x_label = "Access size (Byte)"
        y = KEY_BANDWIDTH_GB
        y_label = "Throughput (GB/s)"
        hue = KEY_NUMA_MEMORY_NODES
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

    def create_heatmap(self, df, title, filename):
        df_heatmap = pd.pivot_table(df, index=KEY_ACCESS_SIZE, columns=KEY_THREAD_COUNT, values=KEY_BANDWIDTH_GB)

        thread_count = len(df[KEY_THREAD_COUNT].unique())
        access_size_count = len(df[KEY_ACCESS_SIZE].unique())
        padding = 2
        x_scale = 0.6
        y_scale = 0.2
        minimum = 4
        plt.figure(
            figsize=(
                max(thread_count * x_scale, minimum) + padding,
                max(access_size_count * y_scale, minimum / 2) + padding
            )
        )
        heatmap = sns.heatmap(
            df_heatmap,
            annot=True,
            annot_kws={"fontsize": 7, "va": "center_baseline"},
            fmt=".2f",
            cmap="magma",
            cbar_kws={"label": "Throughput in GB/s", "pad": 0.02}
        )

        heatmap.set_xlabel("Thread Count")
        heatmap.set_ylabel("Access size (Byte)")
        heatmap.invert_yaxis()
        heatmap.set_title(title)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

        # Add additional padding to the heatmap so that zone marks are not cut off.
        heatmap.set_xlim([-0.1, df_heatmap.shape[1] + 0.1])
        heatmap.set_ylim([-0.1, df_heatmap.shape[0] + 0.1])

        # Get maximum value.
        max_series_over_axis = df_heatmap.max()
        max_value = max_series_over_axis.max()

        # --------------------------------------------------------------------------------------------------------------
        # Identify max bandwidth zones.

        threshold_value = max_value * 0.95
        zones = get_maximum_value_zones(df_heatmap, threshold_value)
        zones = get_largest_two_zones(zones)
        # Add the following line to remove the smaller zone when two zones are overlapping.
        # zones = get_non_overlapping_zones(zones)

        linestyles = ["-", "--", "-.", ":"]
        # For each contiguous region, draw a zone around it.
        for zone_idx, zone in enumerate(zones):
            (x1, y1), (x2, y2) = zone
            print_zone_summary(df_heatmap, zone)

            linestyle_idx = zone_idx % len(linestyles)
            zone = patches.Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                linestyle=linestyles[linestyle_idx],
                linewidth=3,
                edgecolor="grey",
                facecolor="none",
            )
            heatmap.add_patch(zone)

        # --------------------------------------------------------------------------------------------------------------
        # Mark maximum and minimum values.

        # Get the row label and column label of the maximum value.
        max_value_row_label, max_value_col_label = df_heatmap.stack().idxmax()
        # Get the row index and column index of the maximum value.
        max_value_row_idx = sorted(df[KEY_ACCESS_SIZE].unique()).index(max_value_row_label)
        max_value_col_idx = sorted(df[KEY_THREAD_COUNT].unique()).index(max_value_col_label)

        # Add zone around maximum value.
        max_zone = patches.Rectangle(
            (max_value_col_idx, max_value_row_idx), 1, 1, linewidth=5, edgecolor="green", facecolor="none"
        )
        heatmap.add_patch(max_zone)

        # Get the row label and column label of the minimum value.
        min_value_row, min_value_col = df_heatmap.stack().idxmin()
        # Get the row index and column index of the minimum value.
        min_value_row_idx = sorted(df[KEY_ACCESS_SIZE].unique()).index(min_value_row)
        min_value_col_idx = sorted(df[KEY_THREAD_COUNT].unique()).index(min_value_col)

        # Add zone around minimum value.
        min_zone = patches.Rectangle(
            (min_value_col_idx, min_value_row_idx), 1, 1, linewidth=5, edgecolor="red", facecolor="none"
        )
        heatmap.add_patch(min_zone)

        # --------------------------------------------------------------------------------------------------------------
        # Save plot.

        fig = heatmap.get_figure()
        fig.savefig("{}/{}{}.pdf".format(self.output_dir, PLOT_FILE_PREFIX, filename))
        plt.close(fig)

        # End create_heatmap -------------------------------------------------------------------------------------------


def get_maximum_value_zones(df, threshold_value):
    # Values as array of rows, each row being an array of values.
    value_matrix = df.values

    zones = []
    # We iterate over all possible zones and check if all values in the zone are True, i.e., above the
    # threshold value. The points (begin_row_idx, begin_col_idx) and (end_row_idx, end_col_idx) span a rectangular zone.
    for begin_row_idx in range(len(value_matrix)):
        for begin_col_idx in range(len(value_matrix[begin_row_idx])):
            for end_row_idx in range(begin_row_idx, len(value_matrix)):
                for end_col_idx in range(begin_col_idx, len(value_matrix[end_row_idx])):
                    all_cells_above_threshold = True
                    for row_idx in range(begin_row_idx, end_row_idx + 1):
                        if not all_cells_above_threshold:
                            break
                        for col_idx in range(begin_col_idx, end_col_idx + 1):
                            if value_matrix[row_idx][col_idx] < threshold_value:
                                all_cells_above_threshold = False
                                break
                    if all_cells_above_threshold:
                        zones.append(((begin_col_idx, begin_row_idx), (end_col_idx, end_row_idx)))

    # ------------------------------------------------------------------------------------------------------------------

    # We group the zones by their begin point.

    # Store lists of zones with the same begin point in a list.
    grouped_sorted_zones = []

    for zone in zones:
        (begin_point, _) = zone
        if len(grouped_sorted_zones) == 0:
            new_zone_list = [zone]
            grouped_sorted_zones.append(new_zone_list)
        else:
            # Since zones with the same begin point are stored adjacent to each other in 'zones', we only need to check
            # the last element of 'grouped_sorted_zones'.
            last_zone_list = grouped_sorted_zones[-1]
            # get the first item of `last_zone_list` since all begin points of the zones in that list are the same.
            (last_zone_list_begin_point, _) = last_zone_list[0]
            if begin_point == last_zone_list_begin_point:
                last_zone_list.append(zone)
            else:
                new_zone_list = [zone]
                grouped_sorted_zones.append(new_zone_list)

    # ------------------------------------------------------------------------------------------------------------------

    # For the zones of each group, i.e., with the same begin point, we only keep the largest zone.
    filtered_zones = []

    for sorted_zones in grouped_sorted_zones:
        max_x = -1
        max_y = -1
        max_x_zone = None
        max_y_zone = None
        for zone in sorted_zones:
            # Note that a zone is defined by its [0] begin and [1] end point. Since the end point has an x and y
            # index always larger than the begin point, we only need to check the end point (i.e., zone[1]).
            zone_end_point = zone[1]
            if zone_end_point[0] > max_x:
                max_x = zone_end_point[0]
                max_x_zone = zone
            elif zone_end_point[0] == max_x:
                if zone_end_point[1] > max_x_zone[1][1]:
                    max_x_zone = zone
            if zone_end_point[1] > max_y:
                max_y = zone_end_point[1]
                max_y_zone = zone
            elif zone_end_point[1] == max_y:
                if zone_end_point[0] > max_y_zone[1][0]:
                    max_y_zone = zone

        filtered_zones.append(max_x_zone)
        if (max_x_zone[1][0], max_x_zone[1][1]) != (max_y_zone[1][0], max_y_zone[1][1]):
            filtered_zones.append(max_y_zone)

    # ------------------------------------------------------------------------------------------------------------------

    # The previous step might have created zones with the same end point. We group the zones by their end
    # point. The end point is stored as key and the begin points as values in a dictionary.

    begin_points_by_end_point = {}
    for (x1, y1), (x2, y2) in filtered_zones:
        if (x2, y2) not in begin_points_by_end_point:
            new_list = [(x1, y1)]
            begin_points_by_end_point[(x2, y2)] = new_list
        else:
            begin_points_by_end_point[(x2, y2)].append((x1, y1))

    # We filter out zones that are completely contained in another larger zone.
    filtered_zones = []
    for end_point, begin_points in begin_points_by_end_point.items():
        # For a given end point, we only add the zones with the smallest begin point. If the begin point with the
        # smallest x does not equal the begin point with the smalles y, we add both zones.
        min_x = len(value_matrix[0]) + 1
        min_y = len(value_matrix) + 1
        min_x_zone = None
        min_y_zone = None
        for x1, y1 in begin_points:
            if x1 < min_x:
                min_x = x1
                min_x_zone = ((x1, y1), end_point)
            elif x1 == min_x:
                if y1 < min_x_zone[0][1]:
                    min_x_zone = ((x1, y1), end_point)
            if y1 < min_y:
                min_y = y1
                min_y_zone = ((x1, y1), end_point)
            elif y1 == min_y:
                if x1 < min_y_zone[0][0]:
                    min_y_zone = ((x1, y1), end_point)

        filtered_zones.append(min_x_zone)
        if (min_x_zone[0][0], min_x_zone[0][1]) != (min_y_zone[0][0], min_y_zone[0][1]):
            filtered_zones.append(min_y_zone)

    return filtered_zones


def get_largest_two_zones(zones):
    def zone_size(zone):
        return (zone[1][0] - zone[0][0]) * (zone[1][1] - zone[0][1])

    sorted_zones = sorted(zones, key=lambda zone: zone_size(zone), reverse=True)
    # We only consider the largest two zones.
    if len(sorted_zones) > 2:
        sorted_zones = sorted_zones[:2]

    return sorted_zones


def get_non_overlapping_zones(zones):
    def overlap(zone1, zone2):
        zone1_x1 = zone1[0][0]
        zone1_x2 = zone1[1][0]
        zone1_y1 = zone1[0][1]
        zone1_y2 = zone1[1][1]
        zone2_x1 = zone2[0][0]
        zone2_x2 = zone2[1][0]
        zone2_y1 = zone2[0][1]
        zone2_y2 = zone2[1][1]
        # zone2 is on the left of zone1
        if zone2_x2 <= zone1_x1:
            return False
        # zone2 is on the right of zone1
        if zone2_x1 >= zone1_x2:
            return False
        # zone2 is above zone1
        if zone2_y1 >= zone1_y2:
            return False
        # zone2 is below zone1
        if zone2_y2 <= zone1_y1:
            return False

        return True

    assert len(zones) == 2
    if overlap(zones[0], zones[1]):
        return [zones[0]]
    return zones


def print_zone_summary(df, zone):
    (x1, y1), (x2, y2) = zone
    # iloc takes [row_idx, column_idx], i.e., [y, x].
    sub_df = df.iloc[y1 : y2 + 1, x1 : x2 + 1]
    max_series_over_axis = sub_df.max()
    max_value = max_series_over_axis.max()
    min_series_over_axis = sub_df.min()
    min_value = min_series_over_axis.min()
    threads = df.columns.values
    sizes = df.index.values
    str_min_val = "Minimum value: {:.1f}".format(min_value)
    str_max_val = "Maximum value: {:.1f}".format(max_value)
    str_threads = "Thread range: {} - {}".format(threads[x1], threads[x2])

    def to_label(size):
        if size > 1024:
            return "{}K".format(size / 1024)
        return "{}B".format(size)

    str_sizes = "Size range: {} - {}".format(to_label(sizes[y1]), to_label(sizes[y2]))
    print(str_min_val, str_max_val, str_threads, str_sizes, sep="\t")
