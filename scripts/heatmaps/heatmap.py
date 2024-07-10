import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, patches

from enums.benchmark_keys import BMKeys
from enums.file_names import PLOT_FILE_PREFIX


class Heatmap:
    def __init__(
        self,
        df: pd.DataFrame,
        title: str,
        output_dir: str,
        filename: str,
        value_label: str,
        value_key: BMKeys,
        color_theme: str = "magma",
        min_color="red",
        max_color="green",
        value_format="d",
        compact=True,
        access_size_limit=8192,
        thread_limit=60,
        mark_linewidth=3,
        # value_format=".2f",
    ):
        self.df = df
        if thread_limit is not None:
            self.df = self.df[self.df[BMKeys.THREAD_COUNT] <= thread_limit]
        if access_size_limit is not None:
            self.df = self.df[self.df[BMKeys.ACCESS_SIZE] <= access_size_limit]
        self.title = title
        self.value_label = value_label
        self.output_path = "{}/{}{}.pdf".format(output_dir, PLOT_FILE_PREFIX, filename)
        self.color_theme = color_theme
        self.min_color = min_color
        self.max_color = max_color
        self.value_format = value_format
        self.compact = compact
        self.mark_linewidth = mark_linewidth
        if self.compact:
            self.mark_linewidth = 1

        self.df_heatmap = pd.pivot_table(
            self.df, index=BMKeys.ACCESS_SIZE, columns=BMKeys.THREAD_COUNT, values=value_key
        )
        self.heatmap = None

    def create(self):
        self.add_heatmap()
        self.mark_max_value_zones()
        if not self.compact:
            self.mark_minimum()
            self.mark_maximum()
        fig = self.heatmap.get_figure()
        fig.savefig(self.output_path)
        plt.close(fig)

    def add_heatmap(self):
        thread_configs_count = len(self.df[BMKeys.THREAD_COUNT].unique())
        access_size_count = len(self.df[BMKeys.ACCESS_SIZE].unique())

        x_padding = 2
        y_padding = 2
        x_scale = 0.25
        y_scale = 0.1
        minimum = 2

        compact_heatmap_groups = ["random_reads", "sequential_writes", "random_writes"]
        assert len(self.df[BMKeys.BM_GROUP].unique()) == 1
        bm_group = self.df[BMKeys.BM_GROUP].unique()[0]

        if self.compact:
            x_scale = 0.16
            y_scale = 0.25
            x_padding = 3 * x_scale
            y_padding = 0
            minimum = 0
            if bm_group in compact_heatmap_groups:
                x_padding = 0

        plt.figure(
            figsize=(
                max(thread_configs_count * x_scale, minimum) + x_padding,
                max(access_size_count * y_scale, minimum / 2) + y_padding,
            )
        )

        rounded_df_heatmap = self.df_heatmap.round().astype(int)
        self.heatmap = sns.heatmap(
            self.df_heatmap,
            annot=rounded_df_heatmap,
            annot_kws={"fontsize": 7, "va": "center_baseline"},
            fmt=self.value_format,
            cmap=self.color_theme,
            linewidths=0.5,  # Add thin grid lines between cells
            linecolor="white",  # Color of the grid lines
            cbar_kws={"label": self.value_label, "pad": 0.02},
            cbar=not self.compact,
        )

        self.heatmap.set_xlabel("Thread Count")
        self.heatmap.set_ylabel("Access size (Byte)")
        self.heatmap.invert_yaxis()
        self.heatmap.set_title(self.title)

        if self.compact:
            # Ensure that get_xticklabels always has labels for all x values
            x_ticks = range(len(self.df_heatmap.columns))
            self.heatmap.set_xticks(x_ticks)
            self.heatmap.set_xticklabels(self.df_heatmap.columns, rotation=90)
            self.heatmap.set_title("")
            if bm_group in compact_heatmap_groups:
                self.heatmap.set_ylabel("")
                self.heatmap.set_yticks([])

        self.heatmap.set_yticklabels(self.heatmap.get_yticklabels(), rotation=0)

        # Add additional padding to the heatmap so that zone marks are not cut off.
        self.heatmap.set_xlim([-0.1, self.df_heatmap.shape[1] + 0.1])
        self.heatmap.set_ylim([-0.1, self.df_heatmap.shape[0] + 0.1])
        plt.tight_layout()

    def mark_maximum(self):
        # Get the row label and column label of the maximum value.
        max_value_row_label, max_value_col_label = self.df_heatmap.stack().idxmax()
        # Get the row index and column index of the maximum value.
        max_value_row_idx = sorted(self.df[BMKeys.ACCESS_SIZE].unique()).index(max_value_row_label)
        max_value_col_idx = sorted(self.df[BMKeys.THREAD_COUNT].unique()).index(max_value_col_label)

        # Add zone around maximum value.
        max_zone = patches.Rectangle(
            (max_value_col_idx, max_value_row_idx),
            1,
            1,
            linewidth=self.mark_linewidth,
            edgecolor=self.max_color,
            facecolor="none",
        )
        self.heatmap.add_patch(max_zone)

    def mark_minimum(self):
        # Get the row label and column label of the minimum value.
        min_value_row, min_value_col = self.df_heatmap.stack().idxmin()
        # Get the row index and column index of the minimum value.
        min_value_row_idx = sorted(self.df[BMKeys.ACCESS_SIZE].unique()).index(min_value_row)
        min_value_col_idx = sorted(self.df[BMKeys.THREAD_COUNT].unique()).index(min_value_col)

        # Add zone around minimum value.
        min_zone = patches.Rectangle(
            (min_value_col_idx, min_value_row_idx),
            1,
            1,
            linewidth=self.mark_linewidth,
            edgecolor=self.min_color,
            facecolor="none",
        )
        self.heatmap.add_patch(min_zone)

    def mark_max_value_zones(self):
        # Get maximum value.
        max_series_over_axis = self.df_heatmap.max()
        max_value = max_series_over_axis.max()

        # --------------------------------------------------------------------------------------------------------------
        # Identify max bandwidth zones.

        threshold_value = max_value * 0.95
        zones = self.get_maximum_value_zones(threshold_value)
        zones = self.get_largest_two_zones(zones)
        # Add the following line to remove the smaller zone when two zones are overlapping.
        if len(zones) > 1:
            zones = self.get_non_overlapping_zones(zones)

        linestyles = ["-", "--", "-.", ":"]
        # For each contiguous region, draw a zone around it.
        for zone_idx, zone in enumerate(zones):
            (x1, y1), (x2, y2) = zone
            self.print_zone_summary(zone)

            linestyle_idx = zone_idx % len(linestyles)
            zone = patches.Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                linestyle=linestyles[0],
                linewidth=self.mark_linewidth,
                edgecolor="grey",
                facecolor="none",
            )
            self.heatmap.add_patch(zone)

    def get_maximum_value_zones(self, threshold_value):
        # Values as array of rows, each row being an array of values.
        value_matrix = self.df_heatmap.values

        zones = []
        # We iterate over all possible zones and check if all values in the zone are True, i.e., above the threshold
        # value. The points (begin_row_idx, begin_col_idx) and (end_row_idx, end_col_idx) span a rectangular zone.
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
                # Since zones with the same begin point are stored adjacent to each other in 'zones', we only need to
                # check the last element of 'grouped_sorted_zones'.
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

    @staticmethod
    def get_largest_two_zones(zones):
        def zone_size(zone):
            return (zone[1][0] - zone[0][0]) * (zone[1][1] - zone[0][1])

        sorted_zones = sorted(zones, key=lambda zone: zone_size(zone), reverse=True)
        # We only consider the largest two zones.
        if len(sorted_zones) > 2:
            sorted_zones = sorted_zones[:2]

        return sorted_zones

    @staticmethod
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
            if zone2_x2 < zone1_x1:
                return False
            # zone2 is on the right of zone1
            if zone2_x1 > zone1_x2:
                return False
            # zone2 is above zone1
            if zone2_y1 > zone1_y2:
                return False
            # zone2 is below zone1
            if zone2_y2 < zone1_y1:
                return False

            return True

        assert len(zones) == 2
        if overlap(zones[0], zones[1]):
            return [zones[0]]
        return zones

    def print_zone_summary(self, zone):
        (x1, y1), (x2, y2) = zone
        # iloc takes [row_idx, column_idx], i.e., [y, x].
        sub_df = self.df_heatmap.iloc[y1 : y2 + 1, x1 : x2 + 1]
        max_series_over_axis = sub_df.max()
        max_value = max_series_over_axis.max()
        min_series_over_axis = sub_df.min()
        min_value = min_series_over_axis.min()
        threads = self.df_heatmap.columns.values
        sizes = self.df_heatmap.index.values
        str_min_val = "Minimum value: {:.1f}".format(min_value)
        str_max_val = "Maximum value: {:.1f}".format(max_value)
        str_threads = "Thread range: {} - {}".format(threads[x1], threads[x2])

        def to_label(size):
            if size > 1024:
                return "{}K".format(size / 1024)
            return "{}B".format(size)

        str_sizes = "Size range: {} - {}".format(to_label(sizes[y1]), to_label(sizes[y2]))
        print(str_min_val, str_max_val, str_threads, str_sizes, sep="\t")
