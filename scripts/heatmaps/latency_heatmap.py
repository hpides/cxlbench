import pandas as pd
from matplotlib import pyplot as plt

from enums.benchmark_keys import BMKeys
from heatmaps.heatmap import Heatmap


class LatencyHeatmap(Heatmap):
    def __init__(self, df: pd.DataFrame, title: str, output_dir: str, filename):
        filename = f"avg_latency_{filename}"
        super().__init__(
            df,
            title,
            output_dir,
            filename,
            "Thread avg. Latency in ns",
            BMKeys.AVG_ACCESS_LATENCY,
            "magma_r",
            "green",
            "red",
            "d",
        )

    def create(self):
        self.add_heatmap()
        fig = self.heatmap.get_figure()
        fig.savefig(self.output_path)
        plt.close(fig)
