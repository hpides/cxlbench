import pandas as pd

from enums.benchmark_keys import BMKeys
from heatmaps.heatmap import Heatmap


class BandwidthHeatmap(Heatmap):
    def __init__(self, df: pd.DataFrame, title: str, output_dir: str, filename):
        filename = f"bandwidth_{filename}"
        super().__init__(
            df,
            title,
            output_dir,
            filename,
            "Throughput [GB/s]",
            BMKeys.BANDWIDTH_GB,
            compact=True,
            access_size_limit=65536,
            thread_limit=73,
        )
