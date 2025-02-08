import pandas as pd

from enums.benchmark_keys import BMKeys
from heatmaps.heatmap import Heatmap


class BandwidthHeatmap(Heatmap):
    def __init__(self, df: pd.DataFrame, title: str, output_dir: str, filename, yaxis_key, yvalue_limit):
        filename = f"bandwidth_{filename}"
        super().__init__(
            df,
            title,
            output_dir,
            filename,
            "Throughput [GB/s]",
            BMKeys.BANDWIDTH_GB,
            yaxis_key,
            compact=True,
            yvalue_limit=yvalue_limit,
            thread_limit=73,
        )
