#! /usr/bin/env python3

import matplotlib
import numpy as np
import pandas as pd
from builtins import len, enumerate
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns

# GH200
loaded_latency_local = [220, 194, 213, 219, 236, 257, 276, 329, 355, 395, 406, 416, 419, 421, 427, 426, 429, 431, 435]
loaded_latency_local_writes = [
    220,
    211,
    235,
    259,
    265,
    285,
    315,
    330,
    308,
    335,
    309,
    334,
    359,
    338,
    362,
    335,
    336,
    337,
    354,
]
loaded_latency_local_reads_random = [
    220,
    201,
    202,
    194,
    199,
    197,
    217,
    206,
    208,
    211,
    229,
    225,
    232,
    240,
    248,
    257,
    269,
    281,
    294,
]
loaded_latency_local_writes_random = [
    220,
    203,
    232,
    279,
    312,
    317,
    319,
    320,
    324,
    323,
    325,
    328,
    335,
    341,
    342,
    340,
    382,
    349,
    353,
]

loaded_latency_nvlink = [
    807,
    805,
    810,
    816,
    826,
    826,
    842,
    859,
    898,
    935,
    967,
    973,
    990,
    988,
    1000,
    999,
    1003,
    1013,
    1022,
]
loaded_latency_nvlink_reads_random = [
    807,
    806,
    804,
    805,
    803,
    807,
    816,
    812,
    819,
    824,
    828,
    829,
    840,
    853,
    870,
    882,
    888,
    905,
    914,
]
loaded_latency_nvlink_writes = [
    807,
    808,
    824,
    817,
    822,
    843,
    893,
    919,
    936,
    932,
    934,
    938,
    935,
    938,
    940,
    937,
    938,
    942,
    942,
]
loaded_latency_nvlink_writes_random = [
    807,
    802,
    803,
    807,
    813,
    821,
    837,
    868,
    899,
    914,
    931,
    931,
    939,
    932,
    940,
    929,
    929,
    931,
    939,
]
loaded_threads = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 71]

df = pd.DataFrame(
    data={
        "Number of Loaded Threads": loaded_threads,
        "Latency in ns": loaded_latency_nvlink,
        "Loaded Operation": "Sequential Read",
        "Memory Region": "HBM3\n(GPU)",
    }
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_nvlink_reads_random,
                "Loaded Operation": "Random Read",
                "Memory Region": "HBM3\n(GPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_nvlink_writes,
                "Loaded Operation": "Sequential Write",
                "Memory Region": "HBM3\n(GPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_nvlink_writes_random,
                "Loaded Operation": "Random Write",
                "Memory Region": "HBM3\n(GPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_local,
                "Loaded Operation": "Sequential Read",
                "Memory Region": "LPDDR5X\n(CPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_local_reads_random,
                "Loaded Operation": "Random Read",
                "Memory Region": "LPDDR5X\n(CPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_local_writes,
                "Loaded Operation": "Sequential Write",
                "Memory Region": "LPDDR5X\n(CPU)",
            }
        ),
    ],
    ignore_index=True,
)
df = pd.concat(
    [
        df,
        pd.DataFrame(
            data={
                "Number of Loaded Threads": loaded_threads,
                "Latency in ns": loaded_latency_local_writes_random,
                "Loaded Operation": "Random Write",
                "Memory Region": "LPDDR5X\n(CPU)",
            }
        ),
    ],
    ignore_index=True,
)

sns.set_style("darkgrid")
sns.set_theme(font_scale=1.5)
g = sns.relplot(
    data=df,
    x="Number of Loaded Threads",
    y="Latency in ns",
    hue="Memory Region",
    style="Memory Region",
    markers=True,
    markersize=8,
    col="Loaded Operation",
    kind="line",
    col_wrap=4,
    height=3.5,
    aspect=1.2,
    # facet_kws={'sharey': True}
)

sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.93, 0.35),
    ncol=1,
    title="",
    frameon=False,
    columnspacing=0.5,
    handlelength=1.2,
    handletextpad=0.5,
)

plt.subplots_adjust(wspace=0.03)

g.axes[0].set_ylabel("Latency [ns]")
g.set(xticks=loaded_threads, yticks=np.arange(1100, step=100))
g.set_titles(col_template="{col_name}")
for idx, ax in enumerate(g.axes.flat):
    ax.set_xlabel("# Load Threads")
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]

# ax.set(xticks=xticks, xlabel="Number of Loaded Threads", ylabel="Latency in ns")

# ax.set(xticks=xticks, xlabel="Number of Loaded Threads", ylabel="Latency in ns")
g.savefig("gh200-latencies.pdf")
