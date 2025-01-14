import matplotlib
import numpy as np
import pandas as pd
from builtins import len, enumerate
from matplotlib import pyplot as plt
import seaborn as sns

local_idle = [95]
local_reads_seq = [99, 103, 109, 116, 134, 146, 155, 160, 164, 167, 173, 179, 184, 188, 201]
local_writes_seq = [101, 106, 112, 118, 128, 142, 146, 150, 153, 157, 161, 166, 173, 183, 254]
local_reads_random = [96, 97, 97, 98, 99, 100, 101, 101, 102, 103, 104, 105, 105, 106, 110]
local_writes_random = [98, 101, 104, 107, 110, 113, 117, 118, 120, 122, 124, 127, 130, 133, 149]

local_reads_seq_f = [106, 111, 117, 125, 144, 155, 164, 174, 178, 182, 187, 191, 194, 197, 201]
local_writes_seq_f = [160, 165, 171, 177, 187, 212, 229, 249, 251, 253, 254, 256, 257, 258, 253]
local_reads_random_f = [99, 100, 100, 101, 102, 103, 103, 105, 106, 106, 107, 108, 109, 109, 110]
local_writes_random_f = [104, 107, 110, 113, 116, 119, 122, 126, 129, 131, 134, 138, 142, 146, 149]

intersocket_idle = [227]
intersocket_reads_seq = [228, 231, 237, 249, 275, 316, 364, 409, 449, 482, 510, 534, 554, 562, 570]
intersocket_writes_seq = [228, 231, 232, 232, 232, 232, 234, 234, 234, 234, 234, 234, 235, 235, 839]
intersocket_reads_random = [227, 227, 227, 227, 227, 228, 228, 229, 229, 229, 230, 230, 231, 232, 235]
intersocket_writes_random = [228, 229, 230, 232, 232, 232, 233, 234, 234, 234, 234, 234, 234, 234, 557]

intersocket_reads_seq_f = [232, 235, 241, 253, 279, 319, 367, 416, 457, 491, 518, 542, 560, 567, 570]
intersocket_writes_seq_f = [289, 292, 293, 293, 293, 303, 574, 839, 839, 838, 839, 834, 838, 838, 837]
intersocket_reads_random_f = [228, 229, 229, 229, 229, 229, 230, 232, 232, 232, 233, 233, 234, 234, 235]
intersocket_writes_random_f = [234, 235, 237, 239, 239, 239, 241, 294, 435, 544, 548, 556, 552, 548, 555]

nvlink_idle = [1116]
nvlink_reads_seq = [1111, 1114, 1118, 1123, 1128, 1133, 1140, 1140, 1139, 1141, 1142, 1143, 1143, 1145, 1152]
nvlink_reads_random = [1111, 1107, 1108, 1110, 1113, 1114, 1118, 1121, 1123, 1126, 1128, 1132, 1135, 1139, 1145]
nvlink_writes_seq = [1119, 1154, 1188, 1205, 1213, 1216, 1243, 1237, 1232, 1229, 1226, 1224, 1221, 1219, 1487]
nvlink_writes_random = [1137, 1177, 1197, 1212, 1224, 1242, 1295, 1300, 1304, 1307, 1310, 1312, 1313, 1314, 1517]
nvlink_reads_seq_f = [1112, 1116, 1120, 1125, 1130, 1135, 1140, 1146, 1147, 1147, 1148, 1149, 1150, 1152, 1152]
nvlink_reads_random_f = [1114, 1112, 1109, 1111, 1113, 1115, 1117, 1121, 1124, 1126, 1129, 1132, 1135, 1139, 1145]
nvlink_writes_seq_f = [1293, 1312, 1334, 1347, 1356, 1360, 1379, 1538, 1530, 1521, 1514, 1506, 1499, 1494, 1487]
nvlink_writes_random_f = [1225, 1254, 1270, 1282, 1293, 1307, 1354, 1512, 1513, 1515, 1514, 1516, 1517, 1518, 1517]


loaded_threads = np.arange(16)

df = pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_reads_seq,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "GPU",
})
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_reads_seq_f,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "GPU",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_reads_seq,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Local",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_reads_seq_f,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Local",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_reads_seq,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Remote Socket",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_reads_seq_f,
    "Loaded Operation": "Sequential Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Remote Socket",
})], ignore_index=True)


df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_reads_random,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "GPU",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_reads_random_f,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "GPU",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_reads_random,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Local",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_reads_random_f,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Local",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_reads_random,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Remote Socket",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_reads_random_f,
    "Loaded Operation": "Random Read",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Remote Socket",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_writes_seq,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "GPU",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_writes_seq_f,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "GPU",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_writes_seq,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Local",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_writes_seq_f,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Local",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_writes_seq,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Remote Socket",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_writes_seq_f,
    "Loaded Operation": "Sequential Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Remote Socket",
})], ignore_index=True)


df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_writes_random,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "GPU",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": nvlink_idle + nvlink_writes_random_f,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "GPU",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_writes_random,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Local",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": local_idle + local_writes_random_f,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Local",
})], ignore_index=True)

df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_writes_random,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign Last",
    "Memory Region": "Remote Socket",
})])
df = pd.concat([df, pd.DataFrame(data={
    "Number of Loaded Threads": loaded_threads,
    "Latency in ns": intersocket_idle + intersocket_writes_random_f,
    "Loaded Operation": "Random Write",
    "Meas.-Core Pair": "Assign First",
    "Memory Region": "Remote Socket",
})], ignore_index=True)

sns.set_style("darkgrid")
sns.set( font_scale=1.5)

g = sns.relplot(
    data=df, x="Number of Loaded Threads",
    y="Latency in ns", hue="Memory Region",
    style="Meas.-Core Pair",
    markers=True,
    col="Loaded Operation",
    kind="line",
    col_wrap=2,
)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.4, 1), ncol=2, title=None, frameon=False,
)
g.set_titles(col_template="{col_name}")
g.set(xticks=loaded_threads, yticks=np.arange(1600, step=200))
for ax in g.axes:
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]

# ax.set(xticks=xticks, xlabel="Number of Loaded Threads", ylabel="Latency in ns")
g.savefig("power9-latencies.pdf")
