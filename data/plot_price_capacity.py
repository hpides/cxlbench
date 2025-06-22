#! /usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Cpu:
  def __init__(self, id, mem_channels, dimms_per_channel, price_usd):
    self.id = id
    self.mem_channels = mem_channels
    self.dimms_per_channel = dimms_per_channel
    self.price = price_usd

  def dimm_count(self):
    return self.mem_channels * self.dimms_per_channel

class Dimm:
  def __init__(self, ddr_version, capacity, price_usd):
    self.ddr_version = ddr_version
    self.capacity = capacity
    self.price = price_usd

  @staticmethod
  def record_field_names():
    return ["ddr","capacity_gib", "price_usd"]

  def to_record(self):
    return [self.ddr_version, self.capacity, self.price]

class CxlDevice:
  def __init__(self, dimm_count, dimm):
    self.dimm_count = dimm_count
    self.dimm = dimm

  def mem_price(self):
    return self.dimm_count * self.dimm.price

  def mem_capacity(self):
    return self.dimm_count * self.dimm.capacity

class Config:
    def __init__(self, cpu, cpu_count, cpu_dimm, cxl_device_count, cxl_device):
      self.cpu = cpu
      self.cpu_count = cpu_count
      self.cpu_dimm = cpu_dimm
      self.cxl_device_count = cxl_device_count
      self.cxl_device = cxl_device

    def cpu_mem_capacity(self):
      return self.cpu_count * self.cpu.mem_channels * self.cpu.dimms_per_channel * self.cpu_dimm.capacity

    def cxl_mem_capacity(self):
      return self.cxl_device_count * self.cxl_device.mem_capacity()

    def mem_capacity(self):
      return self.cpu_mem_capacity() + self.cxl_mem_capacity()

    def cpu_price(self):
      return self.cpu_count * self.cpu.price

    def cpu_mem_price(self):
      return self.cpu_count * self.cpu.mem_channels * self.cpu.dimms_per_channel * self.cpu_dimm.price

    def cxl_mem_price(self):
      return self.cxl_device_count * self.cxl_device.mem_price()

    def price(self):
      return self.cpu_price() + self.cpu_mem_price() + self.cxl_mem_price()

    @staticmethod
    def record_field_names():
      return [
        "cpu_id",
        "cpus",
        "cpu_dimms",
        "ddr",
        "cpu_dimm_size",
        "cpu_dimm_price",
        "devs",
        "dev_dimms",
        "dev_ddr",
        "dev_dimm_size",
        "dev_dimm_price",
        "cpu_capacity",
        "cxl_capacity",
        "capacity",
        "price"
      ]

    def to_record(self):
      return [
        self.cpu.id,
        self.cpu_count,
        self.cpu.dimm_count(),
        self.cpu_dimm.ddr_version,
        self.cpu_dimm.capacity,
        self.cpu_dimm.price,
        self.cxl_device_count,
        self.cxl_device.dimm_count,
        self.cxl_device.dimm.ddr_version,
        self.cxl_device.dimm.capacity,
        self.cxl_device.dimm.price,
        self.cpu_mem_capacity(),
        self.cxl_mem_capacity(),
        self.mem_capacity(),
        self.price()
      ]

### Define Hardware
cpus = [
  # Intel's recommended customer price: $3995. We round to the nearest multiple of ten.
  Cpu("8452Y", 8, 2, 4000)
]

# Prices found on newegg.com rounded up to the nearest multiple of five.
# Prices per dimm calculated by deviding the onle price by the number of DIMMs. Resulting price rounded to
# integer numbers.
# 288 pin DRAM DIMMs
dimms = {
  "null" : Dimm(0, 0, 0),
  # 12.5 $/GB
  "ddr5-256gb" : Dimm(5, 256, 3200), # $6400 for 2x256, https://www.newegg.com/p/pl?N=100007950%20601349782&d=DDR5+512+GB+NEMIX
  # 7.4 $/GB
  "ddr5-128gb" : Dimm(5, 128, 948), # $3790 for 4x128 https://www.newegg.com/p/pl?N=100007950%20601349782&d=DDR5+512+GB
  # 5.4 $/GB
  "ddr5-64gb" : Dimm(5, 64, 346), # $2770 for 8x64 https://www.newegg.com/p/pl?N=100007950%20601349784&d=DDR5+512+GB
  # 4.3 $/GB
  "ddr5-32gb" : Dimm(5, 32, 137), # $2190 for 16x32GB https://www.newegg.com/p/pl?N=100007950%20601349789&d=DDR5+512+GB
  # 5.3 $/GB
  "ddr5-16gb" : Dimm(5, 16, 85), # $85 for 1x 16GB https://www.newegg.com/p/pl?d=DDR5+NEMIX+4800+16GB+Server
  ### DDR4: 3200 MHz
  # 5 $/GB
  "ddr4-256gb" : Dimm(4, 256, 1270), # 2540 for 2x256 https://www.newegg.com/p/pl?d=DDR4+NEMIX+256GB+3200+Server
  # 5.1 $/GB
  "ddr4-128gb" : Dimm(4, 128, 650), # 650 for 1x128 https://www.newegg.com/p/pl?N=600564407%20100007952%20601324426&d=DDR4+NEMIX
  # 2 $/GB
  "ddr4-64gb" : Dimm(4, 64, 125), # 125 for 1x64 https://www.newegg.com/p/pl?d=DDR4+NEMIX+64GB+3200+Server
  # 2.2 $/GB
  "ddr4-32gb" : Dimm(4, 32, 70), # 70 for 1x32 https://www.newegg.com/p/pl?d=DDR4+NEMIX+32GB+3200+Server
  # 1.9 $/GB
  "ddr4-16gb" : Dimm(4, 16, 30), # 30 for 1x16 https://www.newegg.com/p/pl?d=DDR4+NEMIX+3200+16GB
  # Old DDR4 dimms have the same specifications as the ddr4 DIMMs above, but a price of 0.
  # "old-ddr4-256gb" : Dimm(4, 256, 0),
  # "old-ddr4-128gb" : Dimm(4, 128, 0),
  # "old-ddr4-64gb" : Dimm(4, 64, 0),
  # "old-ddr4-32gb" : Dimm(4, 32, 0),
  # "old-ddr4-16gb" : Dimm(4, 16, 0)
}

devices = {
  "null" : CxlDevice(0, dimms["null"]),
  "ddr4-128gb" : CxlDevice(8, dimms["ddr4-128gb"]),
  # "reused-ddr4-128gb" : CxlDevice(8, dimms["old-ddr4-128gb"])
}

cpu_mem_configs = [
  # 1 CPU
  Config(cpus[0], 1, dimms["ddr5-16gb"], 0, devices["null"]),
  Config(cpus[0], 1, dimms["ddr5-32gb"], 0, devices["null"]),
  Config(cpus[0], 1, dimms["ddr5-64gb"], 0, devices["null"]),
  Config(cpus[0], 1, dimms["ddr5-128gb"], 0, devices["null"]),
  Config(cpus[0], 1, dimms["ddr5-256gb"], 0, devices["null"]),
  # 2 CPUs
  Config(cpus[0], 2, dimms["ddr5-16gb"], 0, devices["null"]),
  Config(cpus[0], 2, dimms["ddr5-32gb"], 0, devices["null"]),
  Config(cpus[0], 2, dimms["ddr5-64gb"], 0, devices["null"]),
  Config(cpus[0], 2, dimms["ddr5-128gb"], 0, devices["null"]),
  Config(cpus[0], 2, dimms["ddr5-256gb"], 0, devices["null"]),
]

cpu_1to4cxl_mem_configs = [
  # CPU mem: 16 GiB
  Config(cpus[0], 1, dimms["ddr5-16gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-16gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-16gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-16gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 32 GiB
  Config(cpus[0], 1, dimms["ddr5-32gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-32gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-32gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-32gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 64 GiB
  Config(cpus[0], 1, dimms["ddr5-64gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-64gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-64gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-64gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 128 GiB
  Config(cpus[0], 1, dimms["ddr5-128gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-128gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-128gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-128gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 256 GiB
  Config(cpus[0], 1, dimms["ddr5-256gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-256gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-256gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-256gb"], 4, devices["ddr4-128gb"]),
  ### 2 CPUs
  # CPU mem: 16 GiB
  Config(cpus[0], 2, dimms["ddr5-16gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-16gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-16gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-16gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 32 GiB
  Config(cpus[0], 2, dimms["ddr5-32gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-32gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-32gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-32gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 64 GiB
  Config(cpus[0], 2, dimms["ddr5-64gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-64gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-64gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-64gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 128 GiB
  Config(cpus[0], 2, dimms["ddr5-128gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-128gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-128gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-128gb"], 4, devices["ddr4-128gb"]),
  # CPU mem: 256 GiB
  Config(cpus[0], 2, dimms["ddr5-256gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-256gb"], 2, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-256gb"], 3, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-256gb"], 4, devices["ddr4-128gb"]),
]

cpu_cxl_mem_configs = [
  Config(cpus[0], 1, dimms["ddr5-16gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-32gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-64gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-128gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 1, dimms["ddr5-256gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-16gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-32gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-64gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-128gb"], 1, devices["ddr4-128gb"]),
  Config(cpus[0], 2, dimms["ddr5-256gb"], 1, devices["ddr4-128gb"]),
]

invalid_config = Config(Cpu("invalid", 100, 100, 9999999), 1, Dimm(0, 0, 9999999), 0, devices["null"])

### Data points
data = []
capacity_steps = 256

for capacity_demand in range(capacity_steps, 8192 + 1, capacity_steps):
  ##### CPU mem
  cheapest_config = invalid_config

  for config in cpu_mem_configs:
    if config.mem_capacity() >= capacity_demand and config.price() < cheapest_config.price():
      cheapest_config = config
  
  assert cheapest_config is not invalid_config
  data.append([capacity_demand, "CPU Memory", 1] + cheapest_config.to_record())
  baseline_price = cheapest_config.price()

  ##### CPU mem + 1 CXL device
  cheapest_config = invalid_config

  for config in cpu_cxl_mem_configs:
    if config.mem_capacity() >= capacity_demand and config.price() < cheapest_config.price():
      cheapest_config = config

  assert cheapest_config is not invalid_config
  data.append([capacity_demand, "CPU Memory +\n1 CXL Device", cheapest_config.price()/baseline_price] + cheapest_config.to_record())

  ##### CPU mem + 1-4 CXL devices
  cheapest_config = invalid_config

  for config in cpu_1to4cxl_mem_configs:
    if config.mem_capacity() >= capacity_demand and config.price() < cheapest_config.price():
      cheapest_config = config

  assert cheapest_config is not invalid_config
  data.append([capacity_demand, "CPU Memory +\n1 to 4 CXL Devices", cheapest_config.price()/baseline_price] + cheapest_config.to_record())

df = pd.DataFrame(data, columns=["capacity_demand", "type","price_factor"] + Config.record_field_names())
df.to_csv("./cheapest_config_per_capacity_demand.csv")

### Plot
df["price"] = df["price"]/1000
df["capacity_demand"] = df["capacity_demand"]/1024

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{libertine}'
})

hpi_col = ["#f5a700","#dc6007","#b00539", "#6b009c", "#006d5b", "#0073e6", "#e6007a", "#00C800", "#FFD500", "#0033A0"]

fig = plt.figure(figsize=(2.7, 1.5))
ax = sns.lineplot(
  data=df, y='price', x='capacity_demand', hue='type', style='type', markersize=5, palette=hpi_col)
ax.set_ylabel("Price [Thousand \$]\nof CPUs and DIMMs")
ax.set_xlabel("Memory capacity demand [TiB]")
ax.set_title("")
ax.grid(axis='both', which='major', alpha=0.7, zorder=1)
ax.grid(axis='both', which='minor', alpha=0.2, zorder=1)
# ax.set_yscale('log')
plt.xlim(0)
plt.ylim(0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.get_legend().remove()

fig.legend(title="", loc="upper center", ncol=1, bbox_to_anchor=(1.17, 0.887),
        columnspacing=0.5,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=1,
        borderpad=0.2)

plt.savefig(
  "./{}.pdf".format("cheapest_config_per_capacity_demand"), bbox_inches="tight", pad_inches=0
)

### Old plot approach
# data = []

# for config in configs:
#   data.append(config.to_record())

# df = pd.DataFrame(data, columns=Config.record_field_names())
# # Capacity in 
# df = df.sort_values(by=["devs", "cpus", "cpu_dimm_size"], ascending=[True, True, True])
# df["price"] = df["price"] / 1000
# df["cpu_dimm_size"] = df["cpu_dimm_size"].astype(str)
# df["share_cxl_mem"] = df["cxl_capacity"] / df["capacity"]
# # df["desc"] = (
# #     np.where(df["cpus"] == 1, "1 CPU ",
# #       df["cpus"].astype(str) + " CPUs ") +
# #     np.where(df["devs"] > 0,
# #         " (16$\\times$" + df["cpu_dimm_size"] + " GiB)",
# #         np.where(df["cpus"] == 1, "(16$\\times$16/32/64/128/256 GiB)",
# #           "(" + (df["cpus"]*16).astype(str) + "$\\times$16/32/64/128/256 GiB)")
# #     ) +
# #     np.where(df["devs"] > 0, " + 1/2/3/4 CXL", " ") +
# #     np.where((df["dev_dimm_price"] == 0) & (df["devs"] > 0), " (reused)", "")
# # )
# df["desc"] = (
#     np.where(df["cpus"] == 1, "1 CPU ",
#       df["cpus"].astype(str) + " CPUs ") +
#     np.where(df["devs"] > 0, 
#         "(" + ((16*df["cpu_dimm_size"].astype(int))/1024).astype(str) + " TiB)", ""
#     ) +
#     np.where(df["devs"] > 0, " + CXL", " ") +
#     np.where((df["dev_dimm_price"] == 0) & (df["devs"] > 0), " (reuse)", "")
# )

# print(df)
# df["capacity"] = df["capacity"] / 1024
# df_csv = df.copy()
# df_csv = df_csv.sort_values(by=["capacity"])
# df_csv.to_csv("./data_price_capacity.csv")

# # Plot
# hpi_col = ["#f5a700","#dc6007","#b00539", "#6b009c", "#006d5b", "#0073e6", "#e6007a", "#00C800", "#FFD500", "#0033A0" ]

# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'text.latex.preamble': r'\usepackage{libertine}'
# })

# descs = df["desc"].unique()
# hue_order = [descs[0],descs[2],descs[4],descs[6],descs[1],descs[3],descs[5],descs[7]]
# colors = [hpi_col[0],hpi_col[2],hpi_col[4],hpi_col[6],hpi_col[1],hpi_col[3],hpi_col[5],hpi_col[7]]

# # fig, ax = plt.subplots(1, 2, figsize=(3.55, 1.75))
# fig, ax = plt.subplots(1, 1, figsize=(4.5, 0.8))
# ax = [ax]
# # sns.lineplot(ax=ax, data=df, y='capacity', x='price', hue='desc', style='desc', markers=True, dashes=True, markersize=5, palette=hpi_col)
# sns.scatterplot(ax=ax[0], data=df, y='capacity', x='price', hue='desc', hue_order=hue_order, style='desc', markers=True, palette=colors, s=15, zorder=2)
# ax[0].set_ylabel("Memory Capacity\n[TiB]")
# ax[0].set_xlabel("CPU \& Memory Price [Thousand \$]")
# ax[0].set_title("")
# ax[0].grid(axis='both', which='both', alpha=0.3, zorder=1)
# plt.xlim(0)
# plt.ylim(0)
# ax[0].get_legend().remove()
# ax[0].yaxis.set_major_locator(ticker.MultipleLocator(2))
# ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))
# ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
# ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
# price_full_cpu_16gb = Config(cpus[0], 1, dimms["ddr5-16gb"], 0, devices["null"]).price() / 1000
# price_full_cpu_64gb = Config(cpus[0], 1, dimms["ddr5-64gb"], 0, devices["null"]).price() / 1000
# price_full_cpu_256gb = Config(cpus[0], 1, dimms["ddr5-256gb"], 0, devices["null"]).price() / 1000
# ax[0].axvline(x=price_full_cpu_16gb, color='gray', linewidth=1, linestyle='dashed', alpha=0.7, zorder=1)
# ax[0].axvline(x=price_full_cpu_64gb, color='gray', linewidth=1, linestyle='dashed', alpha=0.7, zorder=1)
# ax[0].axvline(x=price_full_cpu_256gb, color='gray', linewidth=1, linestyle='dashed', alpha=0.7, zorder=1)

# fig.legend(title="", loc="upper center", ncol=2, bbox_to_anchor=(0.515, 1.75),
#         columnspacing=0.5,
#         handlelength=0.6,
#         handletextpad=0.2,
#         labelspacing=0.1,
#         borderpad=0.1)

# plt.savefig(
#   "./{}.pdf".format("price_capacity"), bbox_inches="tight", pad_inches=0
# )

# ############ DDR DIMM prices
# fig, ax = plt.subplots(1, 1, figsize=(4, 0.7))
# ax = [ax]
# idx = 0

# dimm_data = []
# for name, dimm in dimms.items():
#   if not name.startswith("ddr"):
#     continue
#   dimm_data.append(dimm.to_record())

# df = pd.DataFrame(dimm_data, columns=Dimm.record_field_names())
# df = df.sort_values(by=["ddr", "capacity_gib"], ascending=[True, True])
# df["capacity_gib"] = df["capacity_gib"].astype(str)
# df["ddr"] = "DDR" + df["ddr"].astype(str)

# sns.barplot(ax=ax[idx], data=df, y="capacity_gib", x="price_usd", hue="ddr", palette=[hpi_col[0],hpi_col[4]])
# ax[idx].set_xlabel("Price [\$]", labelpad=-10)
# ax[idx].xaxis.set_label_coords(-0.12, -0.18)
# ax[idx].set_ylabel("Capacity\n[GiB]")
# ax[idx].set_title("")
# ax[idx].xaxis.set_major_locator(ticker.MultipleLocator(500))
# ax[idx].xaxis.set_minor_locator(ticker.MultipleLocator(100))
# ax[idx].grid(axis='x', which="major", alpha=0.5, zorder=1)
# ax[idx].grid(axis='x', which="minor", alpha=0.2, zorder=1)
# ax[idx].get_legend().remove()

# fig.legend(title="", loc="upper center", ncol=1, bbox_to_anchor=(0.82, 0.9),
#         columnspacing=0.5,
#         handlelength=0.8,
#         handletextpad=0.3,
#         labelspacing=0.3,
#         fontsize=9)

# plt.savefig(
#   "./{}.pdf".format("dimm_prices"), bbox_inches="tight", pad_inches=0
# )