#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data setup
data = {
    'Aspect': [
        'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)',
        'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)',
        'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)',
        'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)',
        'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)',
        
        'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)', 'Backend\nBound\n(Slots)',
        'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)', 'Memory\nBound\n(Slots)',
        'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)', 'DRAM\nBound\n(Stalls)',
        'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)', 'Memory BW\n(Cycles)',
        'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)', 'Memory Lat\n(Cycles)'
    ],
    'Placement': [
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades",
        'AllLocal', "Col'CXL1Blade", "Col'CXL4Blades", "AllCXL1Blade", "AllCXL4Blades"
    ],
    'Selectivity': [
        '100\%']*25 + ['0.1\%']*25,
    'Percentage': [
        69.6, 84.1, 75.5, 84.8, 79.0, 
        48.5, 75.3, 62.9, 76.1, 68.5,
        54.0, 79.9, 74.2, 74.7, 72.0,
        56.5, 83.8, 80.9, 81.9, 80.1,
        19.4, 3.9, 5.9, 3.6, 5.9,
        
        68.3, 83.2, 75.6, 83.6, 78.3, 
        48.1, 73.9, 66.8, 75.2, 68.7,
        56.3, 79.7, 75.3, 77.7, 73.6,
        54.7, 83.4, 80.7, 80.7, 79.0,
        20.3, 3.9, 6.1, 3.9, 6.1
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Custom color palette
hpi_palette = ["#f5a700", "#dc6007", "#b00539"]
palette = hpi_palette + ["#6b009c", "#006d5b"]

# Font
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{libertine}'
})

# Hatching
hatches = ["/", "\\", "X", "o", "//", "xx"]
fontsize = 32
plt.rcParams.update({"font.size": fontsize})
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["legend.title_fontsize"] = fontsize
plt.rcParams["hatch.linewidth"] = 3

# Plotting with seaborn's FacetGrid to create subplots by 'Selectivity'
g = sns.FacetGrid(df, row='Selectivity', aspect=1, height=6, sharey=True)
g.map_dataframe(sns.barplot, x='Aspect', y='Percentage', hue='Placement', palette=palette, errorbar=None)
g.set_xlabels("")

# Adjust each subplot
for ax in g.axes.flat:
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="y", color="black", linestyle=":")
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

# Adjust legend placement above the plots
g.add_legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
            frameon=True, handlelength=0.8, handletextpad=0.3, columnspacing=0.5)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
            # frameon=True, handlelength=0.8, handletextpad=0.3, columnspacing=0.5)

plt.tight_layout()

# Save the figure
plt.savefig("./scan-toplev.pdf", bbox_inches="tight", pad_inches=0)
# plt.show()
