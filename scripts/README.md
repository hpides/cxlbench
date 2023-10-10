## Visualizing Results

MemA bench offers differeny Python visualization scripts to plot measurement results. To do that, you have to create a
Python virtual environment first. The script `setup_viz.sh`, which requires Python3, Pip3 and python3-venv, must be
sourced:
```shell script
$ source setup_viz.sh
```

Provided scripts in the `viz` directory:

### `compare.py`
This script allows comparing the measurements of two different benchmark runs. It expects the paths to the directory in
which the json files contianing the measurements per setup can be found. In addition, labels for the two setups are
required. Optionally, an output path for the resulting plots can be provided.
Example:
```shell script
python3 viz/compare.py ./results/cxl ./results/dram CXL DRAM
```

The visualization produces PDF files in the output directory or in `./plots/` if no output directory is provided.
Note that each benchmark name must be unique and visualizations are only generated for benchmarks with a maximum of 3 matrix arguments.
