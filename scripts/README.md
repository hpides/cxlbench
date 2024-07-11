
## Python Tooling
We define a [pyproject.toml](../pyproject.toml) to manage python tooling and dependencies.

### Python dependency management
We use [Python-Poetry](https://python-poetry.org/docs) as our package manager.
All dependencies are managed by poetry and automatically added the pyproject.toml.
[poetry.lock](../poetry.lock) is managed by poetry.

To install the python dependencies, run:

```shell script
$ poetry install
```
This will setup a your poetry environment.

To execute python modules with your poetry environment, run:
```
$ poetry run <path-to-module>
```


Find further documentation on how to manage python dependencies [here](https://python-poetry.org/docs/basic-usage/)

### Linting and Formatting
We use [Ruff](https://docs.astral.sh/ruff/) for Linting and Formatting.
[lint.sh](lint.sh) and [format.sh](format.sh) will take care of running ruff.

Alternatively, you can run ruff using ```poetry run ruff <ruff_args>```.
For example, ```poetry run ruff check --fix``` will fix linting errors for you.
The configuration of ruff is defined in [pyproject.toml](../pyproject.toml).

## Visualizing Results
MemA bench offers different Python visualization scripts to plot measurement results.
Provided scripts in the `viz` directory:

### `compare.py`
This script allows comparing the measurements of two different benchmark runs. It expects the paths to the directory in
which the json files contianing the measurements per setup can be found. In addition, labels for the two setups are
required. Optionally, an output path for the resulting plots can be provided.
Example:
```shell script
poetry run viz/compare.py ./results/cxl ./results/dram CXL DRAM
```

The visualization produces PDF files in the output directory or in `./plots/` if no output directory is provided.
Note that each benchmark name must be unique and visualizations are only generated for benchmarks with a maximum of 3 matrix arguments.
