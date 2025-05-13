# SSMS (Sequential Sampling Model Simulators)
Python Package which collects simulators for Sequential Sampling Models.

Find the package documentation [here](https://lnccbrown.github.io/ssm-simulators/).

![PyPI](https://img.shields.io/pypi/v/ssm-simulators)
![PyPI_dl](https://img.shields.io/pypi/dm/ssm-simulators)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/lnccbrown/ssm-simulators/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/ssm-simulators)

### Quick Start

The `ssms` package serves two purposes.

1. Easy access to *fast simulators of sequential sampling models*
2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

A number of tutorial notebooks are available under the `/notebooks` directory.

#### Installation

```sh
pip install ssm-simulators
```

> [!NOTE]
> Building from source or developing this package requires a C compiler (such as GCC).
> On Linux, you can install GCC with:
> ```bash
> sudo apt-get install build-essential
> ```
> Most users installing from PyPI wheels do **not** need to install GCC.

#### Command Line Interface
The package exposes a command-line tool, `generate`, for creating training data from a YAML configuration file.

```bash
generate --config-path <path/to/config.yaml> --output <output/directory> [--log-level INFO]
```

- `--config-path`: Path to your YAML configuration file (required).
- `--output`: Directory where generated data will be saved (required).
- `--log-level`: (Optional) Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default is `WARNING`.

Below is a sample YAML configuration you can use with the `generate` command:

```yaml
MODEL: 'ddm'
N_SAMPLES: 2000
N_PARAMETER_SETS: 100
DELTA_T: 0.001
N_TRAINING_SAMPLES_BY_PARAMETER_SET: 200
N_SUBRUNS: 20
GENERATOR_APPROACH: 'lan'
```

**Example:**

```bash
generate --config-path myconfig.yaml --output ./output --log-level INFO
```

This will generate training data according to your configuration and save it in the specified output directory.

### Tutorial

Check the basic tutorial [here](docs/basic_tutorial/basic_tutorial.ipynb).

### Advanced: Dependency Management with uv

We use `uv` for fast and efficient dependency management. To get started:

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies (including development):
```bash
uv sync --all-groups  # Installs all dependency groups
```
