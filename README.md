# SSMS (Sequential Sampling Model Simulators)

[![DOI](https://zenodo.org/badge/370812185.svg)](https://doi.org/10.5281/zenodo.17156205)
![PyPI](https://img.shields.io/pypi/v/ssm-simulators)
![PyPI_dl](https://img.shields.io/pypi/dm/ssm-simulators)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/lnccbrown/ssm-simulators)](https://github.com/lnccbrown/ssm-simulators/pulls)
![Python Version](https://img.shields.io/pypi/pyversions/ssm-simulators)
[![Run tests](https://img.shields.io/github/actions/workflow/status/lnccbrown/ssm-simulators/run_tests.yml?branch=main&label=tests)](https://github.com/lnccbrown/ssm-simulators/actions/workflows/run_tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/lnccbrown/ssm-simulators/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/ssm-simulators)

Python Package to collect simulators for Sequential Sampling Models.

Find the package documentation [here](https://lnccbrown.github.io/ssm-simulators/).


### Quick Start

The `ssms` package serves two purposes.

1. Easy access to *fast simulators of sequential sampling models*
2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

A number of tutorial notebooks are available under the `/notebooks` directory.

#### Installation

```sh
pip install ssm-simulators
```

**Recommended: Install via conda-forge for full parallel support:**

```sh
conda install -c conda-forge ssm-simulators
```

> [!NOTE]
> **Parallel Execution Requirements:**
>
> For multi-threaded simulation (`n_threads > 1`), the package requires:
> - **OpenMP**: For parallel loop execution
> - **GSL (GNU Scientific Library)**: For validated random number generation
>
> **conda-forge users**: Both dependencies are automatically included.
>
> **pip users**: Install system dependencies first:
> ```bash
> # macOS
> brew install libomp gsl
>
> # Ubuntu/Debian
> sudo apt-get install libgomp-dev libgsl-dev
> ```
> Then reinstall: `pip install --force-reinstall ssm-simulators`
>
> Without these dependencies, the package works in single-threaded mode using NumPy.

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

- `--config-path`: Path to your YAML configuration file (optional, uses default if not provided).
- `--output`: Directory where generated data will be saved (required).
- `--n-files`: (Optional) Number of data files to generate. Default is `1` file.
- `--estimator-type`: (Optional) Likelihood estimator type (`kde` or `pyddm`). Overrides YAML config if specified.
- `--log-level`: (Optional) Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default is `WARNING`.

Below is a sample YAML configuration you can use with the `generate` command:

```yaml
MODEL: 'ddm'
GENERATOR_APPROACH: 'lan'

PIPELINE:
  N_PARAMETER_SETS: 100
  N_SUBRUNS: 20

SIMULATOR:
  N_SAMPLES: 2000
  DELTA_T: 0.001

TRAINING:
  N_SAMPLES_PER_PARAM: 200

ESTIMATOR:
  TYPE: 'kde'  # Options: 'kde' (default) or 'pyddm'
```

Configuration file parameter details follow.

**Top-Level Parameters:**
| Option | Definition |
| ------ | ---------- |
| `MODEL` | The type of model you want to simulate (e.g., `ddm`, `angle`, `levy`) |
| `GENERATOR_APPROACH` | Type of generator used to generate data (`lan` or `cpn`) |

**PIPELINE Section:**
| Option | Definition |
| ------ | ---------- |
| `N_PARAMETER_SETS` | Number of parameter vectors that are used for training |
| `N_SUBRUNS` | Number of repetitions of each call to generate data |

**SIMULATOR Section:**
| Option | Definition |
| ------ | ---------- |
| `N_SAMPLES` | Number of samples a simulation run should entail for a given parameter set |
| `DELTA_T` | Time discretization step used in numerical simulation of the model. Interval between updates of evidence-accumulation. |

**TRAINING Section:**
| Option | Definition |
| ------ | ---------- |
| `N_SAMPLES_PER_PARAM` | Number of times the kernel density estimate (KDE) is evaluated after creating the KDE from simulations of each set of model parameters |

**ESTIMATOR Section:**
| Option | Definition |
| ------ | ---------- |
| `TYPE` | Likelihood estimator type: `kde` (default) or `pyddm` |

To make your own configuration file, you can copy the example above into a new `.yaml` file and modify it with your preferences.

If you are using `uv` (see below), you can use the `uv run` command to run `generate` from the command line

This will generate training data according to your configuration and save it in the specified output directory.

### Key Features

#### Custom Parameter Transforms

Register custom transformations to apply model-specific modifications to sampled parameters:

```python
from ssms import register_transform_function
import numpy as np

# Register a custom transform
def exponential_drift(theta: dict) -> dict:
    if 'v' in theta:
        theta['v'] = np.exp(theta['v'])
    return theta

register_transform_function("exp_v", exponential_drift)

# Use in model configuration
model_config = {
    "name": "my_model",
    "params": ["v", "a", "z", "t"],
    "param_bounds": [...],
    "parameter_transforms": [
        {"type": "exp_v"}  # Your custom transform
    ]
}
```

### Tutorial

Check the [basic tutorial](https://lnccbrown.github.io/ssm-simulators/basic_tutorial/basic_tutorial/) in our documentation.

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

### Contributing

We welcome contributions from the community! Whether you want to add a new model, improve documentation, or fix bugs, your help is appreciated.

#### Contributing New Models

Want to add your own sequential sampling model to the package? Check out our comprehensive guide:

**[📖 Contributing New Models Tutorial](https://lnccbrown.github.io/ssm-simulators/contributing/add_models/)**

This guide walks you through three levels of contribution:
- **Level 1**: Add boundary/drift variants (~15 min)
- **Level 2**: Implement Python simulators (~20 min)
- **Level 3**: Create high-performance Cython implementations (~30 min)

#### Other Contributions

For bug reports, feature requests, or general questions:
- Open an issue on [GitHub Issues](https://github.com/lnccbrown/ssm-simulators/issues)
- Check existing issues to avoid duplicates
- Provide clear descriptions and reproducible examples

### Cite `ssm-simulators`

Please use the this DOI to cite ssm-simulators: [https://doi.org/10.5281/zenodo.17156205](https://doi.org/10.5281/zenodo.17156205)
