# SSMS (Sequential Sampling Model Simulators)
Python Package which collects simulators for Sequential Sampling Models.

Find the package documentation [here](https://lnccbrown.github.io/ssm-simulators/).

![PyPI](https://img.shields.io/pypi/v/ssm-simulators)
![PyPI_dl](https://img.shields.io/pypi/dm/ssm-simulators)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/lnccbrown/ssm-simulators/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/ssm-simulators)

### Quick Start

The `ssms` package serves two purposes.

1. Easy access to *fast simulators of sequential sampling models*
2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

We provide two minimal examples here to illustrate how to use each of the two capabilities.

#### Install

Let's start with *installing* the `ssms` package.

You can do so by typing,

`pip install ssm-simulators`

in your terminal.

Below you find a basic tutorial on how to use the package.

#### Tutorial

Check the basic tutorial [here](docs/basic_tutorial/basic_tutorial.ipynb).

### Dependency Management with uv

We use `uv` for fast and efficient dependency management. To get started:

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies (including development):
```bash
uv sync --all-groups  # Installs all dependency groups
```
