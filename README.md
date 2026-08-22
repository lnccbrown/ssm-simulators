<div style="position: relative; width: 100%;">
  <img src="docs/images/mainlogo.png" style="width: 175px;">
  <a href="https://ccbs.carney.brown.edu/brainstorm" style="position: absolute; right: 0; top: 50%; transform: translateY(-50%);">
    <img src="docs/images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
  </a>
</div>

# SSMS: Sequential Sampling Model Simulators

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17156205-blue)](https://doi.org/10.5281/zenodo.17156205)
![PyPI](https://img.shields.io/pypi/v/ssm-simulators)
[![Downloads](https://static.pepy.tech/badge/ssm-simulators/month)](https://pepy.tech/projects/ssm-simulators)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/lnccbrown/ssm-simulators)](https://github.com/lnccbrown/ssm-simulators/pulls)
[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/ssm-simulators/)
[![Run tests](https://img.shields.io/github/actions/workflow/status/lnccbrown/ssm-simulators/run_tests.yml?branch=main&label=tests)](https://github.com/lnccbrown/ssm-simulators/actions/workflows/run_tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/lnccbrown/ssm-simulators/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/ssm-simulators)

`ssm-simulators` provides fast C/Cython simulators for sequential sampling
models used in cognitive science, neuroscience, and amortized Bayesian
inference. It is the simulator and training-data layer of the
[HSSM ecosystem](https://lnccbrown.github.io/HSSM/ecosystem/).

## Install

```bash
pip install ssm-simulators
```

The [documentation](https://lnccbrown.github.io/ssm-simulators/) is the
canonical source for optional backends, OpenMP/GSL prerequisites, tutorials,
model coverage, configuration, and API contracts.

## Start here

- [Simulate and inspect your first SSM](https://lnccbrown.github.io/ssm-simulators/basic_tutorial/basic_tutorial/)
- [Simulate your first RLSSM](https://lnccbrown.github.io/ssm-simulators/core_tutorials/rlssm_tutorial/)
- [Create a custom model](https://lnccbrown.github.io/ssm-simulators/core_tutorials/tutorial_custom_models/)
- [Browse the API reference](https://lnccbrown.github.io/ssm-simulators/api/basic_simulators/)
- [Report an issue](https://github.com/lnccbrown/ssm-simulators/issues)

## Contributing

Install the development environment, then run the package and documentation
gates:

```bash
uv sync --extra dev
uv run pytest tests/
uv run pytest tests/test_notebooks.py --run-notebooks --no-cov -v
uv run ruff check .
uv run ruff format --check .
./scripts/docs.sh build
```

Preview documentation changes with `./scripts/docs.sh serve`. See the
[contribution guide](https://lnccbrown.github.io/ssm-simulators/contributing/)
and [new-model guide](https://lnccbrown.github.io/ssm-simulators/contributing/add_models/)
for durable development guidance.

## Citation

Please cite `ssm-simulators` with its
[Zenodo DOI](https://doi.org/10.5281/zenodo.17156205).

## License

`ssm-simulators` is distributed under the [MIT License](LICENSE).
