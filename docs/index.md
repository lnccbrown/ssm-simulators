<div>
    <a href="https://ccbs.carney.brown.edu/brainstorm" style="display: block; float: right; padding: 10px">
        <img src="images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
    </a>
    <img src="images/mainlogo.png" style="width: 175px;">
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
inference, spanning classic DDM variants, multi-choice models, attention
models, and reinforcement-learning SSMs.

---

## What ssms Provides

| Area | Documentation path |
| --- | --- |
| Direct SSM simulation | [Basic tutorial](basic_tutorial/basic_tutorial.ipynb), [basic simulator API](api/basic_simulators.md) |
| Model configuration | [Configuration systems](core_tutorials/tutorial_configs.ipynb), [config API](api/config.md) |
| Training data generation | [Data generators](core_tutorials/tutorial_data_generators.ipynb), [dataset generator API](api/dataset_generators.md) |
| RLSSM simulation | [RLSSM tutorial](core_tutorials/rlssm_tutorial.ipynb), [RLSSM API](api/rlssm.md) |
| Choice-only RL models | [Choice-only RL tutorial](core_tutorials/choice_only_rl_models.ipynb), [RLSSM API](api/rlssm.md#choice-only-inverse-temperature-softmax-presets) |
| New model contributions | [Contribute new models](contributing/add_models.md), [parameter adapters](contributing/add_parameter_adapters.md) |

---

## Model Families

ssms covers a broad simulator surface:

- **Diffusion models**: DDM, full DDM, deadline variants, angle and Weibull
  boundaries, Levy, Ornstein-Uhlenbeck, gamma-drift, conflict, tradeoff, and
  shrink-spotlight variants.
- **Multi-choice accumulators**: race, racing diffusion, LBA, LBA4, LCA, and
  Poisson race models.
- **Attention and fixation-conditioned models**: aDDM simulators with observed
  or self-sampled fixation inputs, continuation strategies, and optional
  trajectory metadata.
- **Reinforcement-learning SSMs**: Rescorla-Wagner learning rules, RT + choice
  models, inverse-temperature softmax choice-only models, and posterior
  predictive functions for response-only RL workflows.

!!! note "Choice-only RL support"
    ssms includes inverse-temperature softmax decision processes for two-,
    three-, and four-choice settings. Built-in RL presets include RT + choice
    Rescorla-Wagner DDM/angle/Weibull models, dual-alpha variants, and
    choice-only inverse-temperature softmax bandits such as
    `2AB_RW_InvTempSoftmax`, `3AB_RW_InvTempSoftmax`, and
    `4AB_RW_InvTempSoftmax`.

---

## Ecosystem Fit

`ssm-simulators` is the simulator and data-generation layer of the HSSM
ecosystem.

| Package | Role |
| --- | --- |
| [HSSM](https://github.com/lnccbrown/HSSM) | Consumes simulator-defined model contracts for Bayesian inference, including ssms-defined RLSSMs. |
| [LANfactory](https://github.com/lnccbrown/LANfactory) | Trains likelihood approximation networks from ssms-generated data. |
| [LAN_pipeline_minimal](https://github.com/lnccbrown/LAN_pipeline_minimal) | Runs data-generation and LAN-training pipelines. |

For RLSSMs, ssms owns the learning rule, task environment, response mapping,
simulation loop, and posterior predictive behavior. HSSM consumes the assembled
ssms model through `hssm.rl.RLSSMConfig.from_ssms_model(...)`.

---

## Quick Start

### Classic SSM

```python
from ssms.basic_simulators import Simulator

sim = Simulator("ddm")
out = sim.simulate(
    theta={"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.2},
    n_samples=1000,
)

print(out["rts"].shape, out["choices"].shape)
```

### RLSSM

```python
import ssms.rl as rl

config = rl.preset.get("2AB_RW_InvTempSoftmax")
sim = rl.Simulator(config)

data = sim.simulate(
    theta={"rl_alpha": 0.2, "beta": 2.0},
    n_trials=200,
    n_participants=20,
    random_state=42,
)

response_only = data.drop(columns=["rt"])
config.validate_data(response_only).raise_for_errors()
```

Choice-only simulations emit `rt=-1.0` only as a compatibility placeholder in
generative output. Use response-only data for HSSM handoff and choice-only PPC.

---

## Installation

```sh
pip install ssm-simulators
```

Install the optional JAX backend for differentiable RLSSM learning processes:

```sh
pip install "ssm-simulators[jax]"
```

For full parallel support, conda-forge is recommended:

```sh
conda install -c conda-forge ssm-simulators
```

Pip users who need multi-threaded simulation should install OpenMP and GSL first:

```bash
# macOS
brew install libomp gsl

# Ubuntu/Debian
sudo apt-get install libgomp-dev libgsl-dev
```

---

## Next Steps

- Learn the basic simulator API in the [basic tutorial](basic_tutorial/basic_tutorial.ipynb).
- Explore model families in the [package overview](core_tutorials/tutorial_capabilities.ipynb).
- Generate LAN training data with the [data generator tutorial](core_tutorials/tutorial_data_generators.ipynb).
- Build and simulate RLSSMs with the [RLSSM tutorial](core_tutorials/rlssm_tutorial.ipynb).
- Learn response-only RL simulation in the [choice-only RL tutorial](core_tutorials/choice_only_rl_models.ipynb).
- Use the [API reference](api/ssms.md) when integrating ssms into another package.
