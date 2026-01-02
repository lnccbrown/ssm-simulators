# Contributing New Models to SSM-Simulators

This guide helps academic researchers contribute new sequential sampling models to the `ssm-simulators` codebase. Whether you're a graduate student prototyping a novel model or an established researcher contributing a validated implementation, this tutorial will walk you through the process.

## Table of Contents

1. [Introduction & Prerequisites](#1-introduction--prerequisites)
2. [Understanding the Architecture](#2-understanding-the-architecture)
3. [Level 1: Boundary/Drift Variants](#3-level-1-contributing-boundarydrift-variants)
4. [Level 2: Python Simulators](#4-level-2-contributing-python-simulators)
5. [Level 3: Cython Simulators](#5-level-3-contributing-cython-simulators)
6. [Testing Your Contribution](#6-testing-your-contribution)
7. [Documentation Requirements](#7-documentation-requirements)
8. [Submitting Your PR](#8-submitting-your-pr)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Introduction & Prerequisites

### Who This Guide Is For

This guide is designed for academic researchers—from graduate students to faculty—who want to contribute new models to `ssm-simulators`. We assume you understand sequential sampling model theory; this is not a tutorial on SSM fundamentals.

### What You'll Learn

- How to add new model variants using existing simulators
- How to implement new models in Python for prototyping
- How to optimize models with Cython for production use
- Best practices for testing and documenting contributions

### Which Level Should You Choose?

Use this decision tree to determine your contribution pathway:

```
START: What do you want to contribute?
│
├─ Existing simulator, new boundary/drift?
│  → Level 1 (easiest, no Cython required)
│  Example: DDM with collapsing boundaries
│  Time: ~15 minutes + testing
│
├─ New model class, prototype/moderate use?
│  → Level 2 (Python only)
│  Example: Novel accumulator model, <10k samples typical
│  Time: ~20 minutes + testing
│
└─ New model, production/high-performance?
   → Level 3 (Cython implementation)
   Example: Validated model, >10k samples routinely
   Time: ~30 minutes + testing
```

**Correctness First**: All levels prioritize mathematical correctness over performance. Validate your implementation against theory or published results before optimizing.

### Prerequisites

**Required**:
- Python proficiency (intermediate level)
- Understanding of SSM theory (assumed knowledge)
- Git and GitHub basics for pull requests
- Ability to run tests locally

**Optional**:
- Cython knowledge (Level 3 only)
- Experience with numerical computing
- Familiarity with the package (see [tutorials](../../notebooks/))

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ssm-simulators.git
   cd ssm-simulators
   ```

2. **Install [uv](https://github.com/astral-sh/uv) if needed**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install all dependencies (including development) using uv**:
   ```bash
   uv sync --all-groups
   ```

4. **(Level 3 only) Build Cython extensions**:
   ```bash
   python setup.py build_ext --inplace
   ```

5. **Verify your setup**:
   ```bash
   python -c "from ssms import Simulator; print(Simulator('ddm'))"
   uv pip install pytest  # Only needed if 'pytest' was not installed above
   uv pip run pytest tests/ -k test_ddm --verbose
   ```

---

## 2. Understanding the Architecture

Before contributing, understand how `ssm-simulators` components connect:

### Architecture Diagram

Below is a simplified overview of the main architectural layers for how new models integrate with the package:

| Layer                       | Description                                  | Example Files/Functions         |
|-----------------------------|----------------------------------------------|---------------------------------|
| **User Interface**          | Entry point for users; creates simulator     | `Simulator("my_model")` or `simulator(model="my_model")` |
| **Model Config System**     | Assembles all configuration:                 | `_modelconfig/my_model.py`      |
|                             | - Model, boundary, drift, and simulator      |                                 |
|                             | - Parameter names, bounds, and choices       |                                 |
|                             | - References to boundary and drift registries|                                 |
| **Registries**              | Lookup and validation for:                   | `model_registry.py`,            |
|                             | - Models                                     | `boundary_registry.py`,         |
|                             | - Boundaries                                 | `drift_registry.py`             |
|                             | - Drifts                                     |                                 |
| **Simulator Implementation**| Model simulation code:                       |                                 |
|                             | - Python for prototyping/moderate workloads  | `src/simulators/my_simulator.py`|
|                             | - Cython for production/high-performance     |`src/cssm/my_model.pyx`          |

**Typical call flow:**

1. **User calls:**
   `Simulator("my_model")`
2. **Registry lookup:**
   Finds `my_model` config in the model registry.
3. **Config includes:**
   - List of parameters and their bounds
   - Boundary and drift functions (and any parameters)
   - Reference to simulator implementation (Python or Cython)
4. **Simulator runs:**
   Your simulator function (Python or Cython) is called with input parameters.
5. **Results returned:**
   Output contains simulated reaction times, choices, and optional metadata.

**Both Python and Cython simulators must implement:**
`simulate(theta, n_samples, ...) -> {'rts', 'choices'}`

This system ensures new models are discoverable, validated, and work seamlessly with command-line and programmatic interfaces.

### Key Components

1. **Model Configs** (`ssms/config/_modelconfig/`)
   - Define model metadata (parameters, bounds, choices)
   - Specify boundary and drift functions
   - Reference simulator implementation

2. **Registries** (`ssms/config/*_registry.py`)
   - Central lookup system for models, boundaries, and drifts
   - Enables dynamic model discovery
   - Validates configurations

3. **Simulator Implementations**
   - `Python` modules: Writing a simulator in Python first is helpful for basic testing
   - `Cython` modules (`src/cssm/`): Production and high-performance
   - (Note: you are not strictly bound to `Cython` you may use `numba` or other libraries as well)

4. **Configuration Builders**
   - `ModelConfigBuilder`: Constructs configs from names/overrides
   - Handles validation and defaults

### Example: How "ddm" Executes

1. **User calls**: `Simulator("ddm")`
2. **Registry lookup**: Finds DDM config in model registry
3. **Config specifies**:
   - `simulator=cssm.ddm_flexbound` (`Cython` function)
   - `boundary=bf.constant` (constant boundary function)
   - `params=["v", "a", "z", "t"]`
4. **Execution**: `cssm.ddm_flexbound(v, a, z, t, ...)` runs
5. **Returns**: `{"rts": [...], "choices": [...], "metadata": {...}}`

### File Locations Quick Reference

```bash
ssm-simulators/
├── ssms/
│   ├── config/
│   │   ├── _modelconfig/          # Model configurations
│   │   │   ├── __init__.py        # Exports all models
│   │   │   ├── ddm.py             # Example: DDM config
│   │   │   └── your_model.py      # Your new model config
│   │   ├── model_registry.py      # Model registration system
│   │   ├── boundary_registry.py   # Boundary functions
│   │   └── drift_registry.py      # Drift functions
│   └── basic_simulators/
│       └── boundary_functions.py  # Boundary implementations
├── src/
│   └── cssm/
│       ├── __init__.py            # Cython exports
│       ├── ddm_models.pyx         # Example: DDM Cython
│       └── your_model.pyx         # Your Cython implementation
└── tests/
    └── test_your_model.py         # Your tests
```

---

## 3. Level 1: Contributing Boundary/Drift Variants

**Goal**: Add a new model variant by composing existing simulators with custom boundary or drift functions.

**When to Use**: You want to test e.g. a new DDM (or other existing model) with a novel boundary shape or drift function, but the core simulation logic remains unchanged.

**Example**: DDM with collapsing boundaries (boundaries that move toward each other over time).

### Step 1: Create Your Boundary Function

Edit `ssms/basic_simulators/boundary_functions.py`:

```python
import numpy as np

def collapsing_linear(t: float, alpha: float = 0.1) -> float:
    """Linearly collapsing boundaries.

    Boundaries start at 1.0 and collapse linearly toward 0 with rate alpha.

    Parameters
    ----------
    t : float
        Time since trial start
    alpha : float
        Collapse rate (boundaries/second)

    Returns
    -------
    float
        Boundary value at time t

    Examples
    --------
    >>> collapsing_linear(0, alpha=0.1)  # t=0
    1.0
    >>> collapsing_linear(5, alpha=0.1)  # t=5
    0.5
    """
    return np.maximum(1.0 - alpha * t, 0.1)  # Prevent collapse to 0
```

### Step 2: Register Your Boundary

In the same file or in `ssms/config/boundary_registry.py`:

```python
from ssms.config import register_boundary

register_boundary(
    name="collapsing_linear",
    function=collapsing_linear,
    params=["alpha"]  # Parameters beyond time
)
```

### Step 3: Create Your Model Config

Create `ssms/config/_modelconfig/collapsing_ddm.py`:

```python
"""DDM with linearly collapsing boundaries."""

import cssm
from ssms.basic_simulators import boundary_functions as bf

def get_collapsing_ddm_config():
    """Get configuration for DDM with collapsing boundaries."""
    return {
        "name": "collapsing_ddm",
        "params": ["v", "a", "z", "t", "alpha"],  # Added alpha
        "param_bounds": [
            [-3.0, 0.3, 0.1, 0.0, 0.0],  # Lower bounds
            [3.0, 2.5, 0.9, 2.0, 0.5],   # Upper bounds (alpha ≤ 0.5)
        ],
        "boundary_name": "collapsing_linear",
        "boundary": bf.collapsing_linear,
        "boundary_params": ["alpha"],  # Which params go to boundary
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.1],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound,  # Reuse existing simulator!
    }
```

**Key Insight**: We're reusing `cssm.ddm_flexbound`—no new simulation code needed!

### Step 4: Register Your Model

Edit `ssms/config/_modelconfig/__init__.py`:

```python
# add this import
from .collapsing_ddm import get_collapsing_ddm_config

# ... existing imports ...

def get_model_config():
    """Collect all model configurations."""
    configs = {
        # ... existing models ...
        # add this line
        "collapsing_ddm": get_collapsing_ddm_config(),
    }
    return {name: _normalize_param_bounds(cfg) for name, cfg in configs.items()}
```

### Step 5: Test Your Model

In general we strive to have all core functionality covered by tests.
Below is an example for how you would test this new boundary function.

Create `tests/test_collapsing_ddm.py`:

```python
"""Tests for collapsing DDM model."""

import numpy as np
import pytest
from ssms import Simulator

def test_collapsing_ddm_runs():
    """Test that model runs without errors."""
    sim = Simulator("collapsing_ddm")
    result = sim.simulate(
        theta={"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "alpha": 0.1},
        n_samples=100,
        random_state=42
    )

    assert result["rts"].shape == (100,)
    assert np.all(result["choices"] == np.array([result["choices"]]).flatten())
    assert np.all(result["rts"] > 0)

def test_collapsing_effect():
    """Test that collapsing boundaries affect RT distribution."""
    sim = Simulator("collapsing_ddm")

    # No collapse
    result_no_collapse = sim.simulate(
        theta={"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "alpha": 0.0},
        n_samples=1000,
        random_state=42
    )

    # Strong collapse
    result_collapse = sim.simulate(
        theta={"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3, "alpha": 0.3},
        n_samples=1000,
        random_state=42
    )

    # Collapsing boundaries should generally reduce RTs
    mean_rt_no_collapse = np.mean(result_no_collapse["rts"])
    mean_rt_collapse = np.mean(result_collapse["rts"])

    assert mean_rt_collapse < mean_rt_no_collapse, (
        "Collapsing boundaries should reduce mean RT"
    )
```

### Step 6: Verify Locally

```bash
# Run your tests
pytest tests/test_collapsing_ddm.py -v

# Try it interactively
python -c "
from ssms import Simulator
sim = Simulator('collapsing_ddm')
print(sim.simulate({'v': 1, 'a': 1.5, 'z': 0.5, 't': 0.3, 'alpha': 0.1}, n_samples=10))
"
```

### Correctness Checklist for Level 1

- [ ] Boundary function is mathematically correct
- [ ] Parameters have valid bounds
- [ ] Tested against theoretical expectations
- [ ] Edge cases handled (e.g., boundary doesn't go negative)
- [ ] Documentation includes formula and references

---

## 4. Level 2: Contributing Python Simulators

**Goal**: Implement a completely new model class in pure Python.

**When to Use**: Your model doesn't fit existing frameworks, or you're prototyping a novel architecture. Suitable for moderate-scale use (<10k samples).

**Example**: Shifted Wald model (a simple non-DDM accumulator).

### Step 1: Implement Your Simulator

Create `ssms/basic_simulators/shifted_wald.py`:

```python
"""Shifted Wald (Inverse Gaussian) model implementation.

The Wald distribution models the first-passage time of a Brownian motion
with positive drift to a fixed boundary. Adding a shift parameter accounts
for non-decision time.

References
----------
Anders, R., Alario, F., & Van Maanen, L. (2016). The shifted Wald distribution
for response time data analysis. Psychological Methods, 21(3), 309.
"""

import numpy as np

def shifted_wald_simulator(
    v,  # drift rate
    a,  # boundary
    t,  # non-decision time (shift)
    max_t=20.0,
    n_samples=1000,
    n_trials=1,
    random_state=None,
    return_option="full",
    **kwargs
):
    """
    Simulate RTs from the Shifted Wald (Inverse Gaussian) distribution.

    This is a single-choice model: all responses are the same choice (coded as 1).
    The Wald distribution has a closed-form PDF, so we can sample directly without
    trajectory simulation.

    Parameters
    ----------
    v : np.ndarray, shape (n_trials,)
        Drift rate (evidence accumulation rate)
    a : np.ndarray, shape (n_trials,)
        Boundary (decision threshold)
    t : np.ndarray, shape (n_trials,)
        Non-decision time (shift parameter)
    max_t : float
        Maximum RT (for practical purposes)
    n_samples : int
        Number of samples per trial
    n_trials : int
        Number of trials (parameter sets)
    random_state : int, optional
        Random seed for reproducibility
    return_option : str
        'full' or 'minimal' metadata

    Returns
    -------
    dict
        Dictionary with keys:
        - 'rts': Reaction times, shape (n_trials * n_samples,)
        - 'choices': All 1 (single-choice model), shape (n_trials * n_samples,)
        - 'metadata': Model information

    Notes
    -----
    The Wald (Inverse Gaussian) PDF is:
        f(t | v, a) = (a / sqrt(2πt³)) * exp(-(a - vt)² / (2t))

    We add non-decision time: RT = Wald(v, a) + t
    """
    # Set random seed
    rng = np.random.default_rng(random_state)

    # Convert to arrays
    v = np.atleast_1d(v)
    a = np.atleast_1d(a)
    t = np.atleast_1d(t)

    # Validate inputs
    if np.any(v <= 0):
        raise ValueError("Drift rate v must be positive")
    if np.any(a <= 0):
        raise ValueError("Boundary a must be positive")
    if np.any(t < 0):
        raise ValueError("Non-decision time t must be non-negative")

    # Allocate output arrays
    total_samples = n_trials * n_samples
    rts = np.zeros(total_samples, dtype=np.float32)
    choices = np.ones(total_samples, dtype=np.int8)  # All choice 1

    # Sample for each trial
    for trial_idx in range(n_trials):
        start_idx = trial_idx * n_samples
        end_idx = (trial_idx + 1) * n_samples

        # Extract parameters for this trial
        v_trial = v[trial_idx]
        a_trial = a[trial_idx]
        t_trial = t[trial_idx]

        # Sample from Inverse Gaussian using numpy
        # mu = a/v, lambda = a²
        mu = a_trial / v_trial
        lam = a_trial ** 2

        # Inverse Gaussian sampling
        # Using Michael, Schucany & Haas (1976) algorithm
        nu = rng.standard_normal(n_samples)
        y = nu ** 2
        x = mu + (mu ** 2 * y) / (2 * lam) - (mu / (2 * lam)) * np.sqrt(
            4 * mu * lam * y + mu ** 2 * y ** 2
        )

        # Apply rejection sampling correction
        z = rng.uniform(0, 1, n_samples)
        use_x = z <= mu / (mu + x)

        wald_samples = np.where(use_x, x, mu ** 2 / x)

        # Add non-decision time
        rts[start_idx:end_idx] = wald_samples + t_trial

        # Clip to max_t
        rts[start_idx:end_idx] = np.minimum(rts[start_idx:end_idx], max_t)

    # Build metadata
    if return_option == "full":
        metadata = {
            "model": "shifted_wald",
            "params": ["v", "a", "t"],
            "param_values": {"v": v.tolist(), "a": a.tolist(), "t": t.tolist()},
            "n_samples": n_samples,
            "n_trials": n_trials,
            "max_t": max_t,
            "possible_choices": [1],
        }
    else:
        metadata = {
            "model": "shifted_wald",
            "max_t": max_t,
            "possible_choices": [1],
        }

    return {
        "rts": rts,
        "choices": choices,
        "metadata": metadata,
    }
```

### Step 2: Create Model Config

Create `ssms/config/_modelconfig/shifted_wald.py`:

```python
"""Shifted Wald (Inverse Gaussian) model configuration."""

from ssms.basic_simulators.shifted_wald import shifted_wald_simulator

def get_shifted_wald_config():
    """Get configuration for Shifted Wald model."""
    return {
        "name": "shifted_wald",
        "params": ["v", "a", "t"],
        "param_bounds": [
            [0.1, 0.3, 0.0],    # Lower bounds (v > 0, a > 0, t >= 0)
            [5.0, 3.0, 1.0],    # Upper bounds
        ],
        "n_params": 3,
        "default_params": [1.0, 1.0, 0.3],
        "nchoices": 1,  # Single-choice model
        "choices": [1],
        "n_particles": 1,
        "simulator": shifted_wald_simulator,  # Python function
    }
```

### Step 3: Register and Export

Edit `ssms/config/_modelconfig/__init__.py`:

```python
from .shifted_wald import get_shifted_wald_config

def get_model_config():
    configs = {
        # ... existing models ...
        "shifted_wald": get_shifted_wald_config(),
    }
    return {name: _normalize_param_bounds(cfg) for name, cfg in configs.items()}
```

### Step 4: Write Comprehensive Tests

Create `tests/test_shifted_wald.py`:

```python
"""Tests for Shifted Wald model."""

import numpy as np
import pytest
from scipy import stats
from ssms import Simulator

@pytest.fixture
def wald_sim():
    return Simulator("shifted_wald")

def test_shifted_wald_runs(wald_sim):
    """Test basic execution."""
    result = wald_sim.simulate(
        theta={"v": 1.0, "a": 1.0, "t": 0.3},
        n_samples=100,
        random_state=42
    )
    assert result["rts"].shape == (100,)
    assert np.all(result["choices"] == 1)  # Single choice

def test_shifted_wald_positive_rts(wald_sim):
    """All RTs must be positive."""
    result = wald_sim.simulate(
        theta={"v": 1.0, "a": 1.0, "t": 0.3},
        n_samples=1000,
        random_state=42
    )
    assert np.all(result["rts"] > 0)

def test_shifted_wald_invalid_params(wald_sim):
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="Drift rate v must be positive"):
        wald_sim.simulate(theta={"v": -1.0, "a": 1.0, "t": 0.3}, n_samples=10)

    with pytest.raises(ValueError, match="Boundary a must be positive"):
        wald_sim.simulate(theta={"v": 1.0, "a": -1.0, "t": 0.3}, n_samples=10)

    with pytest.raises(ValueError, match="Non-decision time t must be non-negative"):
        wald_sim.simulate(theta={"v": 1.0, "a": 1.0, "t": -0.1}, n_samples=10)

# ... possibly more tests ...
```

### Correctness Validation for Level 2

**Critical**: Correctness is paramount. Before submitting:

1. **Verify mathematical correctness**:
   - Compare with published equations
   - Test against analytical solutions if available
   - Validate parameter effects match theory

2. **Statistical validation**:
   - Check moments (mean, variance) against theory
   - Compare distributions with known results
   - Test edge cases and extreme parameters

3. **Reproducibility**:
   - Use `random_state` parameter for deterministic testing
   - Document any stochastic components

4. **Documentation**:
   - Cite papers and equations where appropriate
   - Explain any approximations or numerical methods
   - Note limitations

### When to Move to Level 3 (Cython)

We strongly recommend to not Cython-ize until correctness is established.
Also, if you model actually doesn't have any massive for loops etc. and samples
fast for many trials (e.g. `<<1s` for `100k` samples), you might simply not need further
optimization as motivated the original use of `Cython` in many of the pre-existing models.

---

## 5. Level 3: Contributing Cython Simulators

**Goal**: Implement high-performance simulator in Cython for production use.

**When to Use**: Model is validated, will be used at scale (>10k samples routinely), and Python performance is insufficient.

**Example**: `Cython` version of the Shifted Wald model from Level 2.

### Step 1: Study Existing Cython Simulators

Before writing, study similar models:

```bash
# Look at simple models first
cat src/cssm/ddm_models.pyx  # Single-particle DDM
cat src/cssm/ornstein_models.pyx  # OU process
```

**Key patterns to notice**:
- Type declarations: `np.ndarray[float, ndim=1] v`
- Memory views: `cdef float[:] rts_view`
- Utility functions: `set_seed()`, `draw_gaussian()`, `compute_boundary()`
- Return format: Same as Python versions

Try to reuse existing utilities as much as possible.

### Step 2: Create Cython Implementation

Create `src/cssm/shifted_wald_models.pyx`:

```cython
# Global settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Shifted Wald (Inverse Gaussian) Model - Cython Implementation

High-performance implementation for production use.
"""

import cython
from libc.math cimport sqrt, exp, log

import numpy as np
cimport numpy as np

# Import utility functions
from cssm._utils import (
    set_seed,
    random_uniform,
    draw_gaussian,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)

DTYPE = np.float32

def shifted_wald(
    np.ndarray[float, ndim=1] v,  # drift rate
    np.ndarray[float, ndim=1] a,  # boundary
    np.ndarray[float, ndim=1] t,  # non-decision time
    float max_t = 20.0,
    int n_samples = 20000,
    int n_trials = 1,
    random_state = None,
    return_option = 'full',
    **kwargs
):
    """
    Cython implementation of Shifted Wald model.

    Parameters are same as Python version (see shifted_wald.py for details).
    This version is optimized for speed using Cython type declarations.
    """
    # Type declarations for speed
    cdef int trial_idx, sample_idx, total_samples
    cdef float v_trial, a_trial, t_trial
    cdef float mu, lam, nu, y, x, z
    cdef float wald_sample

    # Set random seed
    set_seed(random_state)

    # Validate inputs
    if np.any(v <= 0):
        raise ValueError("Drift rate v must be positive")
    if np.any(a <= 0):
        raise ValueError("Boundary a must be positive")
    if np.any(t < 0):
        raise ValueError("Non-decision time t must be non-negative")

    # Allocate arrays
    total_samples = n_trials * n_samples
    cdef np.ndarray[float, ndim=1] rts = np.zeros(total_samples, dtype=DTYPE)
    cdef np.ndarray[np.int8_t, ndim=1] choices = np.ones(total_samples, dtype=np.int8)

    # Get memory views for fast access
    cdef float[:] rts_view = rts
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] t_view = t

    # Sample for each trial
    for trial_idx in range(n_trials):
        v_trial = v_view[trial_idx]
        a_trial = a_view[trial_idx]
        t_trial = t_view[trial_idx]

        # Inverse Gaussian parameters
        mu = a_trial / v_trial
        lam = a_trial * a_trial

        # Sample using Michael, Schucany & Haas (1976)
        for sample_idx in range(n_samples):
            # Draw standard normal
            nu = draw_gaussian()
            y = nu * nu

            # Compute candidate x
            x = mu + (mu * mu * y) / (2.0 * lam)
            x = x - (mu / (2.0 * lam)) * sqrt(4.0 * mu * lam * y + mu * mu * y * y)

            # Rejection sampling correction
            z = random_uniform()
            if z <= mu / (mu + x):
                wald_sample = x
            else:
                wald_sample = mu * mu / x

            # Add non-decision time and store
            rts_view[trial_idx * n_samples + sample_idx] = wald_sample + t_trial

            # Clip to max_t
            if rts_view[trial_idx * n_samples + sample_idx] > max_t:
                rts_view[trial_idx * n_samples + sample_idx] = max_t

    # Build metadata
    cdef dict param_dict = {
        "v": v.tolist(),
        "a": a.tolist(),
        "t": t.tolist(),
    }

    if return_option == 'full':
        metadata = build_full_metadata(
            model="shifted_wald",
            param_names=["v", "a", "t"],
            param_dict=param_dict,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            possible_choices=[1],
        )
    else:
        metadata = build_minimal_metadata(
            model="shifted_wald",
            max_t=max_t,
            possible_choices=[1],
        )

    return build_return_dict(rts, choices, metadata)
```

### Step 3: Export from Cython Module

Edit `src/cssm/__init__.py`:

```python
# Import from new module
from .shifted_wald_models import shifted_wald

__all__ = [
    # ... existing exports ...
    "shifted_wald",
]
```

### Step 4: Update Model Config to Use Cython

Edit `ssms/config/_modelconfig/shifted_wald.py`:

```python
"""Shifted Wald model configuration."""

import cssm  # Cython module

def get_shifted_wald_config():
    """Get configuration for Shifted Wald model."""
    return {
        "name": "shifted_wald",
        "params": ["v", "a", "t"],
        "param_bounds": [[0.1, 0.3, 0.0], [5.0, 3.0, 1.0]],
        "n_params": 3,
        "default_params": [1.0, 1.0, 0.3],
        "nchoices": 1,
        "choices": [1],
        "n_particles": 1,
        "simulator": cssm.shifted_wald,  # Now uses Cython!
    }
```

### Step 5: Build and Test

```bash
# Build Cython extension
python setup.py build_ext --inplace

# Run tests (should be identical to Python version)
pytest tests/test_shifted_wald.py -v

# Benchmark performance
python -c "
import time
import numpy as np
from ssms import Simulator

sim = Simulator('shifted_wald')
theta = {'v': 1.0, 'a': 1.0, 't': 0.3}

start = time.time()
result = sim.simulate(theta, n_samples=100000, random_state=42)
elapsed = time.time() - start

print(f'Simulated 100k samples in {elapsed:.2f}s')
print(f'Throughput: {100000/elapsed:.0f} samples/sec')
"
```

### Step 6: Validate Correctness Against Python

Before you commit to your Cython vesion, try to validate it
against the Python version of the new model.

Here is an example for how a corresponding test may look like.

```python
def test_cython_matches_python():
    """Verify Cython implementation matches Python version."""
    from ssms.basic_simulators.shifted_wald import shifted_wald_simulator
    import cssm

    theta_arrays = {
        "v": np.array([1.0]),
        "a": np.array([1.0]),
        "t": np.array([0.3]),
    }

    # Python version
    result_python = shifted_wald_simulator(
        **theta_arrays,
        n_samples=1000,
        random_state=42
    )

    # Cython version
    result_cython = cssm.shifted_wald(
        **theta_arrays,
        n_samples=1000,
        random_state=42
    )

    # Should be identical with same seed
    np.testing.assert_array_equal(result_python["rts"], result_cython["rts"])
    np.testing.assert_array_equal(result_python["choices"], result_cython["choices"])
```

## 6. Testing Your Contribution

Regardless of level, we strongly suggest a comprehensive test-suite for any new model.
The review process will reflect this philosophy concerning requests for extra tests where
appropriate.

### Test File Structure

Create `tests/test_your_model.py`:

```python
"""Tests for your_model."""

import numpy as np
import pytest
from ssms import Simulator

@pytest.fixture
def model_sim():
    """Fixture providing simulator instance."""
    return Simulator("your_model")

# 1. Smoke Tests

def test_model_runs(model_sim):
    """Basic execution without errors."""
    result = model_sim.simulate(
        theta={"param1": 1.0, "param2": 2.0},
        n_samples=100,
        random_state=42
    )
    assert result["rts"].shape == (100,)
    assert result["choices"].shape == (100,)
    assert "metadata" in result

# 2. Correctness Tests (MOST IMPORTANT)

def test_parameter_bounds(model_sim):
    """Parameters respect configured bounds."""
    # Test that invalid parameters are rejected or handled
    # ...


# ... more tests below ...
```

### Running Tests

```bash
# Run all tests for your model
pytest tests/test_your_model.py -v

# Run with coverage
pytest tests/test_your_model.py --cov=ssms --cov-report=html

# Run specific test
pytest tests/test_your_model.py::test_mathematical_correctness -v

# Run all tests (ensure you didn't break anything)
pytest tests/ -v
```

### Testing Checklist

Below a short checklist to guide your testing efforts:

- [ ] Smoke test: Model runs without crashing
- [ ] Correctness: Mathematical implementation validated
- [ ] Edge cases: Extreme parameters handled gracefully
- [ ] Reproducibility: Same seed gives identical results
- [ ] Integration: Works with ModelConfigBuilder

---

## 7. Documentation Requirements

Good documentation helps others understand and use your model. Include:

### 1. Model Config Docstring

In your `_modelconfig/your_model.py`:

```python
"""Your Model Name

Description
-----------
Brief description of what the model represents and its key features.

This model implements [brief explanation]. It is useful for [use cases].

Parameters
----------
param1 : float
    Description of param1. Valid range: [min, max]
    Theory: param1 controls [what it controls]
param2 : float
    Description of param2. Valid range: [min, max]
    Theory: param2 affects [what it affects]

Model Characteristics
--------------------
- Number of choices: X
- Boundary type: [constant/collapsing/etc]
- Drift type: [constant/time-varying/etc]
- Key assumptions: [list key assumptions]

References
----------
.. [1] Author, A. B. (YEAR). Paper title. Journal Name, VOL(ISSUE), pages.
       https://doi.org/...
.. [2] Author, C. D. (YEAR). Another paper. Journal, VOL, pages.

Examples
--------
>>> from ssms import Simulator
>>> sim = Simulator("your_model")
>>> result = sim.simulate(
...     theta={"param1": 1.0, "param2": 2.0},
...     n_samples=1000
... )
>>> import numpy as np
>>> print(f"Mean RT: {np.mean(result['rts']):.3f}")

See Also
--------
related_model1 : Related model for comparison
related_model2 : Alternative approach

Notes
-----
- Any important caveats or limitations
- Computational considerations
- When to use this vs related models
"""
```

### 2. Simulator Implementation Comments

In your simulator code:

```python
def your_simulator(...):
    """[Brief one-line description]

    [Longer description with equations if needed]

    Parameters
    ----------
    [Standard parameter documentation]

    Returns
    -------
    [Standard return documentation]

    Notes
    -----
    Algorithm:
    1. [Step 1 description]
    2. [Step 2 description]

    The key equation is:
        x(t+Δt) = x(t) + v*Δt + σ*√Δt*ε
    where ε ~ N(0,1)

    References
    ----------
    [Citations]
    """

    # Implementation with inline comments for tricky parts
    # Compute boundary value at current time
    boundary = compute_boundary(t, ...)  # See Eq. 3 in [1]

    # Apply rejection sampling correction
    # This ensures proper Inverse Gaussian sampling (Michael et al., 1976)
    if z <= mu / (mu + x):
        sample = x
    else:
        sample = mu ** 2 / x
```

### 3. Test Documentation

In your test file, document what each test validates:

```python
def test_mathematical_correctness(model_sim):
    """Verify output statistics match theoretical expectations.

    Theory predicts that for DDM with v=1, a=1, z=0.5:
        Mean RT ≈ a/(2v) = 0.5 seconds (ignoring non-decision time)

    We test this with large n to reduce sampling variance.

    Reference: Ratcliff (1978), Eq. 7
    """
    # ...
```

### 4. Optional: Tutorial Notebook

Consider creating a little notebook `notebooks/tutorial_your_model.ipynb`
that provides documentation for the model (or model class) you just added.

### Documentation Checklist

- [ ] Config file has comprehensive docstring with references
- [ ] Simulator function has detailed docstring
- [ ] Inline comments explain non-obvious code
- [ ] Tests document what they're validating
- [ ] Parameter meanings and ranges are clear
- [ ] References to papers/equations included
- [ ] Usage examples provided
- [ ] Optional: Tutorial notebook created

### Creating Your Pull Request

1. **Push your branch**:
   ```bash
   git checkout -b add-your-model
   git add ssms/config/_modelconfig/your_model.py
   git add tests/test_your_model.py
   # Add other relevant files
   git commit -m "Add Your Model implementation

   - Implement Your Model simulator (Level X)
   - Add comprehensive tests with theoretical validation
   - Document model equations and parameters
   - Benchmark shows X samples/sec (if Cython)
   "
   git push origin add-your-model
   ```

2. **Open PR on GitHub**:
   - Go to your fork on GitHub
   - Click "Pull Request"
   - Fill out the template (if applicable, see below)
   - Request review

### Pull Request Template

When you open a PR, you should see a template. We kindly ask you to check if the pattern
in there is relevant to your PR and if yes, please fill out the template as concretely as possible.

```markdown
## New Model: [Your Model Name]

### Description

[1-2 paragraph description of the model and its purpose]

### Type of Contribution

- [ ] Level 1: Boundary/Drift variant
- [ ] Level 2: Python simulator
- [ ] Level 3: Cython simulator

### Model Details

- **Parameters**: [list parameters with brief descriptions]
- **Number of choices**: [X]
- **Reference**: [paper citation(s)]
- **Use case**: [when to use this model vs alternatives]

### Correctness Validation

- [ ] Tested against theoretical predictions
- [ ] Compared with published results (if available)
- [ ] Edge cases tested and handled
- [ ] Statistical properties validated

### Testing

- Tests written: Yes
- All tests pass: Yes
- Test coverage: [X%] (if known)
- Performance benchmark (if Cython): [X samples/sec]

### Documentation

- Docstrings complete: Yes
- References cited: Yes
- Example usage provided: Yes
- Tutorial notebook: [Yes/No/Planned]

### Checklist

- [ ] Code follows existing style
- [ ] All tests pass locally
- [ ] Documentation is complete
- [ ] Existing tests still pass
- [ ] Model is registered and importable

### Additional Notes

[Any other information relevant to reviewers]
```

### After Your PR is Merged

Congratulations! Your model is now part of ssm-simulators!

- You'll be added to contributors list
- Your model will be available in next release
- Consider writing a tutorial or blog post
- Help answer questions about your model

### Getting Help

If you're stuck:

1. **Check existing models**: Look at similar implementations
   - Simple: `ddm.py`, `ornstein.py`
   - Complex: `race.py`, `lca.py`

2. **Read documentation**:
   - [Core Tutorials](../core_tutorials/)
   - [API Documentation](../api/)

3. **GitHub issues**: Search for similar problems
   - [Known issues](https://github.com/ssms/ssm-simulators/issues)

4. **Ask questions**: Open an issue
   - [Discussions](https://github.com/ssms/ssm-simulators/discussions)
   - Use tag: `question` or `help-wanted`
   - Provide: code snippet, error message, what you've tried

### Resources

- [Interactive Tutorials](../../notebooks/): Learn by doing
- [API Reference](../../api/): Complete API documentation
- [GitHub Issues](https://github.com/ssms/ssm-simulators/issues): Questions and bugs
- [Disussions](https://github.com/ssms/ssm-simulators/discussions)
