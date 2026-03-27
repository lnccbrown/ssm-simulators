# ssm-simulators — Project Context for Claude

## What is ssm-simulators?

Fast simulators and training data generators for Sequential Sampling Models (DDM, LBA, LCA, Race, Levy, etc.) used in cognitive science and neuroscience. The core simulators are implemented in C/Cython for performance. This is the foundational package in the HSSM ecosystem — HSSM, LANfactory, and LAN_pipeline_minimal all depend on it. For ecosystem-wide context, see the HSSMSpine repo.

## Project Structure

```
ssms/                          # Main package
  basic_simulators/            # Core API: simulator() function + Simulator class
  cli/                         # Typer CLI: `generate` command for batch data generation
  config/                      # Registry-based config system (models, boundaries, drifts)
    _modelconfig/              # Per-model config definitions (22 models)
    generator_config/          # Data generation pipeline configs
  dataset_generators/          # Training data generation for LANfactory (protocols, pipelines, strategies)
  external_simulators/         # PyDDM integration (optional)
  support_utils/               # KDE class, utilities
  transforms/                  # Parameter sampling and simulation transforms
  hssm_support.py              # HSSM integration layer — critical interface contract
src/cssm/                      # Cython/C source files (9 modules: ddm, race, lba, levy, etc.)
tests/                         # pytest suite with custom markers
docs/                          # MkDocs documentation source
examples/                      # Example scripts (custom transforms, nested configs)
benchmarks/                    # Performance benchmarks
```

## Build & Tooling

- **Build system:** setuptools + Cython (C extensions compiled from `src/cssm/*.pyx`)
- **Package manager:** uv (with `uv.lock`)
- **Python:** >=3.10, <3.14 (classifiers target 3.11, 3.12, 3.13)
- **System dependencies (required for C extensions):**
  - C compiler (Xcode CLI tools on macOS, build-essential on Linux)
  - GSL (GNU Scientific Library) — `brew install gsl` / `apt install libgsl-dev`
  - OpenMP — `brew install libomp` / `apt install libomp-dev`
- **Linting:** ruff (via pre-commit)
- **Type checking:** mypy

## Common Commands

```bash
# Install with dev dependencies (compiles Cython extensions — takes a few minutes)
uv sync --extra dev

# Run tests (fast subset)
uv run pytest tests/ -x --timeout=60

# Run with custom test categories
uv run pytest tests/ --run-notebooks      # execute notebook tests
uv run pytest tests/ --run-statistical    # statistical equivalence tests

# Lint & format
uv run ruff check . && uv run ruff format --check .

# Rebuild Cython extensions after C/pyx changes
uv run python setup.py build_ext --inplace

# Build docs
uv run --extra docs mkdocs build
uv run --extra docs mkdocs serve

# CLI: generate training data from YAML config
uv run generate --config-path <path> --output <dir>
```

## Key Architecture Patterns

### Config Registry System

Models, boundary functions, and drift functions are registered in a registry system:
- `ssms.config.get_model_registry()` — all registered model simulators
- `ssms.config.get_boundary_registry()` — boundary function builders
- `ssms.config.get_drift_registry()` — drift function builders
- `ssms.config.model_config` — CopyOnAccessDict of all 22 model configs (safe to modify)
- `ModelConfigBuilder.from_model(name, **overrides)` — get/customize a model config

### Cython Simulator Layer

Nine `.pyx` modules in `src/cssm/` implement the actual simulators in C:
`ddm_models`, `race_models`, `lba_models`, `levy_models`, `ornstein_models`,
`poisson_race_models`, `sequential_models`, `parallel_models`, `_c_rng`.
These use GSL for random number generation and OpenMP for multi-threading.

### Core Public API

- **`Simulator` class** — the primary public interface. Supports model selection
  by name or custom simulator function, custom boundary/drift functions via callable
  or registry name. Method: `simulate(theta, n_samples, ...)`.
  **Always prefer `Simulator` over the bare `simulator()` function** in examples,
  tutorials, and new code.
- **`simulator(theta, model, n_samples, ...)`** — lower-level function used internally.
  Returns `{'rts': ndarray, 'choices': ndarray, 'metadata': dict}`.
  Exists for backward compatibility; new code should use the `Simulator` class.
- **`TrainingDataGenerator`** — generates training data for LANfactory/LAN networks.
  Config-driven pipeline: parameter sampling → simulation → likelihood estimation

### HSSM Integration Contract (`hssm_support.py`)

This module bridges ssm-simulators and HSSM. The contract is critical:
- Simulator functions must expose `.model_name`, `.choices`, `.obs_dim` attributes
- `hssm_sim_wrapper()` adapts simulators to HSSM's expected interface
- `rng_fn()` provides the RNG function wrapper HSSM uses for sampling
- Output shape: `(..., obs_dim)` with last columns as (RT, choice) pairs

Changing this interface requires coordinating with HSSM.

## Key Conventions

- **`OMISSION_SENTINEL = -999.0`** — marks deadline timeouts in simulation output
- **Multiprocessing:** uses `spawn` method by default (required for OpenMP safety)
- **Deadline models:** any model supports a `_deadline` suffix (e.g., `ddm_deadline`)
- **Max threads:** 256 (compile-time limit for per-thread RNG state arrays)
- **22 model variants** across DDM, Angle, LBA, LCA, Race, Poisson Race, Racing Diffusion, Levy, Ornstein families

## CI Workflows

| Workflow | Purpose |
|----------|---------|
| `run_tests.yml` | Tests on Python 3.11/3.12/3.13 + separate multithreading job (installs GSL/OpenMP) |
| `build_wheels.yml` | Build wheels (cibuildwheel), upload to TestPyPI → PyPI on release publish |

## Compaction

When compacting, preserve: file list of modified files, the HSSM integration
contract details, system dependency requirements, and all test commands.
