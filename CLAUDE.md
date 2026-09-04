# ssm-simulators — Project Context for Claude

## What is ssm-simulators?

Fast simulators and training data generators for Sequential Sampling Models (DDM, LBA, LCA, Race, Levy, etc.) used in cognitive science and neuroscience. The core simulators are implemented in C/Cython for performance. This is the foundational package in the HSSM ecosystem — HSSM, LANfactory, and LAN_pipeline_minimal all depend on it. For ecosystem-wide context, see the HSSMSpine repo.

## Project Structure

```
ssms/                          # Main package
  basic_simulators/            # Core API: simulator() function + Simulator class
  cli/                         # Typer CLI: `generate` command for batch data generation
  config/                      # Registry-based config system (models, boundaries, drifts)
    _modelconfig/              # Per-model config definitions (113 registered configs)
    generator_config/          # Data generation pipeline configs
  dataset_generators/          # Training data generation for LANfactory (protocols, pipelines, strategies)
  external_simulators/         # PyDDM integration (optional)
  support_utils/               # KDE class, utilities
  transforms/                  # Parameter sampling and simulation transforms
  hssm_support.py              # HSSM integration layer — critical interface contract
src/cssm/                      # Cython/C sources (12 .pyx: 9 simulators + _c_rng/_openmp_status/_utils)
ssm_configs/                   # Workspace subpackage: Pydantic config schemas, registries, plugin system
  src/ssm_configs/             # Package source (schema.py, registry.py, _plugins.py)
  tests/                       # Its own pytest suite — NOT collected by the root suite
hssm_example_model/            # Standalone workspace member: minimal example ssm_configs plugin
tests/                         # pytest suite with custom markers
docs/                          # MkDocs documentation source
examples/                      # Example scripts (custom transforms, nested configs)
benchmarks/                    # Performance benchmarks
```

## Build & Tooling

- **Build system:** setuptools + Cython (C extensions compiled from `src/cssm/*.pyx`)
- **Package manager:** uv (with `uv.lock`)
- **Python:** >=3.12, <3.15 (classifiers target 3.12, 3.13, 3.14)
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

# Run the ssm_configs subpackage's own tests (either form works)
uv run pytest ssm_configs/tests            # from the repo root
cd ssm_configs && uv run pytest            # from the subpackage

# Run with custom test categories
uv run pytest tests/ --run-notebooks      # execute notebook tests
uv run pytest tests/ --run-statistical    # statistical equivalence tests

# Lint & format
uv run ruff check . && uv run ruff format --check .

# Rebuild Cython extensions after C/pyx changes
uv run python setup.py build_ext --inplace

# Build docs strictly or preview them locally
./scripts/docs.sh build
./scripts/docs.sh serve

# CLI: generate training data from YAML config
uv run generate --config-path <path> --output <dir>
```

## Key Architecture Patterns

### Config Registry System

Models, boundary functions, and drift functions are registered in a registry system:
- `ssms.config.get_model_registry()` — all registered model simulators
- `ssms.config.get_boundary_registry()` — boundary function builders
- `ssms.config.get_drift_registry()` — drift function builders
- `ssms.config.model_config` — CopyOnAccessDict of all 113 model configs (safe to modify)
- `ModelConfigBuilder.from_model(name, **overrides)` — get/customize a model config

Every entry above is a **consumer** API — how to *read* a config. Authoring one
goes the other way round: a module under `ssms/config/_modelconfig/` names its
boundary and drift functions directly (`bf.angle`, `df.gamma_drift`) and is added
to the `configs` dict in `_modelconfig/__init__.py`. `model_registry.py` builds
the registry from that dict at import, so a new entry there is registered
everywhere automatically — reaching for `get_model_registry()` inside a config
module would bypass the source the registry is built from. See the
`add-ssm-model` skill for the full workflow.

### `ssm_configs` Subpackage

`ssm_configs/` is a separate distribution (`ssm-configs`) inside the uv
workspace, installed through the `dev` dependency group rather than as a runtime
dependency of `ssm-simulators`. It holds the Pydantic config schemas
(`schema.py`), the prefix-keyed model registries (`registry.py` — `hssm_registry`,
`rlssm_registry`), and the pluggy-based plugin system (`_plugins.py`).

**It owns its tests.** They live in `ssm_configs/tests/` and are configured by
`ssm_configs/pyproject.toml`, so the root `uv run pytest` (whose `testpaths` is
`["tests"]`) does **not** pick them up — run them with `uv run pytest
ssm_configs/tests`. Tests for the subpackage belong there, not in the root
`tests/` tree.

Third-party packages extend the registries as plugins: a distribution named
`<registry prefix>-<model name>` (e.g. `hssm-my-cool-model`) that implements the
`ssm_configs_config_path` hook adds `my_cool_model` to the HSSM registry's
`external_models` on install — no import required. Discovery is lazy (first read
of `external_models`) and warns-and-skips on any problem. See
`ssm_configs/README.md` for the author-facing contract.

`hssm_example_model/` at the repo root is a complete, working plugin — its own
distribution depending only on `ssm-configs` and `pluggy`, and the template to
copy when writing one. It is a workspace member installed by
`uv sync --all-groups`, so CI exercises the plugin path against a real installed
distribution.

### Cython Simulator Layer

Nine `.pyx` simulator modules in `src/cssm/` implement the actual simulators in C:
`ddm_models`, `addm_models`, `race_models`, `lba_models`, `levy_models`,
`ornstein_models`, `poisson_race_models`, `sequential_models`, `parallel_models` —
plus three helper modules (`_c_rng`, `_openmp_status`, `_utils`), for 12 `.pyx` files
total. These use GSL for random number generation and OpenMP for multi-threading.

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
- **113 registered model configs** across DDM, Angle, Weibull, Levy, Ornstein, LBA, LCA, Race, Racing Diffusion, Poisson Race, MIC2, Conflict, shrink-spotlight, tradeoff, and RLWM/softmax choice-only families (`_deadline` variants derived at runtime)

## Skills

- **add-ssm-model** — guided workflow for adding a new model: config creation,
  optional boundary/drift functions, registration, and validation

## CI Workflows

| Workflow | Purpose |
|----------|---------|
| `run_tests.yml` | Tests on Python 3.12/3.13/3.14 + separate multithreading job (installs GSL/OpenMP) |
| `build_wheels.yml` | Build wheels (cibuildwheel), upload to TestPyPI → PyPI on release publish |

## Compaction

When compacting, preserve: file list of modified files, the HSSM integration
contract details, system dependency requirements, and all test commands.
