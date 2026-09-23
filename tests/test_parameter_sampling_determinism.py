"""Multi-process regression tests for parameter-sampling determinism.

These are the tests that would have caught the two defects fixed by
`B_upstream_fix.patch`. Both defects are invisible to a single-process test,
which is exactly why they survived: the sampler passed its unit tests and a
64-index pipeline test inside ONE interpreter.

Defect 1 -- `AbstractParameterSampler._topological_sort` walked
`self._dependency_graph`, a dict whose insertion order was seeded by a `set` of
parameter names. String hashing is randomised per interpreter, so the
parameter -> RNG-draw assignment differed between processes: same seed, same
draws, different parameters receiving them.

Defect 2 -- `SimulationPipeline.generate_for_parameter_set` used the theta INDEX
directly as the parameter-RNG seed, and every caller passed
`range(0, n_parameter_sets)`. Independently launched runs (one SLURM array task
per output file, or a `--n-files N` loop) therefore drew the same thetas.

Run:
    pytest -q tests/test_parameter_sampling_determinism.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Worker programs, run in a fresh interpreter so PYTHONHASHSEED actually bites.
# PYTHONHASHSEED must be set before the process starts; setting it in-process
# does nothing, which is the whole reason these tests need subprocesses.
# --------------------------------------------------------------------------- #

_ORDER_WORKER = r"""
import json, sys
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

model = sys.argv[1]
mc = ModelConfigBuilder.from_model(model)
sampler = UniformParameterSampler(param_space=mc["param_bounds_dict"])
print(json.dumps({"order": list(sampler._sampling_order)}))
"""

# Calls the real pipeline entry point -- `generate_for_parameter_set` -- rather
# than re-deriving what it does. Re-implementing the sampling step in the test
# is how the original defect was missed.
_PIPELINE_WORKER = r"""
import json, sys
from ssms.config import ModelConfigBuilder
from ssms.config import get_default_generator_config
from ssms.dataset_generators.pipelines.pipeline_factory import (
    create_data_generation_pipeline,
)

model, offset, n_theta = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# Where the offset is written. "pipeline" is what the CLI produces from a YAML
# PIPELINE.PARAMETER_SAMPLER_INDEX_OFFSET; "root" is the programmatic fallback.
where = sys.argv[4] if len(sys.argv) > 4 else "root"

gc = get_default_generator_config("lan")
gc["model"] = model
gc["pipeline"]["n_parameter_sets"] = n_theta
gc["pipeline"]["n_cpus"] = 1
gc["pipeline"]["n_subruns"] = 1
gc["simulator"]["n_samples"] = 500          # keep it fast; we only need theta
gc["training"]["n_samples_per_param"] = 20
if where == "pipeline":
    gc["pipeline"]["parameter_sampler_index_offset"] = offset
else:
    gc["parameter_sampler_index_offset"] = offset

mc = ModelConfigBuilder.from_model(model)
pipeline = create_data_generation_pipeline(generator_config=gc, model_config=mc)

thetas = []
for index in range(n_theta):
    out = pipeline.generate_for_parameter_set(index, 1234 + index)
    theta = out["theta"]
    thetas.append([float(theta[p].reshape(-1)[0]) for p in mc["params"]])

print(json.dumps({"params": list(mc["params"]), "thetas": thetas}))
"""


def _run(worker: str, args: list[str], hashseed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    proc = subprocess.run(
        [sys.executable, "-c", worker, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"worker failed (PYTHONHASHSEED={hashseed}, args={args})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    # take the last JSON line; libraries may log to stdout
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


HASHSEEDS = ["0", "1", "2", "12345"]

# Models chosen to cover both graph shapes: no dependent bounds (ddm, ddm_sdv)
# and bounds that name another parameter (ddm_st has st bounded by t, full_ddm
# has both sz<-z and st<-t), because the dependency edges are what pinned part
# of the order and hid the rest.
ORDER_MODELS = ["ddm", "ddm_sdv", "ddm_st", "full_ddm"]


@pytest.mark.parametrize("model", ORDER_MODELS)
def test_sampling_order_stable_across_hashseeds(model):
    """Parameter <-> draw assignment must not depend on PYTHONHASHSEED.

    Fails on the pre-fix code: `_topological_sort` iterated a set-seeded dict,
    so the mutually independent parameters (v, a, sv, ...) permuted per process.
    """
    orders = {s: _run(_ORDER_WORKER, [model], s)["order"] for s in HASHSEEDS}
    distinct = {tuple(o) for o in orders.values()}
    assert len(distinct) == 1, (
        f"{model}: sampling order varies with PYTHONHASHSEED -> "
        f"{json.dumps(orders, indent=2)}"
    )


@pytest.mark.parametrize("model", ORDER_MODELS)
def test_sampling_order_respects_dependencies(model):
    """The determinism fix must not break topological validity."""
    from ssms.config import ModelConfigBuilder
    from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

    mc = ModelConfigBuilder.from_model(model)
    bounds = mc["param_bounds_dict"]
    order = UniformParameterSampler(param_space=bounds)._sampling_order
    position = {p: i for i, p in enumerate(order)}
    for param, bound in bounds.items():
        for value in bound:
            if isinstance(value, str):
                assert position[value] < position[param], (
                    f"{model}: '{param}' depends on '{value}' but is sampled first"
                )


@pytest.mark.slow
def test_pipeline_same_offset_reproduces_thetas():
    """Two processes, same offset, different hashseed -> identical theta sets.

    This is the reproducibility half. It fails on the pre-fix code for a reason
    that is easy to misread as success: the theta *values* drawn from the RNG are
    identical, but they land on different parameters, so the theta rows differ.
    """
    a = _run(_PIPELINE_WORKER, ["full_ddm", "0", "4"], "1")
    b = _run(_PIPELINE_WORKER, ["full_ddm", "0", "4"], "2")
    assert a["params"] == b["params"]
    assert a["thetas"] == b["thetas"], (
        "same offset produced different thetas across processes:\n"
        f"hashseed 1: {a['thetas']}\nhashseed 2: {b['thetas']}"
    )


@pytest.mark.slow
def test_pipeline_distinct_offsets_give_disjoint_thetas():
    """Two processes, different offsets -> disjoint theta sets.

    This is the half that matters for training data. It fails on the pre-fix
    code because `parameter_sampler_index_offset` is not read at all, so both
    processes seed the parameter RNG with indices 0..n-1 and every output file
    carries the same thetas.
    """
    n = 4
    a = _run(_PIPELINE_WORKER, ["full_ddm", "0", str(n)], "1")
    b = _run(_PIPELINE_WORKER, ["full_ddm", str(n), str(n)], "1")

    rows_a = {tuple(r) for r in a["thetas"]}
    rows_b = {tuple(r) for r in b["thetas"]}
    assert len(rows_a) == n and len(rows_b) == n, "thetas repeated within one block"
    assert not (rows_a & rows_b), (
        f"offset 0 and offset {n} share {len(rows_a & rows_b)} of {n} theta rows"
    )

    # Per-axis check: a coordinate permutation of one point set would still look
    # disjoint row-wise, so assert no shared value on any single axis either.
    for j, name in enumerate(a["params"]):
        col_a = {r[j] for r in a["thetas"]}
        col_b = {r[j] for r in b["thetas"]}
        assert not (col_a & col_b), f"axis '{name}' shares values across offsets"


@pytest.mark.slow
def test_repeated_generate_calls_do_not_reuse_thetas(tmp_path):
    """`generate_data_training()` twice in one process -> distinct theta sets.

    This is the documented CLI path: `ssms generate --n-files N` loops
    `generate_data_training(save=True)` N times in ONE process
    (ssms/cli/generate.py). Pre-fix, the N files carry N identical theta sets.
    """
    from ssms.config import ModelConfigBuilder
    from ssms.config import get_default_generator_config
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

    gc = get_default_generator_config("lan")
    gc["model"] = "full_ddm"
    gc["pipeline"]["n_parameter_sets"] = 4
    gc["pipeline"]["n_cpus"] = 1
    gc["pipeline"]["n_subruns"] = 1
    gc["simulator"]["n_samples"] = 500
    gc["training"]["n_samples_per_param"] = 20
    gc["output"]["folder"] = str(tmp_path)

    mc = ModelConfigBuilder.from_model("full_ddm")
    gen = TrainingDataGenerator(config=gc, model_config=mc)

    first = gen.generate_data_training(save=False)
    second = gen.generate_data_training(save=False)

    rows_1 = {tuple(map(float, r)) for r in first["theta"]}
    rows_2 = {tuple(map(float, r)) for r in second["theta"]}
    assert not (rows_1 & rows_2), (
        f"two calls in one process share {len(rows_1 & rows_2)} theta rows"
    )


def test_cli_yaml_offset_lands_in_the_pipeline_section(tmp_path):
    """The premise of the nested lookup: the CLI files the YAML key under 'pipeline'.

    `collect_data_generator_config` forwards the whole PIPELINE section and drops
    root-level YAML keys, so a lookup that only reads the config root turns
    PARAMETER_SAMPLER_INDEX_OFFSET into a no-op for every CLI user.
    """
    from ssms.cli.generate import collect_data_generator_config
    from ssms.config.config_utils import get_parameter_sampler_index_offset

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "MODEL: 'ddm'\n"
        "GENERATOR_APPROACH: 'lan'\n"
        "PARAMETER_SAMPLER_INDEX_OFFSET: 77\n"
        "PIPELINE:\n"
        "  N_PARAMETER_SETS: 10\n"
        "  N_SUBRUNS: 2\n"
        "  PARAMETER_SAMPLER_INDEX_OFFSET: 42\n"
        "SIMULATOR:\n"
        "  N_SAMPLES: 2000\n"
        "  DELTA_T: 0.001\n"
        "TRAINING:\n"
        "  N_SAMPLES_PER_PARAM: 200\n"
        "ESTIMATOR:\n"
        "  TYPE: 'kde'\n"
    )
    generator_config = collect_data_generator_config(
        str(yaml_path), base_path=str(tmp_path)
    )["data_config"]

    assert generator_config["pipeline"]["parameter_sampler_index_offset"] == 42
    assert "parameter_sampler_index_offset" not in generator_config, (
        "a root-level YAML key is dropped by the CLI; the root lookup cannot be the "
        "only one"
    )
    assert get_parameter_sampler_index_offset(generator_config) == 42
    # Programmatic callers that set the root key directly still work.
    assert (
        get_parameter_sampler_index_offset({"parameter_sampler_index_offset": 7}) == 7
    )
    assert get_parameter_sampler_index_offset({"pipeline": {}}) == 0


@pytest.mark.slow
def test_pipeline_reads_offset_from_nested_pipeline_section():
    """A nested offset must shift the block exactly as a root offset does."""
    n = 4
    unshifted = _run(_PIPELINE_WORKER, ["full_ddm", "0", str(n), "root"], "1")
    nested = _run(_PIPELINE_WORKER, ["full_ddm", str(n), str(n), "pipeline"], "1")
    root = _run(_PIPELINE_WORKER, ["full_ddm", str(n), str(n), "root"], "1")

    assert nested["thetas"] == root["thetas"], (
        "offset under generator_config['pipeline'] was ignored:\n"
        f"nested: {nested['thetas']}\nroot:   {root['thetas']}"
    )
    assert not (
        {tuple(r) for r in nested["thetas"]} & {tuple(r) for r in unshifted["thetas"]}
    ), "nested offset did not shift the theta block at all"


@pytest.mark.slow
def test_subrun_remainder_consumes_every_theta_index(tmp_path):
    """A non-divisible n_parameter_sets / n_subruns pair must still use every index.

    Floor division alone left the last `n_parameter_sets % n_subruns` indices
    ungenerated while the cursor advanced by the full n_parameter_sets, so those
    indices were skipped for the life of the process and each file came up short.
    """
    from ssms.config import ModelConfigBuilder, get_default_generator_config
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

    gc = get_default_generator_config("lan")
    gc["model"] = "ddm"
    gc["pipeline"]["n_parameter_sets"] = 5  # 5 // 2 == 2, remainder 1
    gc["pipeline"]["n_subruns"] = 2
    gc["pipeline"]["n_cpus"] = 1
    gc["simulator"]["n_samples"] = 500
    gc["training"]["n_samples_per_param"] = 20
    gc["output"]["folder"] = str(tmp_path)

    gen = TrainingDataGenerator(
        config=gc, model_config=ModelConfigBuilder.from_model("ddm")
    )

    consumed: list[int] = []
    inner = gen._generation_pipeline.generate_for_parameter_set

    def recording(parameter_sampling_seed, random_seed=None):
        """Record the theta index, then delegate to the real pipeline."""
        consumed.append(int(parameter_sampling_seed))
        return inner(parameter_sampling_seed, random_seed)

    gen._generation_pipeline.generate_for_parameter_set = recording

    gen.generate_data_training(save=False)
    assert consumed == [0, 1, 2, 3, 4], f"first call consumed {consumed}"

    consumed.clear()
    gen.generate_data_training(save=False)
    assert consumed == [5, 6, 7, 8, 9], f"second call consumed {consumed}"


class _StopAfterSampling(Exception):
    """Sentinel that ends PyDDM generation once theta has been sampled."""


class _ThetaCapture:
    """Estimator-builder stub: records the sampled theta, then aborts the run."""

    def __init__(self):
        """Start with no recorded theta."""
        self.theta = None

    def build(self, theta_dict, simulations=None):
        """Record theta and raise, so no Fokker-Planck solve is needed."""
        self.theta = {
            k: float(np.asarray(v).reshape(-1)[0]) for k, v in theta_dict.items()
        }
        raise _StopAfterSampling


def _pyddm_theta(offset: int, index: int, where: str = "pipeline") -> dict:
    """Sample one theta through PyDDMPipeline with everything downstream stubbed."""
    from ssms.config import ModelConfigBuilder, get_default_generator_config
    from ssms.dataset_generators.pipelines.pyddm_pipeline import PyDDMPipeline

    gc = get_default_generator_config("lan")
    gc["model"] = "full_ddm"
    if where == "pipeline":
        gc["pipeline"]["parameter_sampler_index_offset"] = offset
    else:
        gc["parameter_sampler_index_offset"] = offset

    capture = _ThetaCapture()
    pipeline = PyDDMPipeline(
        generator_config=gc,
        model_config=ModelConfigBuilder.from_model("full_ddm"),
        estimator_builder=capture,
        training_strategy=object(),  # never reached
    )
    with pytest.raises(_StopAfterSampling):
        pipeline.generate_for_parameter_set(index, 1234)
    return capture.theta


def test_pyddm_pipeline_applies_the_same_index_offset():
    """PyDDMPipeline must shift the theta index exactly as SimulationPipeline does.

    `TrainingDataGenerator` hands the identical index to whichever pipeline the
    estimator type selected, so the index must mean the same thing in both. Before
    the fix PyDDM seeded only the legacy global RNG, which `sample()` never reads,
    so its thetas were irreproducible and the offset could not apply.
    """
    assert _pyddm_theta(offset=0, index=3) == _pyddm_theta(offset=0, index=3), (
        "PyDDM theta sampling is not reproducible for a fixed index"
    )
    shifted = _pyddm_theta(offset=4, index=0)
    direct = _pyddm_theta(offset=0, index=4)
    assert shifted == direct, f"offset ignored by PyDDMPipeline: {shifted} != {direct}"
    assert shifted != _pyddm_theta(offset=0, index=0), "offset shifted nothing"
    # The root placement is the programmatic fallback and must agree.
    assert _pyddm_theta(offset=4, index=0, where="root") == direct
