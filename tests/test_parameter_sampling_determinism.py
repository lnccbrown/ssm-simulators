"""Regression tests for parameter-sampling determinism and theta-index blocks.

These are the tests that would have caught two defects introduced together in
57389be (PR #247). Both are invisible to a single-process test, which is exactly
why they survived: the sampler passed its unit tests and a 64-index pipeline
test inside ONE interpreter.

Defect 1 -- `AbstractParameterSampler._topological_sort` walked
`self._dependency_graph`, a dict whose insertion order was seeded by a `set` of
parameter names. String hashing is randomised per interpreter, so the
parameter -> RNG-draw assignment differed between processes: same seed, same
draws, different parameters receiving them.

Defect 2 -- `SimulationPipeline.generate_for_parameter_set` used the theta INDEX
directly as the parameter-RNG seed, and every caller passed
`range(0, n_parameter_sets)`. Independently launched runs (one SLURM array task
per output file, or a `--n-files N` loop) therefore drew the same thetas.

The fix for defect 2 lives in `TrainingDataGenerator`, which is the one place
that applies an offset: every call consumes the next block of `n_parameter_sets`
indices above a base that is either the config's explicit
`parameter_sampler_index_offset` or, without one, an entropy draw made once per
generator. The first index of each file is recorded in the file's
`generator_config` under that same key. The tests below cover both halves:
runs that must differ do, and runs that must agree do.

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

from ssms.config import ModelConfigBuilder, get_default_generator_config
from ssms.config.config_utils import get_parameter_sampler_index_offset
from ssms.dataset_generators.estimator_builders import KDEEstimatorBuilder
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler
from ssms.dataset_generators.pipelines import PyDDMPipeline, SimulationPipeline
from ssms.dataset_generators.protocols import DataGenerationPipelineProtocol
from ssms.dataset_generators.strategies import MixtureTrainingStrategy

# --------------------------------------------------------------------------- #
# Worker programs, run in a fresh interpreter so PYTHONHASHSEED actually bites.
# PYTHONHASHSEED must be set before the process starts; setting it in-process
# does nothing, which is the whole reason these tests need subprocesses. Each
# worker does everything one interpreter can be asked at once: importing ssms
# costs seconds, so one process per hash seed, not one per (seed, model).
# --------------------------------------------------------------------------- #

# argv: JSON list of registered model names, JSON dict of synthetic spaces.
# Prints {space name: sampling order} for all of them.
_ORDER_WORKER = r"""
import json, sys
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

spaces = {
    model: ModelConfigBuilder.from_model(model)["param_bounds_dict"]
    for model in json.loads(sys.argv[1])
}
spaces.update(
    {
        name: {param: tuple(bounds) for param, bounds in space.items()}
        for name, space in json.loads(sys.argv[2]).items()
    }
)
print(
    json.dumps(
        {
            name: list(UniformParameterSampler(param_space=space)._sampling_order)
            for name, space in spaces.items()
        }
    )
)
"""

# argv: a JSON list of generator configs, e.g. one as recorded in an output
# file. Builds a generator from each -- the real entry point, not a
# re-derivation of what it does -- and prints the thetas each one draws.
_REGENERATE_WORKER = r"""
import json, sys
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

print(
    json.dumps(
        {
            "thetas": [
                TrainingDataGenerator(config=gc).generate_data_training()["theta"].tolist()
                for gc in json.loads(sys.argv[1])
            ]
        }
    )
)
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

# Registered models of both graph shapes: numeric bounds only (ddm, ddm_sdv,
# full_ddm) and the one registered model whose bounds name another parameter
# (full_ddm_rv: st is bounded above by t). No other registered model has a
# dependent bound, which is why a synthetic space is needed below.
ORDER_MODELS = ["ddm", "ddm_sdv", "full_ddm", "full_ddm_rv"]

# A space the registry does not contain: one parameter with two dependents (so
# the iteration order of an adjacency set matters), declared after both of
# them, and a parameter that depends on two others.
SYNTHETIC_SPACES = {
    "multi_dependent": {
        "b": ("a", 2.0),
        "c": ("a", 3.0),
        "a": (0.0, 1.0),
        "d": ("b", "c"),
    }
}

# How many (param, dependency) edges each space declares. Asserted, so the
# dependency-order test cannot silently become vacuous again: its first version
# listed ddm_st and full_ddm as dependent models, and neither is.
DEPENDENCY_EDGES = {
    "ddm": 0,
    "ddm_sdv": 0,
    "full_ddm": 0,
    "full_ddm_rv": 1,
    "multi_dependent": 4,
}


def _spaces() -> dict[str, dict]:
    spaces = {
        model: ModelConfigBuilder.from_model(model)["param_bounds_dict"]
        for model in ORDER_MODELS
    }
    spaces.update(SYNTHETIC_SPACES)
    return spaces


def _dependency_edges(space: dict) -> list[tuple[str, str]]:
    """(param, dependency) pairs declared by string bounds."""
    return [
        (param, value)
        for param, bounds in space.items()
        for value in bounds
        if isinstance(value, str)
    ]


def _fast_config(
    model: str,
    n_theta: int,
    folder,
    offset: int | None = None,
    where: str = "pipeline",
) -> dict:
    """A small generator config; only theta matters, so simulations are tiny.

    `where` is where the offset is written: "pipeline" is what the CLI produces
    from a YAML PIPELINE.PARAMETER_SAMPLER_INDEX_OFFSET, "root" the programmatic
    fallback.
    """
    gc = get_default_generator_config("lan")
    gc["model"] = model
    gc["pipeline"]["n_parameter_sets"] = n_theta
    gc["pipeline"]["n_cpus"] = 1
    gc["pipeline"]["n_subruns"] = 1
    gc["simulator"]["n_samples"] = 500
    gc["training"]["n_samples_per_param"] = 20
    gc["output"]["folder"] = str(folder)
    if offset is not None:
        target = gc["pipeline"] if where == "pipeline" else gc
        target["parameter_sampler_index_offset"] = offset
    return gc


def _recorded_offset(output: dict) -> int:
    return output["generator_config"]["pipeline"]["parameter_sampler_index_offset"]


def _stub_result() -> dict:
    """The smallest record `generate_data_training` accepts as a success."""
    return {
        "data": {"theta": np.zeros((1, 1), dtype=np.float32)},
        "theta": {},
        "success": True,
    }


def _consumed_indices(
    gen: TrainingDataGenerator, n_calls: int = 1
) -> tuple[list[list[int]], list[dict]]:
    """Run `generate_data_training` n_calls times with the pipeline stubbed out.

    The theta indices the generator hands to its pipeline are decided before any
    simulation runs, so for tests about those indices the pipeline is replaced
    by a recorder that returns a minimal success record.
    """
    calls: list[list[int]] = []

    def recording(parameter_sampling_seed, random_seed=None):
        calls[-1].append(int(parameter_sampling_seed))
        return _stub_result()

    gen._generation_pipeline.generate_for_parameter_set = recording

    outputs = []
    for _ in range(n_calls):
        calls.append([])
        outputs.append(gen.generate_data_training(save=False))
    return calls, outputs


def _assert_disjoint(a: np.ndarray, b: np.ndarray, params: list[str]) -> None:
    """No shared theta row, and no shared value on any single axis.

    The per-axis check matters: a coordinate permutation of one point set --
    what defect 1 produced across processes -- still looks disjoint row-wise.
    """
    rows_a = {tuple(map(float, r)) for r in a}
    rows_b = {tuple(map(float, r)) for r in b}
    assert len(rows_a) == len(a) and len(rows_b) == len(b), (
        "thetas repeated within one block"
    )
    assert not (rows_a & rows_b), f"blocks share {len(rows_a & rows_b)} theta rows"
    for j, name in enumerate(params):
        assert not ({float(r[j]) for r in a} & {float(r[j]) for r in b}), (
            f"axis '{name}' shares values across blocks"
        )


# --------------------------------------------------------------------------- #
# Defect 1: the sampling order.
# --------------------------------------------------------------------------- #


def test_sampling_order_stable_across_hashseeds():
    """Parameter <-> draw assignment must not depend on PYTHONHASHSEED.

    Fails on the pre-fix code: `_topological_sort` iterated a set-seeded dict,
    so the mutually independent parameters (v, a, sv, ...) permuted per process,
    and so did the two dependents of `a` in the synthetic space.
    """
    args = [json.dumps(ORDER_MODELS), json.dumps(SYNTHETIC_SPACES)]
    orders = {seed: _run(_ORDER_WORKER, args, seed) for seed in HASHSEEDS}
    for name in _spaces():
        distinct = {tuple(orders[seed][name]) for seed in HASHSEEDS}
        assert len(distinct) == 1, (
            f"{name}: sampling order varies with PYTHONHASHSEED -> "
            f"{json.dumps({seed: orders[seed][name] for seed in HASHSEEDS}, indent=2)}"
        )


@pytest.mark.parametrize("name", list(DEPENDENCY_EDGES))
def test_sampling_order_respects_dependencies(name):
    """The determinism fix must not break topological validity."""
    space = _spaces()[name]
    edges = _dependency_edges(space)
    assert len(edges) == DEPENDENCY_EDGES[name], (
        f"{name} declares {len(edges)} dependent bounds, expected "
        f"{DEPENDENCY_EDGES[name]}; update DEPENDENCY_EDGES if the space changed"
    )

    order = UniformParameterSampler(param_space=space)._sampling_order
    position = {param: i for i, param in enumerate(order)}
    for param, dependency in edges:
        assert position[dependency] < position[param], (
            f"{name}: '{param}' depends on '{dependency}' but is sampled first"
        )


# --------------------------------------------------------------------------- #
# Defect 2: the theta-index blocks.
# --------------------------------------------------------------------------- #


def test_generators_without_offset_draw_distinct_bases(tmp_path):
    """Two generators built from one offset-free config share no theta.

    This is the SLURM-array case: every task runs the same YAML and nothing
    tells one apart from another. Each generator draws its own base from
    entropy, records it, and leaves the caller's config alone.
    """
    gc = _fast_config("full_ddm", 4, tmp_path)
    first = TrainingDataGenerator(config=gc).generate_data_training()
    second = TrainingDataGenerator(config=gc).generate_data_training()

    bases = {_recorded_offset(first), _recorded_offset(second)}
    assert len(bases) == 2, f"two offset-free generators drew the same base {bases}"
    assert all(0 <= base < 2**62 for base in bases), bases
    assert "parameter_sampler_index_offset" not in gc["pipeline"], (
        "the caller's config was mutated; a third generator would reuse the base"
    )
    _assert_disjoint(first["theta"], second["theta"], first["model_config"]["params"])


@pytest.mark.slow
def test_recorded_config_regenerates_the_file_in_another_process(tmp_path):
    """The recorded base reproduces the file: same thetas in a fresh interpreter.

    This is the audit path for a file whose YAML set no offset: read the
    `generator_config` it carries, build a generator from it elsewhere, get the
    same thetas back. The worker runs under a different PYTHONHASHSEED, which
    makes this the reproducibility half of defect 1 as well -- the theta
    *values* would survive a permuted sampling order, the rows would not.
    """
    offset_free = _fast_config("full_ddm", 4, tmp_path)
    produced = TrainingDataGenerator(config=offset_free).generate_data_training()

    # The recorded config, and the offset-free config it came from: only the
    # first may reproduce the file, or the recorded value is not what makes it
    # reproducible (a fixed default offset would pass the first check alone).
    configs = [produced["generator_config"], offset_free]
    recorded, fresh = _run(_REGENERATE_WORKER, [json.dumps(configs)], "12345")["thetas"]
    assert recorded == produced["theta"].tolist(), (
        f"recorded offset {_recorded_offset(produced)} did not reproduce the file"
    )
    assert fresh != produced["theta"].tolist(), (
        "an offset-free config reproduced the file: no base was drawn"
    )


def test_offset_blocks_tile_the_index_line(tmp_path):
    """Offset 0 called twice == offsets 0 and n: blocks are disjoint and contiguous.

    The second file of a `--n-files 2` run and the first file of a run started
    at offset n_parameter_sets are the same file. That equality is what makes
    task_id * n_files * n_parameter_sets a collision-free array formula; the
    disjointness is the training-data half of defect 2 (pre-fix, the N files
    of a `--n-files N` loop carried N identical theta sets).
    """
    n = 4
    at_zero = TrainingDataGenerator(config=_fast_config("full_ddm", n, tmp_path, 0))
    first = at_zero.generate_data_training()
    second = at_zero.generate_data_training()
    shifted = TrainingDataGenerator(
        config=_fast_config("full_ddm", n, tmp_path, n)
    ).generate_data_training()

    assert _recorded_offset(first) == 0
    assert _recorded_offset(second) == _recorded_offset(shifted) == n
    assert second["theta"].tolist() == shifted["theta"].tolist(), (
        "the second block of one generator differs from a generator started there"
    )
    _assert_disjoint(first["theta"], second["theta"], first["model_config"]["params"])


def test_explicit_offset_is_the_first_index_wherever_it_is_placed(tmp_path):
    """Nested (CLI) and root (programmatic) placement mean the same thing.

    Each call records its own first index, offset + k * n_parameter_sets for the
    k-th file, which is exactly what regenerating that one file needs.
    """
    n, offset = 4, 1_000
    for where in ("pipeline", "root"):
        gen = TrainingDataGenerator(
            config=_fast_config("ddm", n, tmp_path, offset, where=where)
        )
        calls, outputs = _consumed_indices(gen, n_calls=3)
        expected = [list(range(offset + k * n, offset + (k + 1) * n)) for k in range(3)]
        assert calls == expected, f"{where}: consumed {calls}"
        assert [_recorded_offset(o) for o in outputs] == [
            offset + k * n for k in range(3)
        ]


def test_subrun_remainder_consumes_every_theta_index(tmp_path):
    """A non-divisible n_parameter_sets / n_subruns pair must still use every index.

    Floor division alone left the last `n_parameter_sets % n_subruns` indices
    ungenerated while the cursor advanced by the full n_parameter_sets, so those
    indices were skipped for the life of the process and each file came up short.
    """
    gc = _fast_config("ddm", 5, tmp_path, 0)  # 5 // 2 == 2, remainder 1
    gc["pipeline"]["n_subruns"] = 2
    calls, _ = _consumed_indices(TrainingDataGenerator(config=gc), n_calls=2)
    assert calls == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], f"consumed {calls}"


class _RecordingPipeline:
    """A DataGenerationPipelineProtocol implementation that only records indices.

    Deliberately not a SimulationPipeline: a custom pipeline knows nothing about
    `parameter_sampler_index_offset`, so the offset can only reach it if the
    generator applies it before the index is handed over.
    """

    def __init__(self, generator_config: dict, model_config: dict):
        """Keep the two configs the protocol requires and start recording."""
        self.generator_config = generator_config
        self.model_config = model_config
        self.indices: list[int] = []

    def generate_for_parameter_set(self, parameter_sampling_seed, random_seed=None):
        """Record the index and return a minimal success record."""
        self.indices.append(int(parameter_sampling_seed))
        return _stub_result()

    def get_param_space(self):
        """Return the model's bounds, as the protocol asks."""
        return self.model_config["param_bounds_dict"]


def test_offset_reaches_a_custom_pipeline(tmp_path):
    """A custom pipeline receives final indices, explicit offset or entropy base."""
    n, offset = 3, 500
    mc = ModelConfigBuilder.from_model("ddm")

    explicit = _RecordingPipeline(_fast_config("ddm", n, tmp_path, offset), mc)
    assert isinstance(explicit, DataGenerationPipelineProtocol)
    gen = TrainingDataGenerator(config=explicit)
    gen.generate_data_training()
    gen.generate_data_training()
    assert explicit.indices == list(range(offset, offset + 2 * n))

    bare = _RecordingPipeline(_fast_config("ddm", n, tmp_path), mc)
    output = TrainingDataGenerator(config=bare).generate_data_training()
    base = _recorded_offset(output)
    assert bare.indices == list(range(base, base + n))
    # The pipeline's own config dict is recorded into, not written to: a second
    # generator on this pipeline must still see no explicit offset.
    assert "parameter_sampler_index_offset" not in bare.generator_config["pipeline"]


class _StopAfterSampling(Exception):
    """Sentinel that ends PyDDM generation once theta has been sampled."""


class _ThetaCapture:
    """Estimator-builder stub: records the sampled theta, then aborts the run."""

    def __init__(self):
        """Start with no recorded theta."""
        self.theta = None

    def build(self, theta_dict, simulations=None):
        """Record theta and raise, so no Fokker-Planck solve is needed."""
        self.theta = _scalars(theta_dict)
        raise _StopAfterSampling


def _scalars(theta_dict: dict) -> dict[str, float]:
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in theta_dict.items()}


def _pyddm_theta(gc: dict, mc: dict, index: int) -> dict:
    """Sample one theta through PyDDMPipeline with everything downstream stubbed."""
    capture = _ThetaCapture()
    pipeline = PyDDMPipeline(
        generator_config=gc,
        model_config=mc,
        estimator_builder=capture,
        training_strategy=object(),  # never reached
    )
    with pytest.raises(_StopAfterSampling):
        pipeline.generate_for_parameter_set(index, 1234)
    return capture.theta


def _simulation_theta(gc: dict, mc: dict, index: int) -> dict:
    """Sample one theta through SimulationPipeline's real entry point."""
    pipeline = SimulationPipeline(gc, mc, KDEEstimatorBuilder, MixtureTrainingStrategy)
    return _scalars(pipeline.generate_for_parameter_set(index, 1234)["theta"])


def test_pipelines_use_the_index_as_given(tmp_path):
    """Neither pipeline adds a config offset, and both read an index the same way.

    The generator hands the identical index to whichever pipeline the estimator
    type selected, so the index must mean the same theta in both, and a config
    offset must shift it exactly once -- in the generator, never again here.
    Before the PyDDM fix, that pipeline seeded only the legacy global RNG, which
    `sample()` never reads, so its thetas were irreproducible.
    """
    mc = ModelConfigBuilder.from_model("full_ddm")
    plain = _fast_config("full_ddm", 4, tmp_path)
    with_offset = _fast_config("full_ddm", 4, tmp_path, 4)

    assert _simulation_theta(plain, mc, 3) == _simulation_theta(with_offset, mc, 3), (
        "SimulationPipeline applied the config offset itself"
    )
    assert _pyddm_theta(plain, mc, 3) == _pyddm_theta(with_offset, mc, 3), (
        "PyDDMPipeline applied the config offset itself"
    )
    assert _pyddm_theta(plain, mc, 3) == _simulation_theta(plain, mc, 3), (
        "the two pipelines map one index to different thetas"
    )
    assert _pyddm_theta(plain, mc, 3) == _pyddm_theta(plain, mc, 3), (
        "PyDDM theta sampling is not reproducible for a fixed index"
    )
    assert _pyddm_theta(plain, mc, 3) != _pyddm_theta(plain, mc, 4), (
        "the index does not reach the parameter RNG"
    )


def test_cli_yaml_offset_lands_in_the_pipeline_section(tmp_path):
    """The premise of the nested lookup: the CLI files the YAML key under 'pipeline'.

    `collect_data_generator_config` forwards the whole PIPELINE section and drops
    root-level YAML keys, so a lookup that only reads the config root turns
    PARAMETER_SAMPLER_INDEX_OFFSET into a no-op for every CLI user.
    """
    from ssms.cli.generate import collect_data_generator_config

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
    # Absent is None, not 0: only then does the generator draw a base, and an
    # explicit 0 has to stay an explicit 0.
    assert get_parameter_sampler_index_offset({"pipeline": {}}) is None
    assert (
        get_parameter_sampler_index_offset(
            {"pipeline": {"parameter_sampler_index_offset": 0}}
        )
        == 0
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        get_parameter_sampler_index_offset(
            {"pipeline": {"parameter_sampler_index_offset": -1}}
        )
