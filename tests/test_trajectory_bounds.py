"""Euler-loop simulators must not index past their per-step arrays.

Regression tests for https://github.com/lnccbrown/ssm-simulators/issues/115
("Data generator unstable for some values of delta_t").

The sequential kernels stepped the walker while ``t_particle <= deadline`` and
wrote ``traj[ix]`` (and read ``boundary[ix]``) with no bound on ``ix``. The time
accumulator drifts from ``ix * delta_t`` in floating point, so a walker that does
not hit a boundary before ``max_t`` (the first sample of the first trial records
the trajectory) could step to ``ix == len(traj)`` or beyond:

* float32 clocks (all kernels but sequential LCA) overrun at e.g.
  ``(max_t, delta_t)`` = (20, 0.005), (10, 0.002), (5, 0.001) and, by ~186 rows,
  at (20, 1e-4);
* sequential LCA accumulated an untyped (double) clock and overran at 0.01
  and 0.005 for every ``max_t`` tested.

With bounds checking off, the overrun corrupts the heap: the process aborts
(``malloc(): invalid size``), segfaults or hangs, often only on a later
allocation. Each case therefore runs in a subprocess, under glibc's malloc
consistency checks when they are available, so that even a one-element
overrun aborts the child instead of passing silently or killing the pytest
worker.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ssms.config import model_config

# glibc >= 2.34 moved MALLOC_CHECK_ into this preloadable library. With it, a
# write one byte past a heap block is caught when the block is freed.
_MALLOC_DEBUG_LIB = Path("/lib/x86_64-linux-gnu/libc_malloc_debug.so.0")

# (max_t, delta_t) pairs where the float32 step clock is still <= max_t after
# int(max_t / delta_t) steps, so a time-only loop overran the trajectory.
FLOAT32_OVERRUN_GRID = [(20.0, 0.005), (10.0, 0.002), (5.0, 0.001)]

# Parameters that keep every walker away from its boundary until max_t when
# combined with no_noise=True: zero drift, start inside the boundaries.
NO_HIT_OVERRIDES = {
    # ddm_models.pyx
    "ddm_legacy": {},  # ddm
    "ddm": {},  # ddm_flexbound
    "angle": {"theta": -0.1},  # ddm_flexbound, boundary grows with t
    "full_ddm": {"sz": 0.0, "sv": 0.0, "st": 0.0},  # full_ddm
    "ddm_st": {"st": 0.0},  # full_ddm_rv
    "gamma_drift": {"c": 0.0},  # ddm_flex
    "conflict_stimflexrel1_leak": {"vt": 0.0, "vd": 0.0},  # ddm_flex_leak
    "conflict_stimflexrel1_leak2": {"vt": 0.0, "vd": 0.0},  # ddm_flex_leak2
    "tradeoff_no_bias": {},  # ddm_flexbound_tradeoff
    # levy_models.pyx, ornstein_models.pyx
    "levy": {},
    "ornstein": {},
    # parallel_models.pyx
    "ddm_par2": {},
    # sequential_models.pyx
    "ddm_seq2": {},
    "ddm_mic2_adj": {},  # ddm_flexbound_mic2_ornstein
    "ddm_mic2_multinoise_no_bias": {},  # ddm_flexbound_mic2_multinoise
    # race_models.pyx
    "race_2": {},
    "racing_diffusion_3": {"v0": 0.0, "v1": 0.0, "v2": 0.0},
    "lca_no_bias_angle_4": {"theta": -0.1},
}

# Kernels with no registered model config, called directly.
DIRECT_KERNELS = {
    "full_ddm_hddm_base": (
        "ddm_models",
        dict(v=0.0, a=2.0, z=0.5, t=0.0, sz=0.0, sv=0.0, st=0.0),
    ),
    "ddm_sdv": ("ddm_models", dict(v=0.0, a=1.0, z=0.5, t=0.0, sv=0.0)),
    "ddm_flexbound_flat": ("ddm_models", dict(v=0.0, a=1.0, z=0.5, t=0.0)),
    "ddm_flexbound_mic2_ornstein_multinoise": (
        "sequential_models",
        dict(
            vh=0.0,
            vl1=0.0,
            vl2=0.0,
            a=1.0,
            zh=0.5,
            zl1=0.5,
            zl2=0.5,
            d=0.5,
            g=0.0,
            t=0.0,
        ),
    ),
    "ddm_flexbound_mic2_unnormalized_ornstein_multinoise": (
        "sequential_models",
        dict(
            vh=0.0,
            vl1=0.0,
            vl2=0.0,
            a=1.0,
            zh=0.5,
            zl1=0.5,
            zl2=0.5,
            d=0.5,
            g=0.0,
            t=0.0,
        ),
    ),
}

# Runs in the child: simulate every (max_t, delta_t) case, check the outputs,
# and print a JSON summary. Any heap overrun aborts the child before that.
_CHILD = textwrap.dedent(
    """
    import json, sys, warnings
    import numpy as np
    warnings.simplefilter("ignore")

    spec = json.loads(sys.argv[1])
    OMISSION = -999.0

    def run_model(max_t, delta_t, seed):
        from ssms.basic_simulators.simulator_class import Simulator
        theta = {k: np.array([v], dtype=np.float32) for k, v in spec["theta"].items()}
        return Simulator(spec["model"]).simulate(
            theta=theta, n_samples=spec["n_samples"], delta_t=delta_t, max_t=max_t,
            no_noise=spec["no_noise"], smooth_unif=False, random_state=seed,
        )

    def run_kernel(max_t, delta_t, seed):
        import importlib
        from ssms.basic_simulators.boundary_functions import constant
        fun = getattr(importlib.import_module("cssm." + spec["module"]), spec["kernel"])
        arrays = {k: np.array([v], dtype=np.float32) for k, v in spec["theta"].items()}
        arrays["deadline"] = np.array([999.0], dtype=np.float32)
        arrays["s"] = np.array([0.0], dtype=np.float32)  # no noise
        return fun(
            **arrays, delta_t=delta_t, max_t=max_t, n_samples=spec["n_samples"],
            n_trials=1, boundary_fun=constant, boundary_params={}, random_state=seed,
            return_option="full", smooth_unif=False, print_info=False,
        )

    run = run_kernel if "kernel" in spec else run_model
    summary = []
    for max_t, delta_t in spec["grid"]:
        for seed in range(spec["repeats"]):
            out = run(max_t, delta_t, seed)
        rts = np.asarray(out["rts"], dtype=np.float64).ravel()
        choices = np.asarray(out["choices"]).ravel()
        traj = out["metadata"].get("trajectory")
        has_traj = traj is not None and np.ndim(traj) == 2
        summary.append(dict(
            max_t=max_t, delta_t=delta_t,
            rts=rts.tolist(), choices=choices.tolist(),
            traj_rows=int(np.shape(traj)[0]) if has_traj else None,
            traj_last_row_recorded=(
                bool(np.any(traj[-1] != OMISSION)) if has_traj else None
            ),
        ))
    print(json.dumps(summary))
    """
)


def _run_child(spec, timeout=300, child=_CHILD):
    """Run the ``child`` script on ``spec`` and return its JSON summary."""
    env = dict(os.environ)
    if sys.platform.startswith("linux") and _MALLOC_DEBUG_LIB.exists():
        env["LD_PRELOAD"] = str(_MALLOC_DEBUG_LIB)
        env["MALLOC_CHECK_"] = "3"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", child, json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:  # heap corruption can deadlock malloc
        pytest.fail(f"simulator hung (likely heap corruption): {spec}")
    assert proc.returncode == 0, (
        f"simulator child exited with {proc.returncode} (heap overrun?) for "
        f"{spec}\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _check_no_hit_case(case, valid_choices=None):
    """No-hit walker: RT is capped at the horizon, trajectory reaches its end."""
    max_t, delta_t = case["max_t"], case["delta_t"]
    # The walker takes at most int(max_t / delta_t) + 1 steps; the reported RT is
    # the float32 clock, so allow rounding slack but not extra steps.
    for rt in case["rts"]:
        assert rt == -999.0 or 0.0 <= rt <= max_t + 2 * delta_t + 1e-3, case
    if valid_choices is not None:
        # racing_diffusion reports an omission (rt == -999) with choice -1
        omission_choice = {-1} if -999.0 in case["rts"] else set()
        assert set(case["choices"]) <= set(valid_choices) | omission_choice, case
    if case["traj_rows"] is None:  # kernel records no trajectory
        return
    # The trajectory spans the time grid and a walker that runs out the clock
    # reaches its last row (writes past that row are what aborted the child).
    assert case["traj_rows"] == int(round(max_t / delta_t)) + 1, case
    assert case["traj_last_row_recorded"], case


@pytest.mark.parametrize("delta_t", [0.01, 0.005, 0.002, 0.001])
def test_lca_issue_115_reproducer(delta_t):
    """The reproducer from issue #115: noisy LCA with a growing boundary."""
    theta = dict(
        v0=0.0, v1=0.0, v2=0.0, v3=0.0, a=3.0, z=0.0, g=1.0, b=1.0, t=0.0, theta=-0.1
    )
    spec = dict(
        model="lca_no_bias_angle_4",
        theta=theta,
        grid=[(20.0, delta_t)],
        n_samples=5,
        repeats=40,
        no_noise=False,
    )
    summary = _run_child(spec)
    (case,) = summary
    assert all(0.0 <= rt <= 20.0 + 2 * delta_t for rt in case["rts"]), case
    assert set(case["choices"]) <= set(model_config["lca_no_bias_angle_4"]["choices"])
    assert case["traj_rows"] == int(round(20.0 / delta_t)) + 1


@pytest.mark.parametrize("delta_t", [0.01, 0.005, 0.002, 0.001])
@pytest.mark.parametrize("max_t", [20.0, 5.0])
def test_lca_no_hit(max_t, delta_t):
    """Sequential LCA, whose untyped double clock overran at 0.01 and 0.005."""
    spec = dict(
        model="lca_4",
        theta=_no_hit_theta("lca_4", {}),
        grid=[(max_t, delta_t)],
        n_samples=3,
        repeats=3,
        no_noise=True,
    )
    summary = _run_child(spec)
    for case in summary:
        _check_no_hit_case(case, model_config["lca_4"]["choices"])


def _no_hit_theta(model, overrides):
    config = model_config[model]
    theta = dict(zip(config["params"], map(float, config["default_params"])))
    if "t" in theta:
        theta["t"] = 0.0
    if "z" in theta:
        theta["z"] = 0.6  # off-centre, so a no-noise walker has a sign
    theta.update(overrides)
    return theta


@pytest.mark.parametrize("model", sorted(NO_HIT_OVERRIDES))
def test_no_hit_walker_stays_in_bounds(model):
    """One model per Euler-loop kernel, at the float32-overrun (max_t, delta_t)."""
    spec = dict(
        model=model,
        theta=_no_hit_theta(model, NO_HIT_OVERRIDES[model]),
        grid=FLOAT32_OVERRUN_GRID,
        n_samples=3,
        repeats=3,
        no_noise=True,
    )
    summary = _run_child(spec)
    for case in summary:
        _check_no_hit_case(case, model_config[model]["choices"])


@pytest.mark.parametrize("kernel", sorted(DIRECT_KERNELS))
def test_no_hit_unregistered_kernels(kernel):
    """Kernels without a model config get the same no-hit check."""
    module, theta = DIRECT_KERNELS[kernel]
    spec = dict(
        kernel=kernel,
        module=module,
        theta=theta,
        grid=FLOAT32_OVERRUN_GRID,
        n_samples=3,
        repeats=3,
    )
    summary = _run_child(spec)
    for case in summary:
        _check_no_hit_case(case)


@pytest.mark.parametrize("model", ["ddm", "ddm_seq2", "race_2", "tradeoff_no_bias"])
def test_no_hit_small_delta_t(model):
    """At delta_t=1e-4 the float32 clock lags ~186 steps behind at t=20.

    tradeoff_no_bias records no trajectory, but its per-step bias trace overran.
    """
    spec = dict(
        model=model,
        theta=_no_hit_theta(model, NO_HIT_OVERRIDES[model]),
        grid=[(20.0, 1e-4)],
        n_samples=2,
        repeats=1,
        no_noise=True,
    )
    summary = _run_child(spec)
    for case in summary:
        _check_no_hit_case(case, model_config[model]["choices"])
