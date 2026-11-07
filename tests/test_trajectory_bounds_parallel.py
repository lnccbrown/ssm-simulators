"""Parallel (``n_threads > 1``) Euler loops must stay inside their per-step arrays.

Companion to ``test_trajectory_bounds.py`` (issue #115) for the OpenMP/GSL
paths, which that module's ``n_threads=1`` cases never reach:

* the ddm-family, levy and ornstein kernels copy the boundary (and drift)
  evaluated on ``np.arange(0, max_t + delta_t, delta_t)`` into ``num_steps``
  columns. At e.g. ``(max_t, delta_t) = (20, 0.01)`` that grid has one more point
  than ``num_steps`` and the copy raised ``ValueError``;
* parallel LCA sized its step bound as ``int(max_t / delta_t + 2)``, one more
  than its boundary array has at e.g. ``delta_t=0.005``, so a walker that ran
  out the clock read one boundary value past the end;
* the seq2 low-dim walkers start at the high-dim walker's step index and are
  now bounded by ``num_steps`` as well.

``check_parallel_request`` silently falls back to one thread without OpenMP or
GSL, so these tests skip unless both are compiled in (the ``test_multithreading``
CI job installs GSL and runs this module). The child asserts that the parallel
path really ran. Trajectories are not recorded in parallel, so only RTs, choices
and shapes are checked.
"""

import textwrap

import pytest

from ssms.config import model_config

from .test_trajectory_bounds import (
    FLOAT32_OVERRUN_GRID,
    NO_HIT_OVERRIDES,
    _no_hit_theta,
    _run_child,
)

_openmp_status = pytest.importorskip("cssm._openmp_status")

pytestmark = pytest.mark.skipif(
    not (_openmp_status.is_gsl_available() and _openmp_status.is_openmp_available()),
    reason=(
        "parallel path needs cssm built with GSL and OpenMP; without them "
        "n_threads>1 silently falls back to the sequential path"
    ),
)

N_THREADS = 2

# A time grid with one point more than num_steps: the boundary/drift copies in
# the parallel paths raised ValueError here.
EXTRA_GRID_POINT = (20.0, 0.01)

# Runs in the child with n_threads=2: check that the parallel path was taken,
# simulate every (max_t, delta_t) case and print a JSON summary.
_PARALLEL_CHILD = textwrap.dedent(
    """
    import json, sys, warnings
    import numpy as np
    from cssm._openmp_status import check_parallel_request
    warnings.simplefilter("ignore")

    spec = json.loads(sys.argv[1])
    n_threads = spec["n_threads"]
    assert check_parallel_request(n_threads, warn=False) == n_threads

    def run_model(max_t, delta_t, seed):
        from ssms.basic_simulators.simulator_class import Simulator
        theta = {k: np.array([v], dtype=np.float32) for k, v in spec["theta"].items()}
        return Simulator(spec["model"]).simulate(
            theta=theta, n_samples=spec["n_samples"], delta_t=delta_t, max_t=max_t,
            no_noise=spec["no_noise"], smooth_unif=False, random_state=seed,
            n_threads=n_threads,
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
            n_threads=n_threads,
        )

    run = run_kernel if "kernel" in spec else run_model
    summary = []
    for max_t, delta_t in spec["grid"]:
        for seed in range(spec["repeats"]):
            out = run(max_t, delta_t, seed)
            meta = out["metadata"]
            # The kernels report the thread count they actually used ...
            if "n_threads" in meta:
                assert meta["n_threads"] == n_threads, meta["n_threads"]
            # ... and only the sequential path writes the trajectory (row 0 at
            # least), so an untouched sentinel array means the parallel path ran.
            traj = meta.get("trajectory")
            if traj is not None and np.ndim(traj) == 2:
                assert np.all(np.asarray(traj) == -999), "sequential path ran"
        rts = np.asarray(out["rts"], dtype=np.float64)
        choices = np.asarray(out["choices"])
        summary.append(dict(
            max_t=max_t, delta_t=delta_t,
            rts_shape=list(rts.shape), choices_shape=list(choices.shape),
            rts=rts.ravel().tolist(), choices=choices.ravel().tolist(),
        ))
    print(json.dumps(summary))
    """
)


def _run_parallel(spec):
    spec = dict(spec, n_threads=N_THREADS)
    return _run_child(spec, child=_PARALLEL_CHILD)


def _check_parallel_case(case, shape, valid_choices):
    """RTs are the capped no-hit horizon (or an omission), choices are valid.

    ``shape`` is ``[n_samples, n_trials]`` from ``Simulator`` and
    ``[n_samples, n_trials, 1]`` from a raw kernel.
    """
    max_t, delta_t = case["max_t"], case["delta_t"]
    assert case["rts_shape"] == shape, case["rts_shape"]
    assert case["choices_shape"] == shape, case["choices_shape"]
    for rt in case["rts"]:
        assert rt == -999.0 or 0.0 <= rt <= max_t + 2 * delta_t + 1e-3, case
    # racing_diffusion reports an omission (rt == -999) with choice -1
    omission_choice = {-1} if -999.0 in case["rts"] else set()
    assert set(case["choices"]) <= set(valid_choices) | omission_choice, case


def _model_spec(model, grid, overrides=None, n_samples=4, repeats=2):
    if overrides is None:
        overrides = NO_HIT_OVERRIDES.get(model, {})
    return dict(
        model=model,
        theta=_no_hit_theta(model, overrides),
        grid=grid,
        n_samples=n_samples,
        repeats=repeats,
        no_noise=True,
    )


# Models whose parallel path copies a precomputed boundary (and drift) array.
BOUNDARY_COPY_MODELS = [
    "ddm",  # ddm_flexbound
    "angle",  # ddm_flexbound, boundary grows with t
    "full_ddm",  # full_ddm
    "ddm_st",  # full_ddm_rv
    "gamma_drift",  # ddm_flex: drift and boundary copies
    "conflict_stimflexrel1_leak",  # ddm_flex_leak: drift and boundary copies
    "conflict_stimflexrel1_leak2",  # ddm_flex_leak2: 2-D drift and boundary copies
    "levy",  # levy_flexbound
    "ornstein",  # ornstein_uhlenbeck
]

# Same copy, in kernels without a registered model config (z off-centre, so the
# no-noise walker has a sign and a valid choice).
BOUNDARY_COPY_KERNELS = {
    "ddm_sdv": ("ddm_models", dict(v=0.0, a=1.0, z=0.6, t=0.0, sv=0.0)),
    "ddm_flexbound_flat": ("ddm_models", dict(v=0.0, a=1.0, z=0.6, t=0.0)),
}


@pytest.mark.parametrize("model", BOUNDARY_COPY_MODELS)
def test_parallel_boundary_copy_extra_grid_point(model):
    """The boundary/drift copy must fit ``num_steps`` columns."""
    n_samples = 4
    spec = _model_spec(model, [EXTRA_GRID_POINT], n_samples=n_samples)
    for case in _run_parallel(spec):
        _check_parallel_case(case, [n_samples, 1], model_config[model]["choices"])


@pytest.mark.parametrize("kernel", sorted(BOUNDARY_COPY_KERNELS))
def test_parallel_boundary_copy_unregistered_kernels(kernel):
    module, theta = BOUNDARY_COPY_KERNELS[kernel]
    n_samples = 4
    spec = dict(
        kernel=kernel,
        module=module,
        theta=theta,
        grid=[EXTRA_GRID_POINT],
        n_samples=n_samples,
        repeats=2,
    )
    for case in _run_parallel(spec):
        _check_parallel_case(case, [n_samples, 1, 1], [-1, 1])


@pytest.mark.parametrize("delta_t", [0.01, 0.005, 0.002, 0.001, 1e-4])
def test_parallel_lca_no_hit(delta_t):
    """LCA's step bound was one past its boundary array at e.g. delta_t=0.005."""
    n_samples = 4
    spec = _model_spec("lca_4", [(20.0, delta_t)], overrides={}, n_samples=n_samples)
    for case in _run_parallel(spec):
        _check_parallel_case(case, [n_samples, 1], model_config["lca_4"]["choices"])


@pytest.mark.parametrize(
    "model", ["ddm_seq2", "ddm_par2", "race_2", "racing_diffusion_3"]
)
def test_parallel_no_hit_horizon(model):
    """Multi-stage and race kernels run out the clock without leaving their arrays."""
    n_samples = 4
    grid = [EXTRA_GRID_POINT, *FLOAT32_OVERRUN_GRID, (20.0, 1e-4)]
    spec = _model_spec(model, grid, n_samples=n_samples, repeats=1)
    for case in _run_parallel(spec):
        _check_parallel_case(case, [n_samples, 1], model_config[model]["choices"])
