"""Pluggable fixation-continuation strategies + a positive-distribution factory for the aDDM.

The aDDM posterior-predictive path conditions on observed fixations (Mode 2). When a
re-simulated particle has not decided by the last observed fixation, the drift "freezes" at
the last gaze. This module makes that tail behaviour pluggable, and shares a ``scipy.stats``
positive-distribution factory with the aDDM's Mode-1 self-sampling.

Strategies (name -> callable), each mapping
``(node_r, d_r, flag_r, T, params, rng) -> (node_r, d_r, flag_r, max_d)``:

* ``prolong_last_fixation``  -- no new fixations; the engine holds the last gaze to ``T``
  (today's behaviour; the default).
* ``sample_continuation``    -- keep the observed fixations; draw the continuation (tail)
  durations from a chosen positive distribution and keep alternating past the last observed
  onset. (Appending an onset after the last observed onset resamples the censored last
  fixation's duration; further onsets are new alternating fixations.)
* ``resample_all_fixations`` -- ignore the observed fixation schedule; keep the observed
  stimulus (``r1``/``r2`` untouched upstream), re-sample the first-gaze ``flag`` and a fresh
  schedule. The drift still conditions on the stimulus, so the PPC stays comparable to data.

All durations come from :data:`POSITIVE_DISTRIBUTIONS` (scipy.stats), so any positive-support
distribution is available by name with scipy-native parameters. Mirrors the
``attention_process`` registry idiom (name -> callable + a ``resolve_*`` helper).
"""

from collections.abc import Callable

import numpy as np
from scipy import stats

# scipy.stats positive-support continuous distributions, by user-facing name. Extend by PR.
POSITIVE_DISTRIBUTIONS: dict[str, object] = {
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "invgamma": stats.invgamma,
    "weibull": stats.weibull_min,
    "invgauss": stats.invgauss,
    "halfnormal": stats.halfnorm,
    "exponential": stats.expon,
}


def resolve_distribution(name):
    """Return the scipy.stats distribution registered under ``name``."""
    try:
        return POSITIVE_DISTRIBUTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown continuation distribution {name!r}; "
            f"available: {sorted(POSITIVE_DISTRIBUTIONS)}."
        ) from None


def draw_durations(dist_name, dist_params, size, rng):
    """Draw an array of ``size`` positive durations from ``dist_name`` (scipy-native params)."""
    samples = resolve_distribution(dist_name).rvs(
        size=size, random_state=rng, **(dist_params or {})
    )
    return np.asarray(samples, dtype=np.float64)


def generate_schedule(n, T, dist_name, dist_params, rng):
    """Self-sample a padded ``(n, max_d)`` fixation-onset schedule + per-row stage count ``d``.

    Durations are drawn from ``dist_name`` and cumulatively summed; column 0 is anchored at 0.0
    (trial start). ``d`` = number of onsets strictly before ``T`` (>=1). The pre-draw budget is
    sized from the distribution's analytic mean, then over-allocated columns (onsets >= T) are
    trimmed to the realised ``max_d``.
    """
    mean = float(resolve_distribution(dist_name).mean(**(dist_params or {})))
    budget = max(int(T / mean) + 50, 10)
    durations = draw_durations(dist_name, dist_params, (n, budget), rng)
    sacc = np.concatenate(
        [np.zeros((n, 1), dtype=np.float64), np.cumsum(durations, axis=1)], axis=1
    )
    d = np.maximum((sacc < T).sum(axis=1), 1).astype(np.int32)
    max_d = int(d.max())
    return np.ascontiguousarray(sacc[:, :max_d], dtype=np.float64), d, max_d


def _require_dist(params, mode):
    if params is None or "dist" not in params:
        raise ValueError(
            f"{mode} requires continuation_params="
            "{'dist': <name>, 'dist_params': {...}}."
        )
    return params["dist"], params.get("dist_params", {})


# --------------------------------------------------------------------------- #
# Strategies: (node_r, d_r, flag_r, T, params, rng) -> (node_r, d_r, flag_r, max_d)
# --------------------------------------------------------------------------- #
def prolong_last_fixation(node_r, d_r, flag_r, T, params, rng):
    """No continuation: the engine holds the last observed gaze to ``T`` (today's behaviour)."""
    if params is not None:
        raise ValueError(
            "prolong_last_fixation takes no continuation_params (got a non-None value)."
        )
    return node_r, d_r, flag_r, node_r.shape[1]


def sample_continuation(node_r, d_r, flag_r, T, params, rng):
    """Keep observed fixations; draw the continuation (tail) durations from ``params['dist']``."""
    dist_name, dist_params = _require_dist(params, "sample_continuation")
    if params.get("resample_last_fixation", True) is not True:
        raise NotImplementedError(
            "resample_last_fixation=False (rt-conditioned continuation) is not yet supported."
        )
    n = node_r.shape[0]
    last_onset = node_r[np.arange(n), d_r - 1]  # last observed onset per row (anchor)
    mean = float(resolve_distribution(dist_name).mean(**(dist_params or {})))
    n_cont = max(int((T - last_onset).max() / mean) + 50, 10)
    onsets = last_onset[:, None] + np.cumsum(
        draw_durations(dist_name, dist_params, (n, n_cont), rng), axis=1
    )
    node_r = np.ascontiguousarray(
        np.concatenate([node_r, onsets], axis=1), dtype=np.float64
    )
    d_r = (d_r + (onsets < T).sum(axis=1)).astype(np.int32)
    return node_r, d_r, flag_r, node_r.shape[1]


def resample_all_fixations(node_r, d_r, flag_r, T, params, rng):
    """Ignore observed fixations; re-sample first-gaze + a fresh schedule (keep stimulus r1/r2)."""
    dist_name, dist_params = _require_dist(params, "resample_all_fixations")
    n = node_r.shape[0]
    flag_r = rng.integers(0, 2, n).astype(
        np.int64
    )  # re-sample first gaze; r1/r2 stay observed
    node_r, d_r, max_d = generate_schedule(n, T, dist_name, dist_params, rng)
    return node_r, d_r, flag_r, max_d


FIXATION_CONTINUATION_MODES: dict[str, Callable] = {
    "prolong_last_fixation": prolong_last_fixation,
    "sample_continuation": sample_continuation,
    "resample_all_fixations": resample_all_fixations,
}


def resolve_continuation_mode(mode):
    """Resolve a registry name or a callable to a concrete continuation strategy."""
    if callable(mode):
        return mode
    if isinstance(mode, str):
        try:
            return FIXATION_CONTINUATION_MODES[mode]
        except KeyError:
            raise ValueError(
                f"Unknown continuation_mode {mode!r}; "
                f"available: {sorted(FIXATION_CONTINUATION_MODES)}."
            ) from None
    raise TypeError(
        f"continuation_mode must be a registry name (str) or a callable, got {type(mode)!r}."
    )


if (
    __name__ == "__main__"
):  # pragma: no cover - smoke check (runs without the cython build)
    _rng = np.random.default_rng(0)
    _node = np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.0]])
    _d = np.array([3, 2], dtype=np.int32)
    _flag = np.array([0, 1], dtype=np.int64)
    _p = {"dist": "gamma", "dist_params": {"a": 2.0, "scale": 0.3}}

    n0, d0, f0, m0 = prolong_last_fixation(_node, _d, _flag, 5.0, None, _rng)
    assert (
        n0.shape == _node.shape and np.array_equal(d0, _d) and np.array_equal(f0, _flag)
    )

    n1, d1, f1, m1 = sample_continuation(_node, _d, _flag, 5.0, _p, _rng)
    assert m1 > _node.shape[1] and np.array_equal(
        f1, _flag
    )  # extends schedule, flag unchanged
    assert (
        n1[0, _d[0]] > _node[0, _d[0] - 1]
    )  # first continuation onset > last observed onset

    n2, d2, f2, m2 = resample_all_fixations(_node, _d, _flag, 5.0, _p, _rng)
    assert n2.shape[0] == 2 and bool(
        (n2[:, 0] == 0.0).all()
    )  # fresh schedule anchored at 0.0

    for bad in (
        lambda: resolve_distribution("nope"),
        lambda: resolve_continuation_mode("nope"),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise SystemExit("expected ValueError")

    print("fixation_continuation smoke check passed")
