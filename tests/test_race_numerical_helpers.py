"""Numerical regressions for the race research helpers in notebooks/volterra_accuracy.

The helpers are not package code; they are tested here so their numerics stay
checked without executing a notebook.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import invgauss, levy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from volterra_accuracy.paper_method_prototypes import (  # noqa: E402
    CenteredAccumulator,
    first_passage_times,
    sample_runner,
)
from volterra_accuracy.race import RaceAccumulator  # noqa: E402
from volterra_accuracy.race_single_stage import big_F  # noqa: E402


def hit_cdf(t, v, sigma, d):
    """P(first passage <= t) from SciPy's distributions, independent of the helpers.

    A Brownian motion with drift ``v`` and noise ``sigma`` reaches a level ``d``
    above it at an inverse-Gaussian time; for ``v < 0`` only with probability
    ``exp(2*v*d/sigma**2)``, and for ``v == 0`` at a Levy time.
    """
    shape = (d / sigma) ** 2
    if v == 0:
        return levy(scale=shape).cdf(t)
    reached = invgauss(mu=d / abs(v) / shape, scale=shape).cdf(t)
    return reached if v > 0 else np.exp(2 * v * d / sigma**2) * reached


# (relative drift, sigma, gap): both drift signs, zero, and three cases whose
# second term has a far-tail argument (the old approximation was off by up to 3e-5).
SINGLE_STAGE_CASES = [
    (0.67, 0.8, 1.35),
    (0.0, 0.8, 1.35),
    (-0.18, 0.8, 1.35),
    (-3.0, 1.0, 2.0),
    (3.999, 1.0, 3.999),
    (6.0, 0.5, 3.0),
    (8.0, 1.0, 4.0),
]


@pytest.mark.parametrize("v, sigma, d", SINGLE_STAGE_CASES)
def test_big_F_matches_scipy_and_is_monotone(v, sigma, d):
    t = np.geomspace(1e-3, 50, 400)
    cdf = big_F(t, v, sigma, d, 0.0, 100.0, 0.0)
    # Both sides are accurate to a few ulp; 1e-12 leaves three orders of headroom.
    np.testing.assert_allclose(cdf, hit_cdf(t, v, sigma, d), rtol=0, atol=1e-12)
    assert np.diff(cdf).min() >= -1e-15


def test_big_F_is_continuous_across_the_old_tail_switch():
    # The removed approximation switched formulas where the second term's
    # argument passes -8, and stepped down by 2.7e-5 there (#385).
    v = d = 3.999
    switch = ((8 + np.sqrt(64 - 4 * v * d)) / (2 * v)) ** 2
    t = switch + np.array([-1e-9, 1e-9])
    cdf = big_F(t, v, 1.0, d, 0.0, 10.0, 0.0)
    assert 0 <= cdf[1] - cdf[0] < 1e-8
    np.testing.assert_allclose(cdf, hit_cdf(t, v, 1.0, d), rtol=0, atol=1e-12)


def constant_accumulator(cls, breaks, v, sigma, d, order=80):
    """One unchanged process cut at ``breaks``: the cuts must not change its law."""
    k = len(breaks) - 1
    return cls(breaks, [v] * k, [sigma] * k, [0.0] * k, d, quadrature_order=order)


@pytest.mark.parametrize("cls", [RaceAccumulator, CenteredAccumulator])
@pytest.mark.parametrize("breaks", [[0, 0.01, 13], [0, 0.01, 0.5, 13]])
def test_stage_cdf_keeps_earlier_crossings_under_strong_negative_drift(cls, breaks):
    # eta = v*sqrt(elapsed)/sigma passes -11 and -12 inside the last stage
    # (t near 7.6 and 9). The shared density/CDF cutoff lost crossing mass from
    # there on, down to a CDF of 0.204 instead of 0.449 at t = 13 (#384).
    # The three-stage variant propagates through each backend's node grid.
    t = np.r_[0.01, 2.0, 8.86, 9.0, 13.0, np.linspace(7.0, 10.5, 141)]
    cdf = constant_accumulator(cls, breaks, -4.0, 1.0, 0.1).marginal(t)[1]
    # 1e-8: the adaptive integral runs at 1e-10 absolute / 1e-8 relative, and
    # Laguerre propagation at order 80 is good to about 1e-9 here.
    np.testing.assert_allclose(cdf, hit_cdf(t, -4.0, 1.0, 0.1), rtol=0, atol=1e-8)
    assert np.diff(cdf[5:]).min() >= -1e-12


@pytest.mark.parametrize("v", [0.67, 0.0, -0.18, -2.5])
def test_second_stage_cdf_is_accurate_and_continuous_at_the_node(v):
    # Two stages use the exact incoming density, so only the within-stage
    # integral is under test.
    acc = constant_accumulator(RaceAccumulator, [0, 0.3, 2], v, 0.8, 1.35, order=20)
    t = np.r_[np.linspace(0.01, 2, 300), 0.3 - 1e-9, 0.3 + 1e-9]
    cdf = acc.marginal(t)[1]
    np.testing.assert_allclose(cdf, hit_cdf(t, v, 0.8, 1.35), rtol=0, atol=1e-9)
    assert np.diff(cdf[:300]).min() >= -1e-12
    assert 0 <= cdf[-1] - cdf[-2] < 1e-8


def cdf_distance(sample, cdf, end_mass):
    """sup |empirical CDF - cdf| over all times; ``inf`` entries are 'not reached'.

    ``end_mass`` is the CDF's limit where the finite samples stop: the
    probability of ever reaching the boundary, or the CDF at a censoring time.
    """
    x = np.sort(sample[np.isfinite(sample)])
    steps = np.arange(x.size + 1) / sample.size
    exact = cdf(x)
    return max(
        np.abs(steps[1:] - exact).max(),
        np.abs(steps[:-1] - exact).max(),
        abs(steps[-1] - end_mass),
    )


def dkw_bound(n, alpha=1e-6):
    """Distance a correct sampler exceeds with probability at most ``alpha``."""
    return np.sqrt(np.log(2 / alpha) / (2 * n))


# (relative drift, sigma, gap): ordinary drifts of both signs, zero, drifts too
# small for rng.wald (a rounding residue among them), a strong drift, a drift
# away from the boundary that mostly never arrives, and a tiny gap.
FIRST_PASSAGE_CASES = [
    (0.67, 0.8, 1.35),
    (-0.18, 0.8, 1.35),
    (0.0, 0.8, 1.35),
    (1e-12, 0.8, 1.35),
    (-1e-12, 0.8, 1.35),
    (0.3 - 0.1 - 0.2, 0.8, 1.35),
    (1e-300, 0.8, 1.35),
    (5.0, 0.5, 2.0),
    (-1.0, 1.0, 1.0),
    (0.67, 0.8, 1e-3),
]


@pytest.mark.parametrize("v, sigma, d", FIRST_PASSAGE_CASES)
def test_first_passage_times_follow_the_exact_law(v, sigma, d):
    n = 200_000
    tau = first_passage_times(np.full(n, d), v, sigma, np.random.default_rng(386))
    assert (tau > 0).all()  # also rejects nan
    # A drift below 1e-9 changes the law by far less than the bound resolves,
    # so those cases are held to the driftless law.
    law = 0.0 if abs(v) < 1e-9 else v
    ever_reached = np.exp(min(2 * law * d / sigma**2, 0.0))
    distance = cdf_distance(tau, lambda t: hit_cdf(t, law, sigma, d), ever_reached)
    assert distance < dkw_bound(n)


def test_sample_runner_survives_a_rounding_residue_drift():
    # 0.3 - 0.1 - 0.2 is -2.8e-17, not 0; rng.wald turned that into 58% hits
    # at time zero (#386). The hit fraction must be the driftless one.
    d, sigma, horizon, n = 1.35, 0.8, 0.55, 20_000
    acc = RaceAccumulator([0.0, horizon], [0.3 - 0.1 - 0.2], [sigma], [0.0], d)
    hits, _ = sample_runner(acc, n, np.random.default_rng(3))
    p = hit_cdf(horizon, 0.0, sigma, d)
    assert (hits > 0).all()
    assert abs(np.isfinite(hits).mean() - p) < 5 * np.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("v", [0.67, 0.0, -0.18])
def test_sample_runner_is_unchanged_by_artificial_stage_cuts(v):
    # Each cut redraws the survivors' positions and restarts the clock; with
    # identical parameters in every stage the law must stay the single-stage one.
    n, deadline = 200_000, 2.0
    breaks = [0, 0.02, 0.57, 0.95, 1.42, deadline]
    acc = constant_accumulator(RaceAccumulator, breaks, v, 0.8, 1.35)
    hits, _ = sample_runner(acc, n, np.random.default_rng(368))
    distance = cdf_distance(
        hits, lambda t: hit_cdf(t, v, 0.8, 1.35), hit_cdf(deadline, v, 0.8, 1.35)
    )
    assert distance < dkw_bound(n)
