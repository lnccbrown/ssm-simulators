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

from volterra_accuracy.paper_method_prototypes import CenteredAccumulator  # noqa: E402
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
