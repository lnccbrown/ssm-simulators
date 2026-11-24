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
