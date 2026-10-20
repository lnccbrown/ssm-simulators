"""How much of the reported numbers is approximation error.

The registered MLBA has no time discretisation: ``cssm.lba_vanilla`` solves the
first-passage time in closed form per sample, so ``delta_t`` plays no role and
the *only* approximation in a simulated choice probability is Monte-Carlo
sampling.  The reference used to pin that down -- the closed-form LBA in
``tests/mlba/mlba_reference.py`` -- does have an approximation (Simpson quadrature over
finishing time), so it is checked first, against a start-point quadrature and
against grid refinement, before being used to audit the simulator.

Layers checked here, innermost first:

1. the closed-form single-accumulator CDF/PDF, against a brute-force
   Gauss-Legendre integral over the start point and against finite differences;
2. the race quadrature, against refinement in both resolution and range;
3. the folded (``|N(d, s)|``, what ``cssm`` does) versus unfolded
   (``N(d, s)``, the textbook LBA) drift convention;
4. the simulator itself, against the closed form, at every condition the
   contribution reports a number for.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlba_reference import (
    CHOICE_SETS,
    CONDITIONS,
    PAPER_PARAMS,
    choice_probabilities,
    lba_cdf,
    lba_pdf,
    simulate_paper,
)
from scipy.stats import norm

from ssms.transforms.simulation.mlba_front_end import drift_rates

A, B, S = 1.0, 2.0, 1.0  # the paper's A = 1, chi = 2, s = 1
N_SIM = 200_000


# Gauss-Legendre is exact to machine precision for a smooth integrand long
# before 501 nodes; building the rule is the expensive part, so cache it.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(501)


def _brute_force_cdf(t: float, d: float, folded: bool) -> float:
    """(1/A) * int_0^A P(hit by t | start point z) dz, by Gauss-Legendre."""
    nodes, weights = _GL_NODES, _GL_WEIGHTS
    z = 0.5 * A * (nodes + 1.0)
    x = (B - z) / t  # drift needed to arrive by t
    p = 1.0 - norm.cdf((x - d) / S)
    if folded:
        p = p + norm.cdf((-x - d) / S)
    return float(0.5 * (weights * p).sum())


@pytest.mark.parametrize("folded", [True, False])
@pytest.mark.parametrize("d", [0.3, 1.0, 5.0, 7.4])
@pytest.mark.parametrize("t", [0.05, 0.2, 0.3, 0.5, 1.0, 5.0])
def test_closed_form_cdf_matches_start_point_quadrature(t, d, folded):
    got = float(lba_cdf(t, d, A, B, S, folded))
    want = _brute_force_cdf(t, d, folded)
    assert got == pytest.approx(want, abs=1e-10), (got, want)


@pytest.mark.parametrize("folded", [True, False])
@pytest.mark.parametrize("d", [0.3, 5.0])
def test_closed_form_pdf_is_the_derivative_of_the_cdf(d, folded):
    t = np.array([0.1, 0.25, 0.4, 0.8, 2.0])
    h = 1e-6
    fd = (lba_cdf(t + h, d, A, B, S, folded) - lba_cdf(t - h, d, A, B, S, folded)) / (
        2 * h
    )
    np.testing.assert_allclose(lba_pdf(t, d, A, B, S, folded), fd, rtol=1e-6, atol=1e-8)


def test_closed_form_cdf_matches_the_engines_own_samples():
    """The closed form describes the engine cssm.lba_vanilla actually runs."""
    rng = np.random.default_rng(99)
    n = 2_000_000
    d = 5.0
    v = np.abs(rng.normal(d, S, n))  # exactly what lba_vanilla draws
    finish = (B - rng.uniform(0.0, A, n)) / v
    for t in (0.2, 0.3, 0.5, 1.0):
        mc = float((finish <= t).mean())
        se = np.sqrt(mc * (1 - mc) / n)
        assert abs(float(lba_cdf(t, d, A, B, S, True)) - mc) < 5 * se + 1e-12


def _condition_drifts(condition: str) -> np.ndarray:
    levels = CHOICE_SETS[condition]
    P = np.array([[p for p, _ in levels]])
    Q = np.array([[q for _, q in levels]])
    pp = PAPER_PARAMS
    return drift_rates(
        P, Q, pp["m"], pp["lambda1"], pp["lambda2"], pp["I0"], pp["beta"]
    )[0]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_race_quadrature_is_converged(condition):
    """Refining the grid, and widening it, must not move the answer."""
    d = _condition_drifts(condition)
    base = choice_probabilities(d, A, B, S, True, 400.0, 400_001)
    finer = choice_probabilities(d, A, B, S, True, 400.0, 1_600_001)
    wider = choice_probabilities(d, A, B, S, True, 1600.0, 1_600_001)

    assert np.abs(base - finer).max() < 1e-12
    assert np.abs(base - wider).max() < 1e-12
    # the probabilities are a proper distribution
    assert base.sum() == pytest.approx(1.0, abs=1e-9)
    # and a deliberately coarse grid is visibly wrong, so the check has teeth
    coarse = choice_probabilities(d, A, B, S, True, 400.0, 4_001)
    assert np.abs(coarse - base).max() > 1e-3


@pytest.mark.parametrize("condition", CONDITIONS)
def test_drift_folding_is_irrelevant_at_the_papers_parameters(condition):
    """cssm folds negative drifts instead of letting them never finish.

    At the paper's parameters the mean drifts are near 5 with s = 1, so a
    negative draw is both rare (P ~ 1e-6) and tiny in magnitude, which makes
    the folded accumulator finish far too late to win.  The two conventions
    therefore agree to machine precision here.
    """
    d = _condition_drifts(condition)
    folded = choice_probabilities(d, A, B, S, True)
    unfolded = choice_probabilities(d, A, B, S, False)
    assert np.abs(folded - unfolded).max() < 1e-12


def test_drift_folding_does_matter_when_drifts_are_small():
    """Sensitivity check for the test above: make negative drifts common."""
    d = np.array([0.25, 0.5, 0.25])
    folded = choice_probabilities(d, 0.3, 0.5, 0.5, True)
    unfolded = choice_probabilities(d, 0.3, 0.5, 0.5, False)
    assert np.abs(folded - unfolded).max() > 0.02

    # Under the unfolded convention the race never terminates when every drift
    # is non-positive, and exactly that much probability goes missing.
    never_finishes = float(np.prod(norm.cdf(-d / 0.5)))
    assert never_finishes > 0.01
    assert unfolded.sum() == pytest.approx(1.0 - never_finishes, abs=1e-3)
    assert folded.sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_simulator_matches_the_closed_form(condition):
    """Monte-Carlo error is the only approximation left in the reported numbers."""
    d = _condition_drifts(condition)
    analytic = choice_probabilities(d, A, B, S, True)

    out = simulate_paper(PAPER_PARAMS, condition, N_SIM, 101)
    choice = out["choice"]
    empirical = np.array([(choice == k + 1).mean() for k in range(d.size)])

    se = np.sqrt(analytic * (1.0 - analytic) / N_SIM)
    deviation = np.abs(empirical - analytic)
    assert np.all(deviation < 4.0 * se + 1e-6), (condition, deviation, se)

    # the reported quantity itself: the relative choice share of X over Y
    rel_a = analytic[0] / (analytic[0] + analytic[1])
    rel_e = empirical[0] / (empirical[0] + empirical[1])
    n_xy = (empirical[0] + empirical[1]) * N_SIM
    se_rel = np.sqrt(rel_a * (1 - rel_a) / n_xy)
    assert abs(rel_e - rel_a) < 4.0 * se_rel, (condition, rel_e, rel_a, se_rel)


def test_monte_carlo_error_shrinks_as_one_over_root_n():
    """The reported quantities are MC estimates; this sizes their error.

    attraction_range_set1 is the most demanding condition reported: it has the
    smallest cell probability (p(decoy) ~ 0.065), so its relative error is the
    largest of the six.
    """
    condition = "attraction_range_set1"
    d = _condition_drifts(condition)
    analytic = choice_probabilities(d, A, B, S, True)
    rel_a = analytic[0] / (analytic[0] + analytic[1])

    sizes = (2_000, 20_000, 200_000)
    errors = {}
    for n in sizes:
        shares = []
        for rep in range(6):
            out = simulate_paper(PAPER_PARAMS, condition, n, 500 + rep)
            choice = out["choice"]
            n_x = int((choice == 1).sum())
            n_y = int((choice == 2).sum())
            shares.append(n_x / (n_x + n_y))
        errors[n] = float(np.sqrt(np.mean((np.array(shares) - rel_a) ** 2)))

    # The error must fall monotonically, and by roughly sqrt(100) = 10 over a
    # hundredfold increase in n.  Six replicates estimate an RMSE to about
    # +-30%, hence the wide band.
    assert errors[2_000] > errors[20_000] > errors[200_000], errors
    ratio = errors[2_000] / errors[200_000]
    assert 3.0 < ratio < 30.0, (ratio, errors)
    # At the largest size the error is already an order of magnitude inside
    # the paper's two-decimal reporting; MLBA_FULL_REPRODUCE=1 uses 500k, finer still.
    assert errors[200_000] < 3e-3, errors
