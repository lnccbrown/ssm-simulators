"""The front end, against a literal transcription of the paper's equations.

``reference_drifts`` below is written straight from the article, with explicit
loops and the trigonometric form of Appendix C (``theta = arctan(Q/P)``,
Equations C2/C3), rather than the vectorised, ratio-based form used in
``ssms.transforms.simulation.mlba_front_end``.  The two must agree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssms.transforms.simulation.mlba_front_end import drift_rates, subjective_values


def reference_subjective(P: float, Q: float, m: float) -> tuple[float, float]:
    """Equations C1-C3, verbatim, for an option on the line x/a + y/b = 1.

    Objective indifference is additive (``P + Q`` constant), so the x- and
    y-intercepts of the line through the option in the direction
    ``[-1, 1] / sqrt(2)`` are both ``a = b = P + Q``.
    """
    a = b = P + Q
    theta = math.atan(Q / P)
    uP = b / (math.tan(theta) ** m + (b / a) ** m) ** (1.0 / m)
    uQ = a * math.tan(theta) / (1.0 + (a / b) ** m * math.tan(theta) ** m) ** (1.0 / m)
    return uP, uQ


def reference_drifts(options, m, lambda1, lambda2, I0, beta=1.0):
    """Equations 1a-1c, 3, 4 and 5, verbatim, with explicit loops."""
    u = [reference_subjective(p, q, m) for p, q in options]
    n = len(options)
    drifts = []
    for i in range(n):
        d = I0
        for j in range(n):
            if i == j:
                continue
            duP = u[i][0] - u[j][0]
            duQ = u[i][1] - u[j][1]
            lam_P = lambda1 if duP >= 0 else lambda2
            lam_Q = lambda1 if duQ >= 0 else lambda2
            wP = math.exp(-lam_P * abs(duP))
            wQ = math.exp(-lam_Q * beta * abs(duQ))
            d += wP * duP + wQ * duQ
        drifts.append(d)
    return drifts


# Table 3 of the paper, plus the Figure 7 decoys reported in the text.
PAPER_SETS = [
    [(4.0, 6.0), (6.0, 4.0), (3.4, 6.0)],
    [(4.0, 6.0), (6.0, 4.0), (6.0, 3.4)],
    [(4.0, 6.0), (6.0, 4.0), (4.0, 5.4)],
    [(4.0, 6.0), (6.0, 4.0), (5.4, 4.0)],
    [(4.0, 6.0), (6.0, 4.0), (3.7, 5.7)],
    [(4.0, 6.0), (6.0, 4.0), (5.7, 3.7)],
    [(4.0, 6.0), (6.0, 4.0), (3.8, 6.2)],
    [(4.0, 6.0), (6.0, 4.0), (6.2, 3.8)],
    [(4.0, 6.0), (6.0, 4.0), (8.0, 2.0)],
    [(4.0, 6.0), (6.0, 4.0), (2.0, 8.0)],
]


@pytest.mark.parametrize("options", PAPER_SETS)
def test_matches_reference_on_paper_stimuli(options):
    P = np.array([[p for p, _ in options]])
    Q = np.array([[q for _, q in options]])
    got = drift_rates(P, Q, m=5.0, lambda1=0.2, lambda2=0.4, I0=5.0, beta=1.0)[0]
    want = reference_drifts(options, m=5.0, lambda1=0.2, lambda2=0.4, I0=5.0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_matches_reference_on_random_parameters():
    rng = np.random.default_rng(20140001)
    worst = 0.0
    for _ in range(400):
        options = [tuple(rng.uniform(0.5, 9.5, 2)) for _ in range(3)]
        m = float(rng.uniform(0.2, 20.0))
        l1 = float(rng.uniform(0.0, 1.0))
        l2 = float(rng.uniform(0.0, 1.0))
        beta = float(rng.uniform(0.2, 4.0))
        i0 = float(rng.uniform(0.0, 20.0))
        P = np.array([[p for p, _ in options]])
        Q = np.array([[q for _, q in options]])
        got = drift_rates(P, Q, m=m, lambda1=l1, lambda2=l2, I0=i0, beta=beta)[0]
        want = np.array(reference_drifts(options, m, l1, l2, i0, beta))
        worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-9, worst


def test_subjective_values_lie_on_the_curve():
    """(u_P/a)**m + (u_Q/b)**m == 1 with a = b = P + Q (Equation 2)."""
    rng = np.random.default_rng(7)
    P = rng.uniform(0.5, 9.5, (50, 3))
    Q = rng.uniform(0.5, 9.5, (50, 3))
    for m in (0.3, 1.0, 1.68, 5.0, 32.02):
        uP, uQ = subjective_values(P, Q, m)
        a = P + Q
        np.testing.assert_allclose((uP / a) ** m + (uQ / a) ** m, 1.0, rtol=1e-10)
        # and on the ray through the objective point
        np.testing.assert_allclose(uQ / uP, Q / P, rtol=1e-10)


def test_m_equals_one_is_the_identity_map():
    P = np.array([[4.0, 6.0, 3.4]])
    Q = np.array([[6.0, 4.0, 6.0]])
    uP, uQ = subjective_values(P, Q, 1.0)
    np.testing.assert_allclose(uP, P, rtol=1e-12)
    np.testing.assert_allclose(uQ, Q, rtol=1e-12)


def test_objectively_indifferent_options_stay_indifferent():
    """Symmetric options are equally preferred (p. 186: 'the curves also
    preserve equal preference for symmetric options')."""
    P = np.array([[4.0, 6.0]])
    Q = np.array([[6.0, 4.0]])
    for m in (0.5, 1.0, 5.0, 20.0):
        d = drift_rates(P, Q, m=m, lambda1=0.2, lambda2=0.4, I0=5.0)[0]
        assert d[0] == pytest.approx(d[1], abs=1e-12)


def test_identical_options_have_identical_drifts_equal_to_I0():
    """The exact special case the mlba2 -> lba2 reduction rests on."""
    P = np.array([[3.0, 3.0, 3.0]])
    Q = np.array([[7.0, 7.0, 7.0]])
    d = drift_rates(P, Q, m=5.0, lambda1=0.2, lambda2=0.4, I0=1.25)
    assert np.all(d == 1.25)


def test_unweighted_identity_front_end_is_a_sum_of_attribute_totals():
    """With m = 1 and lambda1 = lambda2 = 0, d_i = 3*(P_i+Q_i) - sum_k(P_k+Q_k) + I0.

    This closed form is what the mlba3 -> lba3 reduction setting was solved
    from, so it is worth pinning down separately.
    """
    rng = np.random.default_rng(4)
    P = rng.uniform(0.5, 9.5, (20, 3))
    Q = rng.uniform(0.5, 9.5, (20, 3))
    total = (P + Q).sum(axis=1, keepdims=True)
    want = 3.0 * (P + Q) - total + 0.75
    got = drift_rates(P, Q, m=1.0, lambda1=0.0, lambda2=0.0, I0=0.75)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        subjective_values(np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]]), 2.0)
    with pytest.raises(ValueError):
        subjective_values(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]), 0.0)
