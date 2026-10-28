"""The registered models reproduce the relative choice shares the paper reports.

Every quantity is a relative choice share p(X) / (p(X) + p(Y)) for the focal
option X = (4, 6) against the non-focal option Y = (6, 4), estimated from the
registered ``mlba3`` / ``mlba2`` simulators at the paper's qualitative-analysis
parameters (m = 5, lambda1 = .2, lambda2 = .4, I0 = 5, A = 1, chi = 2, s = 1).

The default run uses fewer trials than the declared 500k so the suite stays
quick; the Monte-Carlo slack this costs is added to the declared tolerance.
Set ``MLBA_FULL_REPRODUCE=1`` to run it at the declared sample size instead
(~100 s), which is what the numbers quoted in CONTRIBUTION.md come from.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from mlba_reference import PAPER_PARAMS, simulate_paper

# 500k trials per condition puts the Monte-Carlo standard error of a relative
# choice share at ~7e-4, well inside the paper's two-decimal reporting.
FULL_N_TRIALS = 500_000
FULL = os.environ.get("MLBA_FULL_REPRODUCE") == "1"
N_TRIALS = FULL_N_TRIALS if FULL else 100_000
# Monte-Carlo standard error of a relative share at N_TRIALS, times four.
MC_SLACK = 0.0 if FULL else 4 * np.sqrt(0.25 / (N_TRIALS * 0.9))


# quantity id -> paper choice set
_QUANTITY_CONDITION = {
    "rel_share_attraction_range": "attraction_range_set1",
    "rel_share_attraction_range_frequency": "attraction_range_frequency_set1",
    "rel_share_attraction_frequency": "attraction_frequency_set1",
    "rel_share_similarity": "similarity_set1",
    "rel_share_compromise": "compromise_set2",
    "binary_symmetric_share": "binary_xy",
}


def relative_share(condition: str, n: int, seed: int) -> float:
    """p(X) / (p(X) + p(Y)) from a simulated run of one choice set."""
    out = simulate_paper(PAPER_PARAMS, condition, n, seed)
    choice = out["choice"]
    n_x = int((choice == 1).sum())
    n_y = int((choice == 2).sum())
    return n_x / (n_x + n_y)


def quantities(seed: int = 0, n_trials: int = FULL_N_TRIALS) -> dict:
    """Return one finite scalar per declared paper quantity."""
    out = {}
    for offset, (qid, condition) in enumerate(_QUANTITY_CONDITION.items()):
        out[qid] = float(relative_share(condition, n_trials, seed + offset))
    return out


@pytest.fixture(scope="module")
def computed():
    return quantities(seed=0, n_trials=N_TRIALS)


def test_keys_match_the_declaration_exactly(computed, declarations):
    declared = {
        q["id"] for q in declarations["paper_quantities"] if q.get("computed") is True
    }
    assert set(computed) == declared


def test_every_value_is_a_finite_scalar(computed):
    for key, value in computed.items():
        assert isinstance(value, float), (key, type(value))
        assert np.isfinite(value), key
        assert 0.0 <= value <= 1.0, (key, value)


def test_values_match_the_paper(computed, declarations):
    declared = {
        q["id"]: q
        for q in declarations["paper_quantities"]
        if q.get("computed") is True
    }
    failures = []
    for key, value in computed.items():
        spec = declared[key]
        budget = spec["tolerance"] + MC_SLACK
        if abs(value - spec["paper_value"]) > budget:
            failures.append((key, spec["paper_value"], value, budget))
    assert not failures, failures


def test_the_symmetric_binary_prediction_is_exactly_one_half(computed):
    """The paper states Pr[X | {X, Y}] = .5 for symmetric options at any
    parameter values, so this one should sit on .5 up to Monte-Carlo noise."""
    se = np.sqrt(0.25 / N_TRIALS)
    assert abs(computed["binary_symmetric_share"] - 0.5) < 4 * se
