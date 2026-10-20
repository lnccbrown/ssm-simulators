"""The paper-parameterisation helper keeps its contract and agrees with ``MLBA_SPEC``.

``simulate_paper`` runs the registered ``mlba2``/``mlba3`` in the paper's own
parameter names (``chi`` for the threshold) at the choice sets of Figure 7 /
Table 3; the paper-quantity and numerical-accuracy tests rely on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlba_reference import CHOICE_SETS, CONDITIONS, PAPER_PARAMS, simulate_paper


def test_conditions_match_the_declaration(declarations):
    declared = [c["id"] for c in declarations["conditions"]]
    assert CONDITIONS == declared


def test_paper_params_match_the_declaration(declarations):
    assert PAPER_PARAMS == pytest.approx(declarations["paper_parameters"])


def test_condition_levels_match_the_declaration(declarations):
    """The attribute values ``simulate_paper`` simulates are the ones declared."""
    for condition in declarations["conditions"]:
        options = CHOICE_SETS[condition["id"]]
        levels = condition["levels"]
        want = [(levels["X_P"], levels["X_Q"]), (levels["Y_P"], levels["Y_Q"])]
        if "decoy_P" in levels:
            want.append((levels["decoy_P"], levels["decoy_Q"]))
        assert list(options) == [tuple(o) for o in want], condition["id"]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_simulate_returns_the_documented_shape(condition):
    out = simulate_paper(PAPER_PARAMS, condition, 2000, 17)
    assert set(out) == {"rt", "choice"}
    assert out["rt"].shape == (2000,)
    assert out["choice"].shape == (2000,)
    assert out["choice"].dtype.kind == "i"

    n_options = len(CHOICE_SETS[condition])
    assert set(np.unique(out["choice"])) <= set(range(1, n_options + 1))
    # internally controlled stopping: nothing is forced, so no nan RTs
    assert np.all(np.isfinite(out["rt"]))
    assert np.all(out["rt"] > 0)
    # time is in seconds including non-decision time (zero in the paper)
    assert 0.01 < float(np.median(out["rt"])) < 5.0


def test_simulate_is_reproducible_and_seed_dependent():
    a = simulate_paper(PAPER_PARAMS, "similarity_set1", 1000, 3)
    b = simulate_paper(PAPER_PARAMS, "similarity_set1", 1000, 3)
    c = simulate_paper(PAPER_PARAMS, "similarity_set1", 1000, 4)
    np.testing.assert_array_equal(a["rt"], b["rt"])
    np.testing.assert_array_equal(a["choice"], b["choice"])
    assert not np.array_equal(a["rt"], c["rt"])


def test_none_condition_is_accepted_and_unknown_ones_are_not():
    out = simulate_paper(PAPER_PARAMS, None, 100, 1)
    assert out["choice"].size == 100
    with pytest.raises(ValueError):
        simulate_paper(PAPER_PARAMS, "not_a_condition", 10, 1)


def test_non_decision_time_shifts_rt():
    """t is a registered parameter even though the paper fixes it at 0."""
    params = dict(PAPER_PARAMS)
    base = simulate_paper(params, "binary_xy", 20000, 8)
    params["t"] = 0.4
    shifted = simulate_paper(params, "binary_xy", 20000, 8)
    np.testing.assert_allclose(
        np.median(shifted["rt"]) - np.median(base["rt"]), 0.4, atol=1e-5
    )
