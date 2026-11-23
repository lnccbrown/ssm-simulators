"""bridge.py and reproduce.py agree with declarations.yaml and with Eqs. 1-3.

Optional paper-reproduction checks. ``bridge.py``, ``reproduce.py`` and
``declarations.yaml`` belong to the original contribution bundle, not to the
library, so this module is skipped unless ``TRDM_BUNDLE_DIR`` points at the
bundle's ``contribution/`` directory, e.g.

    TRDM_BUNDLE_DIR=<bundle>/contribution .venv/bin/python -m pytest tests/trdm/test_trdm_bridge_reproduce.py
"""

import os
import pathlib
import sys

import numpy as np
import pytest

from ssms.support_utils.trdm_analytic import Wald, race_probabilities

_BUNDLE = os.environ.get("TRDM_BUNDLE_DIR")
if not _BUNDLE:
    pytest.skip("TRDM_BUNDLE_DIR not set", allow_module_level=True)
if str(pathlib.Path(_BUNDLE).resolve()) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(_BUNDLE).resolve()))
bridge = pytest.importorskip("bridge")


def test_conditions_match_declarations(bundle_declarations):
    assert bridge.CONDITIONS == [c["id"] for c in bundle_declarations["conditions"]]
    assert set(bridge.PAPER_PARAMS) == set(bundle_declarations["paper_parameters"])
    for cid, vals in bundle_declarations["paper_parameters"].items():
        for k, v in vals.items():
            assert bridge.PAPER_PARAMS[cid][k] == pytest.approx(v), (cid, k)


@pytest.mark.parametrize("cond", bridge.CONDITIONS)
def test_simulate_contract(cond):
    r = bridge.simulate(bridge.PAPER_PARAMS, cond, 3000, 1)
    assert r["rt"].shape == (3000,) and r["choice"].shape == (3000,)
    assert set(np.unique(r["choice"])) <= {1, -1}
    ok = ~np.isnan(r["rt"])
    assert np.all(r["rt"][ok] > 0)
    dl = bridge.PAPER_PARAMS[cond].get("deadline_s")
    if dl is not None:
        assert np.all(r["rt"][ok] < dl)
    if cond.startswith("gonogo"):
        assert np.all(np.isnan(r["rt"][r["choice"] == -1]))
    a = bridge.simulate(None, cond, 500, 7)
    b = bridge.simulate(bridge.PAPER_PARAMS, cond, 500, 7)
    np.testing.assert_array_equal(a["choice"], b["choice"])


@pytest.mark.parametrize(
    "cond", ["perceptual_speed", "lexical_accuracy_NW", "implicit_6.4", "explicit_10"]
)
def test_bridge_conversion_matches_paper_units(cond):
    """Accuracy from the bridge (library units) = Eq. 1 integrated in paper units."""
    p = bridge.PAPER_PARAMS[cond]
    gp = p["gamma_E_mu"] + p["gamma_E_zeta"] / 2
    gm = p["gamma_E_mu"] - p["gamma_E_zeta"] / 2
    a1 = p["alpha_E_mu"] + p["alpha_E_zeta"] / 2
    a2 = p["alpha_E_mu"] - p["alpha_E_zeta"] / 2
    a_corr, a_err = (a2, a1) if cond.endswith("_NW") else (a1, a2)
    ev = [
        Wald(gp, p["sigma_E"], a_corr, p["tau_E"]),
        Wald(gm, p["sigma_E"], a_err, p["tau_E"]),
    ]
    tm = Wald(p["gamma_T"], p["sigma_T"], p["alpha_T"], p["tau_T"])
    upper = p.get("deadline_s", np.inf)
    r = race_probabilities(ev, tm, upper=upper)
    n = 200_000
    sim = bridge.simulate(None, cond, n, 3)
    free = ~np.isnan(sim["rt"])
    p_correct_free = r["evidence"][0] + r["timer"] / 2
    assert np.mean((sim["choice"] == 1) & free) == pytest.approx(
        p_correct_free, abs=5e-3
    )


def test_gonogo_timer_responses_are_go():
    r = bridge.simulate(None, "gonogo_go", 100_000, 2)
    assert np.mean(r["choice"] == 1) > 0.97  # Fig. 11: p(go|go, M) .97-1


def test_reproduce_keys_and_values(bundle_declarations):
    import reproduce

    q = reproduce.quantities(seed=0)
    declared = [
        d["id"] for d in bundle_declarations["paper_quantities"] if d["computed"]
    ]
    assert sorted(q) == sorted(declared)
    for d in bundle_declarations["paper_quantities"]:
        if not d["computed"]:
            continue
        v = q[d["id"]]
        assert isinstance(v, float) and np.isfinite(v)
        if d["status"] == "expected":
            assert abs(v - d["paper_value"]) <= d["tolerance"], d["id"]
        else:
            assert d.get("reason"), d["id"]
