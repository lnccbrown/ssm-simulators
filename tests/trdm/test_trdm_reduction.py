"""Declared reductions, exercised with the public registered simulators.

trdm_3 with the settings from declarations.yaml must have the distribution of
racing_diffusion_3 at the mapped parameters. racing_diffusion_3 is an Euler
scheme whose grid adds a bias of about +7 ms to RTs at its default
delta_t = 1e-3 (audits/reduction_dt_audit.py); the comparison therefore uses
delta_t = 1e-4, where that bias is below the Monte Carlo resolution here.
"""

import numpy as np
import pytest
from scipy import stats

from ssms.basic_simulators.simulator import simulator
from ssms.support_utils.trdm_analytic import Wald, race_probabilities

POINTS = [
    {"v0": 1.0, "v1": 1.0, "v2": 1.0, "A": 0.5, "b": 1.5, "t": 1e-3},
    {"v0": 2.0, "v1": 1.0, "v2": 0.5, "A": 0.3, "b": 1.0, "t": 0.3},
]


def _reduction(declarations):
    (m,) = [m for m in declarations["models"] if m["name"] == "trdm_3"]
    (red,) = m["reductions"]
    assert red["to"] == "racing_diffusion_3" and m["base_model"] == "racing_diffusion_3"
    return red


@pytest.mark.parametrize("lib_theta", POINTS)
def test_trdm_3_timer_off_is_racing_diffusion_3(declarations, lib_theta):
    red = _reduction(declarations)
    ours = {mine: lib_theta[theirs] for mine, theirs in red["map"].items()}
    ours.update(red["settings"])
    n = 30000
    a = simulator(theta=ours, model="trdm_3", n_samples=n, random_state=5)
    b = simulator(
        theta=lib_theta, model=red["to"], n_samples=n, random_state=6, delta_t=1e-4
    )
    rt_a, rt_b = np.asarray(a["rts"]).ravel(), np.asarray(b["rts"]).ravel()
    ch_a, ch_b = np.asarray(a["choices"]).ravel(), np.asarray(b["choices"]).ravel()
    assert stats.ks_2samp(rt_a, rt_b).pvalue > 1e-3
    np.testing.assert_allclose(a["choice_p"], b["choice_p"], atol=0.015)
    for c in range(3):
        assert stats.ks_2samp(rt_a[ch_a == c], rt_b[ch_b == c]).pvalue > 1e-3
    assert abs(rt_a.mean() - rt_b.mean()) < 0.01


def test_trdm_3_timer_off_matches_closed_form_race(declarations):
    """With A = 0 the reduced model is an exact Wald race (no grid at all)."""
    red = _reduction(declarations)
    lib_theta = {"v0": 1.5, "v1": 1.0, "v2": 0.5, "A": 0.0, "b": 1.2, "t": 0.2}
    ours = {mine: lib_theta[theirs] for mine, theirs in red["map"].items()}
    ours.update(red["settings"])
    n = 200_000
    a = simulator(theta=ours, model="trdm_3", n_samples=n, random_state=9)
    ev = [
        Wald(lib_theta[f"v{i}"], 1.0, lib_theta["b"], lib_theta["t"]) for i in range(3)
    ]
    off = Wald(0.0, 1e-9, 1.0, 0.0)  # numerically never finishes
    r = race_probabilities(ev, off)
    np.testing.assert_allclose(a["choice_p"].ravel(), r["evidence"], atol=4e-3)


def test_timer_off_makes_timer_onset_irrelevant(declarations):
    """Settings switch the mechanism off exactly: tT has no effect (sample-identical)."""
    red = _reduction(declarations)
    base = {mine: POINTS[1][theirs] for mine, theirs in red["map"].items()}
    outs = []
    for tT in (0.0, 0.3, 0.9):
        th = dict(base, vT=red["settings"]["vT"], sT=red["settings"]["sT"], tT=tT)
        outs.append(simulator(theta=th, model="trdm_3", n_samples=2000, random_state=4))
    for o in outs[1:]:
        np.testing.assert_array_equal(o["rts"], outs[0]["rts"])
        np.testing.assert_array_equal(o["choices"], outs[0]["choices"])


def test_trdm_2_guess_at_half_is_trdm_2_sample_for_sample():
    th = {
        "v0": 1.2,
        "v1": 0.7,
        "b0": 1.1,
        "b1": 0.9,
        "t": 0.3,
        "vT": 1.3,
        "sT": 0.4,
        "tT": 0.05,
    }
    a = simulator(theta=th, model="trdm_2", n_samples=5000, random_state=21)
    b = simulator(
        theta=dict(th, g=0.5), model="trdm_2_guess", n_samples=5000, random_state=21
    )
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])


def test_guess_probability_controls_timer_responses():
    th = {
        "v0": 0.2,
        "v1": 0.2,
        "b0": 2.0,
        "b1": 2.0,
        "t": 0.3,
        "vT": 3.0,
        "sT": 0.1,
        "tT": 0.0,
    }  # timer nearly always wins
    for g in (0.0, 0.25, 1.0):
        out = simulator(
            theta=dict(th, g=g), model="trdm_2_guess", n_samples=20000, random_state=1
        )
        assert out["choice_p"].ravel()[0] == pytest.approx(g, abs=0.015)
