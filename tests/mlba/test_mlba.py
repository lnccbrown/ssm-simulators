"""Tests for the MLBA contribution (Trueblood, Brown & Heathcote, 2014)."""

import numpy as np
import pytest
from scipy.stats import norm

from ssms import Simulator
from ssms.basic_simulators.mlba_analytic import (
    choice_probabilities,
    choice_probabilities_quad,
)
from ssms.basic_simulators.mlba_frontend import mlba_drifts, subjective_values
from ssms.basic_simulators.mlba_units import paper_to_library
from ssms.basic_simulators.simulator import OMISSION_SENTINEL, simulator
from ssms.config import ModelConfigBuilder, get_model_registry

SENTINEL = -999.0
MODELS = ["mlba2", "mlba3"]
FIG7 = {
    "m": 5.0,
    "lambda1": 0.2,
    "lambda2": 0.4,
    "I0": 5.0,
    "A": 1.0,
    "chi": 2.0,
    "s": 1.0,
}
X, Y = (4.0, 6.0), (6.0, 4.0)


class bridge:
    """Paper conditions of Trueblood et al. (2014) run through ``mlba2``/``mlba3``.

    The paper's parameters (Fig. 7 / regularity analysis) are converted to
    library units with ``paper_to_library`` and simulated with the public
    simulator at the default ``sigma_noise``. Conditions are the Table 3 choice
    sets (X, Y, decoy) plus the binary set {X, Y} of p. 192. Choice coding:
    0 = X, 1 = Y, 2 = decoy; omitted trials get ``rt = nan``.
    """

    PAPER_PARAMS = dict(FIG7)
    DECOYS = {
        "attraction_range_set1": (3.4, 6.0),
        "attraction_range_set2": (6.0, 3.4),
        "attraction_frequency_set1": (4.0, 5.4),
        "attraction_frequency_set2": (5.4, 4.0),
        "attraction_range_frequency_set1": (3.7, 5.7),
        "attraction_range_frequency_set2": (5.7, 3.7),
        "similarity_set1": (3.8, 6.2),
        "similarity_set2": (6.2, 3.8),
        "compromise_set1": (8.0, 2.0),
        "compromise_set2": (2.0, 8.0),
    }
    CONDITIONS = [*DECOYS, "binary_xy"]

    @staticmethod
    def options_for(condition):
        if condition is None:
            condition = bridge.CONDITIONS[0]
        if condition == "binary_xy":
            return [X, Y]
        if condition not in bridge.DECOYS:
            raise ValueError(f"unknown condition {condition!r}")
        return [X, Y, bridge.DECOYS[condition]]

    @staticmethod
    def simulate(paper_params, condition, n, seed):
        params = dict(bridge.PAPER_PARAMS)
        params.update(paper_params or {})
        opts = bridge.options_for(condition)
        out = simulator(
            model=f"mlba{len(opts)}",
            theta=paper_to_library(params, opts),
            n_samples=int(n),
            random_state=int(seed),
        )
        rts = np.asarray(out["rts"], dtype=np.float64).reshape(-1)
        choices = np.asarray(out["choices"]).reshape(-1).astype(int)
        rts = np.where(rts == SENTINEL, np.nan, rts)
        return {"rt": rts, "choice": choices}


def defaults(name):
    cfg = ModelConfigBuilder.from_model(name)
    return cfg, dict(zip(cfg["params"], cfg["default_params"]))


# --------------------------------------------------------------- registration
@pytest.mark.parametrize("name", MODELS)
def test_registered_and_valid(name):
    assert get_model_registry().has_model(name)
    cfg, _ = defaults(name)
    ok, errors = ModelConfigBuilder.validate_config(cfg, strict=True)
    assert ok, errors
    lo, hi = cfg["param_bounds"]
    assert cfg["n_params"] == len(cfg["params"]) == len(cfg["default_params"])
    assert all(a < b for a, b in zip(lo, hi))
    assert all(a <= d <= b for a, d, b in zip(lo, cfg["default_params"], hi))
    dl = ModelConfigBuilder.from_model(name + "_deadline")
    assert dl["params"][-1] == "deadline"


@pytest.mark.parametrize("name", MODELS)
def test_defaults_run_and_terminate(name):
    cfg, th = defaults(name)
    out = simulator(model=name, theta=th, n_samples=5000, random_state=3)
    assert np.all(out["rts"] != SENTINEL) and np.all(out["rts"] > 0)
    assert set(np.unique(out["choices"])) == set(cfg["choices"])
    np.testing.assert_allclose(out["choice_p"].sum(), 1.0)
    out2 = Simulator(name).simulate(theta=th, n_samples=500, random_state=3)
    assert out2["rts"].shape[0] == 500


def test_same_seed_same_samples():
    _, th = defaults("mlba3")
    a = simulator(model="mlba3", theta=th, n_samples=2000, random_state=11)
    b = simulator(model="mlba3", theta=th, n_samples=2000, random_state=11)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])


def test_trial_wise_parameters():
    _, th = defaults("mlba3")
    th = {k: np.array([v, v]) for k, v in th.items()}
    th["P2"] = np.array([0.34, 0.6])  # decoy near X, then near Y
    th["Q2"] = np.array([0.6, 0.34])
    out = simulator(model="mlba3", theta=th, n_samples=20000, random_state=1)
    p = out["choice_p"]
    assert p.shape == (2, 3)
    assert p[0, 0] > p[0, 1] and p[1, 1] > p[1, 0]  # attraction effect both ways


def test_b_below_A_raises():
    _, th = defaults("mlba3")
    th.update(A=0.3, b=0.2)
    with pytest.raises(ValueError):
        simulator(model="mlba3", theta=th, n_samples=10, random_state=0)


# ------------------------------------------------------------------ front end
def test_subjective_values_identity_at_m1_and_on_curve():
    P, Q = np.array([2.0, 4.0, 3.3]), np.array([8.0, 6.0, 7.7])
    uP, uQ = subjective_values(P, Q, 1.0)
    np.testing.assert_allclose(uP, P)
    np.testing.assert_allclose(uQ, Q)
    for m in (0.5, 3.0, 32.0):
        uP, uQ = subjective_values(P, Q, m)
        a = P + Q
        np.testing.assert_allclose((uP / a) ** m + (uQ / a) ** m, 1.0, rtol=1e-10)
        np.testing.assert_allclose(uQ / uP, Q / P, rtol=1e-12)  # same ray (Eq. C1)


def test_drifts_by_hand():
    # Eq. 1/3/4 written out for one pair of options with lambda1 != lambda2.
    m, l1, l2, I0 = 2.0, 0.3, 0.7, 1.5
    opts = np.array([[3.0, 6.0], [5.0, 5.0]])
    uP, uQ = subjective_values(opts[:, 0], opts[:, 1], m)

    def w(d):
        return np.exp(-(l1 if d >= 0 else l2) * abs(d))

    dP, dQ = uP[0] - uP[1], uQ[0] - uQ[1]
    V01 = w(dP) * dP + w(dQ) * dQ
    V10 = w(-dP) * -dP + w(-dQ) * -dQ
    d = mlba_drifts(opts[:, 0], opts[:, 1], m, l1, l2, I0)[0]
    np.testing.assert_allclose(d, [V01 + I0, V10 + I0], rtol=1e-12)


def test_beta_one_is_eq4_and_beta_reweights_Q():
    o = np.array([X, Y, (3.4, 6.0)])
    d1 = mlba_drifts(o[:, 0], o[:, 1], 5, 0.2, 0.4, 5)
    d2 = mlba_drifts(o[:, 0], o[:, 1], 5, 0.2, 0.4, 5, beta=1.0)
    np.testing.assert_array_equal(d1, d2)
    d3 = mlba_drifts(o[:, 0], o[:, 1], 5, 0.2, 0.4, 5, beta=2.0)
    assert not np.allclose(d1, d3)


def test_attention_off_at_lambda_zero():
    o = np.array([X, Y, (8.0, 2.0)])
    uP, uQ = subjective_values(o[:, 0], o[:, 1], 3.0)
    d = mlba_drifts(o[:, 0], o[:, 1], 3.0, 0.0, 0.0, 1.0)[0]
    expect = [
        sum((uP[i] - uP[j]) + (uQ[i] - uQ[j]) for j in range(3) if j != i) + 1.0
        for i in range(3)
    ]
    np.testing.assert_allclose(d, expect, rtol=1e-12)


def test_scale_covariance_of_drifts():
    o = np.array([X, Y, (3.8, 6.2)])
    c = 0.1
    d = mlba_drifts(o[:, 0], o[:, 1], 5, 0.2, 0.4, 5)
    dc = mlba_drifts(c * o[:, 0], c * o[:, 1], 5, 0.2 / c, 0.4 / c, 5 * c)
    np.testing.assert_allclose(dc, c * d, rtol=1e-12)


def test_mirror_symmetry_binary_is_half():
    p = choice_probabilities(mlba_drifts([4, 6], [6, 4], 5, 0.2, 0.4, 5)[0])
    np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-12)


# ------------------------------------------------ units: paper <-> library
def test_library_units_equal_paper_units_through_public_simulator():
    """Paper values at sigma_noise = s = 1 and the converted library-unit theta
    at the default sigma_noise give the same samples (same seed)."""
    opts = [X, Y, (3.8, 6.2)]
    lib = paper_to_library(FIG7, opts)
    paper = {
        "m": 5.0,
        "lambda1": 0.2,
        "lambda2": 0.4,
        "I0": 5.0,
        "beta": 1.0,
        "A": 1.0,
        "b": 2.0,
        "t": 0.0,
    }
    for i, (p, q) in enumerate(opts):
        paper[f"P{i}"], paper[f"Q{i}"] = p, q
    a = simulator(model="mlba3", theta=lib, n_samples=20000, random_state=5)
    b = simulator(
        model="mlba3", theta=paper, n_samples=20000, random_state=5, sigma_noise=1.0
    )
    np.testing.assert_array_equal(a["choices"], b["choices"])
    np.testing.assert_allclose(a["rts"], b["rts"], rtol=2e-5)


# ------------------------------------------------ simulator vs analytic LBA
@pytest.mark.parametrize("cond", bridge.CONDITIONS)
def test_bridge_choice_probabilities_match_analytic(cond):
    n = 100_000
    out = bridge.simulate(bridge.PAPER_PARAMS, cond, n, seed=7)
    opts = np.asarray(bridge.options_for(cond))
    p = choice_probabilities(mlba_drifts(opts[:, 0], opts[:, 1], 5, 0.2, 0.4, 5)[0])
    phat = np.bincount(out["choice"], minlength=len(opts)) / n
    z = (phat - p) / np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(z) < 4.5), (phat, p)


def test_negative_drifts_are_not_folded():
    """Table 6 similarity fit: frequent negative rates. Choice probabilities
    follow the standard (non-folded) LBA, unlike the library's lba3."""
    pp = {
        "m": 32.02,
        "lambda1": 0.325,
        "lambda2": 0.897,
        "I0": 0.31,
        "A": 1.0,
        "chi": 2.0,
        "s": 1.0,
    }
    opts = [(3.5, 7.5), (5.5, 5.5), (3.3, 7.7)]
    o = np.asarray(opts)
    d = mlba_drifts(o[:, 0], o[:, 1], 32.02, 0.325, 0.897, 0.31)[0]
    p = choice_probabilities(d)
    n = 200_000
    out = simulator(
        model="mlba3",
        theta=paper_to_library(pp, opts),
        n_samples=n,
        random_state=2,
        max_t=1e6,
    )
    rts = np.asarray(out["rts"]).ravel()
    ch = np.asarray(out["choices"]).ravel()
    phat = np.array([np.mean((ch == i) & (rts != SENTINEL)) for i in range(3)])
    z = (phat - p) / np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(z) < 4.5), (phat, p)
    never = np.prod(norm.cdf(-d))
    om = np.mean(rts == SENTINEL)
    assert abs(om - never) < 4.5 * np.sqrt(never / n) + 1e-5
    # folded-rate library LBA at the same drifts gives a visibly different answer
    th = {"A": 0.1, "b": 0.2, "v0": d[0] * 0.1, "v1": d[1] * 0.1, "v2": d[2] * 0.1}
    lo = simulator(model="lba3", theta=th, n_samples=n, random_state=2)
    pf = np.bincount(np.asarray(lo["choices"]).ravel(), minlength=3) / n
    assert np.max(np.abs(pf - p)) > 0.02


def test_quadrature_matches_adaptive_reference():
    for dec in [(3.4, 6.0), (8.0, 2.0), (3.8, 6.2)]:
        o = np.array([X, Y, dec])
        d = mlba_drifts(o[:, 0], o[:, 1], 5, 0.2, 0.4, 5)[0]
        np.testing.assert_allclose(
            choice_probabilities(d), choice_probabilities_quad(d), atol=1e-10
        )


# ------------------------------------------- family relation with lba3
def test_identical_options_match_lba3_when_rates_stay_positive():
    """Identical options make every mean drift I0 (V_ij = 0); with I0 >> sd the
    folding in lba3 is immaterial and the two races agree in distribution."""
    n = 100_000
    th = {
        "m": 3.0,
        "lambda1": 2.0,
        "lambda2": 4.0,
        "I0": 0.5,
        "beta": 1.0,
        "A": 0.1,
        "b": 0.2,
        "t": 0.0,
    }
    for i in range(3):
        th[f"P{i}"], th[f"Q{i}"] = 0.5, 0.5
    a = simulator(model="mlba3", theta=th, n_samples=n, random_state=1)
    b = simulator(
        model="lba3",
        theta={"A": 0.1, "b": 0.2, "v0": 0.5, "v1": 0.5, "v2": 0.5},
        n_samples=n,
        random_state=2,
    )
    np.testing.assert_allclose(a["choice_p"], b["choice_p"], atol=0.01)
    qa = np.quantile(a["rts"], [0.1, 0.5, 0.9])
    qb = np.quantile(b["rts"], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(qa, qb, rtol=0.01)


# --------------------------------------------------- omission contract
@pytest.mark.parametrize("name", MODELS)
def test_forced_omissions_deadline(name):
    cfg, th = defaults(name + "_deadline")
    th = dict(th, deadline=0.18)  # median decision time is ~0.19
    out = simulator(model=name + "_deadline", theta=th, n_samples=20000, random_state=0)
    rts = np.asarray(out["rts"]).ravel()
    ch = np.asarray(out["choices"]).ravel()
    om = rts == SENTINEL
    assert 0.2 < om.mean() < 0.9
    assert not np.any(ch == SENTINEL)
    assert set(np.unique(ch[om])) <= set(out["metadata"]["possible_choices"])
    assert np.all(rts[~om] < 0.18)
    cp = out["choice_p"]
    assert np.all(np.isfinite(cp)) and cp.shape[-1] == len(cfg["choices"])
    np.testing.assert_allclose(cp.sum(), 1.0)
    np.testing.assert_allclose(out["omission_p"], om.mean())


def test_forced_omissions_max_t_and_never_finishing():
    _, th = defaults("mlba3")
    out = simulator(
        model="mlba3", theta=th, n_samples=20000, random_state=0, max_t=0.15
    )
    rts = np.asarray(out["rts"]).ravel()
    ch = np.asarray(out["choices"]).ravel()
    assert 0.2 < np.mean(rts == SENTINEL) < 1.0
    assert set(np.unique(ch)) <= {0, 1, 2}
    np.testing.assert_allclose(out["choice_p"].sum(), 1.0)
    # every mean drift strongly negative: almost no trial ever finishes
    th2 = dict(th, I0=0.0, lambda1=0.0, lambda2=0.0, b=1.0, A=0.0)
    for i in range(3):
        th2[f"P{i}"], th2[f"Q{i}"] = 0.5, 0.5
    out2 = simulator(
        model="mlba3", theta=th2, n_samples=20000, random_state=1, sigma_noise=0.01
    )
    rts2 = np.asarray(out2["rts"]).ravel()
    ch2 = np.asarray(out2["choices"]).ravel()
    assert np.mean(rts2 == SENTINEL) > 0.4  # rate N(0, .01) -> half the races never end
    assert set(np.unique(ch2)) <= {0, 1, 2}
    np.testing.assert_allclose(out2["choice_p"].sum(), 1.0)


# ------------------------------------------------------------- bridge
def test_bridge_contract():
    assert len(bridge.CONDITIONS) == len(set(bridge.CONDITIONS))
    out = bridge.simulate(bridge.PAPER_PARAMS, "compromise_set2", 1000, seed=0)
    assert out["rt"].shape == out["choice"].shape == (1000,)
    assert set(np.unique(out["choice"])) <= {0, 1, 2}
    b = bridge.simulate(bridge.PAPER_PARAMS, "binary_xy", 1000, seed=0)
    assert set(np.unique(b["choice"])) <= {0, 1}


# ------------------------------------- omission-label contract (Python race)
@pytest.mark.parametrize("name", MODELS)
def test_mlba_marks_omissions_in_rts_not_choices_and_labels_agree(name):
    """MLBA case of ``tests/test_omission_labels.py``: the MLBA censors in Python,
    so it must itself keep the sentinel out of ``choices`` and the pipeline's
    auxiliary labels must give ``choice_p`` summing to one."""
    from ssms.dataset_generators.pipelines.simulation_pipeline import (
        SimulationPipeline,
    )

    cfg, th = defaults(name + "_deadline")
    th = dict(th, deadline=0.18)
    out = simulator(model=name + "_deadline", theta=th, n_samples=8000, random_state=0)
    choices = np.asarray(out["choices"]).ravel()
    rts = np.asarray(out["rts"]).ravel()
    assert OMISSION_SENTINEL == SENTINEL
    assert (rts == OMISSION_SENTINEL).any(), "no omissions: lower the deadline"
    assert not (choices == OMISSION_SENTINEL).any()

    labels = SimulationPipeline._compute_auxiliary_labels(
        None,
        {
            "choices": choices,
            "rts": rts,
            "metadata": {"possible_choices": list(cfg["choices"])},
        },
    )
    expected = float((rts == OMISSION_SENTINEL).mean())
    assert expected > 0.05
    assert labels["omission_p"][0, 0] == pytest.approx(expected, abs=1e-9)
    assert labels["choice_p"][0].sum() == pytest.approx(1.0, abs=1e-9)
