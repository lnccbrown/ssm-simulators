"""Tests for the Multiple-Threshold Race (``mtr``) model.

Run from the repository root:
    .venv/bin/python -B -m pytest tests/test_mtr.py -q -o addopts= -p no:cacheprovider
"""

import numpy as np
import pytest
from scipy import stats

from ssms import Simulator
from ssms.basic_simulators.mtr import (
    DONT_KNOW,
    OMISSION_SENTINEL,
    multiple_threshold_race,
)
from ssms.basic_simulators.simulator import simulator as legacy_simulator
from ssms.config import ModelConfigBuilder, get_model_registry
from ssms.config._modelconfig import get_model_config, get_mtr_config
from ssms.config._modelconfig.validation import (
    get_invalid_configs,
    get_invalid_ndt_edge_shift_configs,
)
from ssms.dataset_generators.pipelines.simulation_pipeline import (
    SimulationPipeline,
)
from tests import mtr_analytic as analytic

DK = DONT_KNOW
SENT = OMISSION_SENTINEL


@pytest.fixture
def theta():
    cfg = ModelConfigBuilder.from_model("mtr")
    return dict(zip(cfg["params"], cfg["default_params"]))


def _flat(out):
    return np.asarray(out["choices"]).ravel(), np.asarray(out["rts"]).ravel()


# --------------------------------------------------------------- registration
def test_registered_and_passes_static_validation():
    assert get_model_registry().has_model("mtr")
    cfg = get_mtr_config()
    # The checks the library runs on its own registry at import time.
    assert get_invalid_configs({"mtr": cfg}) == []
    assert get_invalid_ndt_edge_shift_configs({"mtr": cfg}) == []
    lo, hi = cfg["param_bounds"]
    assert len(lo) == len(hi) == len(cfg["params"]) == cfg["n_params"]
    assert all(a <= d <= b for a, d, b in zip(lo, cfg["default_params"], hi))
    dcfg = ModelConfigBuilder.from_model("mtr_deadline")
    assert dcfg["params"][-1] == "deadline"
    assert get_invalid_configs({"mtr_deadline": dcfg}) == []


def test_statically_registered():
    # ``mtr`` comes from the static ``get_model_config()`` table, not from
    # runtime registration glue.
    cfg = get_model_config()["mtr"]
    assert cfg["simulator"] is multiple_threshold_race
    assert cfg["params"] == get_mtr_config()["params"]
    assert get_model_registry().has_model("mtr")


# ------------------------------------------------------------------ basics
def test_runs_shapes_and_seed(theta):
    sim = Simulator("mtr")
    a = sim.simulate(theta=theta, n_samples=500, random_state=3)
    b = sim.simulate(theta=theta, n_samples=500, random_state=3)
    assert a["rts"].shape == (500, 1) and a["choices"].shape == (500, 1)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])
    assert set(np.unique(a["choices"])) <= {0, 1, 2}
    assert a["metadata"]["possible_choices"] == [0, 1, 2]
    assert np.all(a["rts"] > theta["t"])


def test_trialwise_parameters_via_legacy_simulator(theta):
    cfg = ModelConfigBuilder.from_model("mtr")
    mat = np.tile(np.array([theta[p] for p in cfg["params"]]), (300, 1))
    mat[:150, cfg["params"].index("D0")] = 0.0  # every loss of acc 0 -> DK
    out = legacy_simulator(theta=mat, model="mtr", n_samples=1, random_state=0)
    ch = np.asarray(out["choices"]).ravel()
    assert ch.shape == (300,)
    assert not np.any(ch[:150] == 1)  # acc 1 can never win definitively there


def test_invalid_parameters_raise(theta):
    for bad in ({"B0": 0.0}, {"D1": 1.2}, {"sv0": 0.0}, {"A": -0.1}):
        th = {**theta, **bad}
        with pytest.raises(ValueError):
            multiple_threshold_race(**th, n_samples=10)


# ------------------------------------------------------- omission contract
def test_forced_omission_contract_deadline(theta):
    th = {**theta, "deadline": 0.9}  # well inside the RT distribution
    out = Simulator("mtr_deadline").simulate(theta=th, n_samples=20000, random_state=1)
    ch, rts = _flat(out)
    omitted = rts == SENT
    assert 0.2 < omitted.mean() < 0.95
    assert not np.any(ch == SENT)
    assert set(np.unique(ch[omitted])) <= set(out["metadata"]["possible_choices"])
    assert np.all(rts[~omitted] < 0.9)
    labels = SimulationPipeline._compute_auxiliary_labels(
        None, {"choices": ch, "rts": rts, "metadata": out["metadata"]}
    )
    cp = labels["choice_p"]
    assert np.all(np.isfinite(cp)) and np.isclose(cp.sum(), 1.0)
    assert np.isclose(labels["omission_p"][0, 0], omitted.mean())
    # censoring does not change the latent responses (same seed, no deadline)
    ref = Simulator("mtr").simulate(theta=theta, n_samples=20000, random_state=1)
    np.testing.assert_array_equal(np.asarray(ref["choices"]).ravel(), ch)


def test_forced_omission_contract_max_t(theta):
    th = {**theta, "B0": 5.0, "B1": 5.0}
    out = Simulator("mtr").simulate(
        theta=th, n_samples=20000, max_t=1.2, random_state=2
    )
    ch, rts = _flat(out)
    omitted = rts == SENT
    assert omitted.mean() > 0.2
    assert set(np.unique(ch)) <= {0, 1, 2}
    labels = SimulationPipeline._compute_auxiliary_labels(
        None, {"choices": ch, "rts": rts, "metadata": out["metadata"]}
    )
    assert np.isclose(labels["choice_p"].sum(), 1.0)


def test_deadline_omission_rate_matches_reference(theta):
    th = {**theta, "deadline": 1.0}
    out = Simulator("mtr_deadline").simulate(theta=th, n_samples=200000, random_state=5)
    om = np.mean(np.asarray(out["rts"]).ravel() == SENT)
    ref = analytic.response_probabilities(theta, upper=1.0 - theta["t"])
    p_resp = ref[0] + ref[1] + ref[2]
    se = np.sqrt(p_resp * (1 - p_resp) / 200000)
    assert abs(om - (1 - p_resp)) < 4.5 * se


# ----------------------------------------------- mechanism and limit cases
def test_D_one_switches_dontknow_off_and_leaves_race_unchanged(theta):
    """D0 = D1 = 1 (DK = 0): the MTR is the plain two-accumulator truncated-normal
    LBA race. Same seed -> identical RTs for any D, and responses = race winner."""
    sim = Simulator("mtr")
    off = sim.simulate(
        theta={**theta, "D0": 1.0, "D1": 1.0}, n_samples=50000, random_state=9
    )
    on = sim.simulate(theta=theta, n_samples=50000, random_state=9)
    np.testing.assert_array_equal(off["rts"], on["rts"])
    assert not np.any(off["choices"] == DK)
    lat = multiple_threshold_race(
        **theta, n_samples=50000, random_state=9, return_latent=True
    )
    np.testing.assert_array_equal(
        np.asarray(off["choices"]).ravel(), lat["winners"].ravel()
    )
    defin = np.asarray(on["choices"]).ravel() != DK
    # a definitive response is always the race winner's
    np.testing.assert_array_equal(
        np.asarray(on["choices"]).ravel()[defin], lat["winners"].ravel()[defin]
    )


def test_D_zero_makes_every_response_dontknow(theta):
    out = Simulator("mtr").simulate(
        theta={**theta, "D0": 0.0, "D1": 0.0}, n_samples=20000, random_state=4
    )
    assert np.all(np.asarray(out["choices"]) == DK)


def test_dontknow_probability_increases_with_DK(theta):
    ps = [
        analytic.response_probabilities({**theta, "D0": 1 - x, "D1": 1 - x})[2]
        for x in (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    ]
    assert ps[0] == pytest.approx(0.0, abs=1e-12)
    assert ps[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(ps) > 0)


def test_asymmetric_D_targets_the_losing_accumulator(theta):
    """Lowering only D1 adds don't-knows only when accumulator 0 wins."""
    th = {**theta, "D0": 1.0, "D1": 0.6}
    lat = multiple_threshold_race(
        **th, n_samples=50000, random_state=0, return_latent=True
    )
    ch, w = lat["choices"].ravel(), lat["winners"].ravel()
    assert np.all(ch[w == 1] == 1)
    assert np.any(ch[w == 0] == DK)


# -------------------------------------------- simulator vs analytic reference
CASES = [
    {},
    {"A": 0.0},
    {"A": 2.5, "B0": 0.3, "B1": 0.5, "D0": 0.1, "D1": 0.3},  # d < A possible
    {"v0": 0.0, "v1": 0.5, "sv0": 0.3, "sv1": 2.0},  # heavy truncation
    {"v0": 6.0, "v1": 1.0, "D0": 0.95, "D1": 0.5, "t": 0.0},
]


@pytest.mark.parametrize("override", CASES)
def test_response_probabilities_match_reference(theta, override):
    th = {**theta, **override}
    ref = analytic.response_probabilities(th)
    assert abs(ref["residual"]) < 1e-8
    n = 200000
    ch = np.asarray(
        Simulator("mtr").simulate(theta=th, n_samples=n, random_state=11)["choices"]
    ).ravel()
    for k in range(3):
        se = np.sqrt(max(ref[k] * (1 - ref[k]), 1e-12) / n)
        assert abs(np.mean(ch == k) - ref[k]) < 4.5 * se + 1e-9, (k, override)


def test_rt_distribution_matches_reference(theta):
    """KS test of decision times per response against the analytic CDF."""
    n = 100000
    out = Simulator("mtr").simulate(theta=theta, n_samples=n, random_state=21)
    ch, rts = _flat(out)
    probs = analytic.response_probabilities(theta)
    for k in range(3):
        dt = np.sort(rts[ch == k] - theta["t"])

        def cdf(x, k=k):
            x = np.atleast_1d(x)
            return (
                np.array(
                    [
                        analytic.response_probabilities(
                            theta, upper=xi, epsabs=1e-10, epsrel=1e-8
                        )[k]
                        for xi in x
                    ]
                )
                / probs[k]
            )

        grid = np.quantile(dt, np.linspace(0.02, 0.98, 25))
        emp = np.searchsorted(dt, grid, side="right") / dt.size
        assert np.max(np.abs(emp - cdf(grid))) < 1.63 * 1.5 / np.sqrt(dt.size) + 2e-3


def test_conditional_mean_rt_matches_reference(theta):
    out = Simulator("mtr").simulate(theta=theta, n_samples=300000, random_state=8)
    ch, rts = _flat(out)
    dt = rts - theta["t"]
    ref = analytic.mean_decision_times(theta, upper=2.0)
    for k in range(3):
        x = dt[(ch == k) & (dt < 2.0)]
        assert abs(x.mean() - ref[k]) < 4.5 * x.std() / np.sqrt(x.size)


# ------------------------------------- family relation to lba2 (not a reduction)
def test_family_lba2_close_when_truncation_is_negligible():
    """``lba2`` folds negative rates (|N(v, sd)|, sd = 0.1, t = 0); the MTR
    truncates them. With sv = 0.1 and v >= 0.5 the two rate laws differ by
    < 3e-7 in mass, so at D = 1 the races agree in distribution to that order.
    This is a family check, NOT a declared reduction: the laws differ in general."""
    np.random.seed(123)  # lba2's kernel draws from NumPy's global RNG
    lba = Simulator("lba2").simulate(
        theta={"A": 0.3, "b": 0.6, "v0": 0.8, "v1": 0.5},
        n_samples=50000,
        random_state=1,
    )
    mtr = Simulator("mtr").simulate(
        theta={
            "A": 0.3,
            "B0": 0.3,
            "B1": 0.3,
            "D0": 1.0,
            "D1": 1.0,
            "v0": 0.8,
            "v1": 0.5,
            "sv0": 0.1,
            "sv1": 0.1,
            "t": 0.0,
        },
        n_samples=50000,
        random_state=1,
    )
    lc, lr = _flat(lba)
    mc, mr = _flat(mtr)
    assert abs(np.mean(lc == 0) - np.mean(mc == 0)) < 0.01
    for k in (0, 1):
        assert stats.ks_2samp(lr[lc == k], mr[mc == k]).pvalue > 1e-4


# ------------------------------------------- omission labels (legacy path)
def test_legacy_deadline_omission_labels():
    """``mtr`` is simulated outside ``src/cssm``, so it needs its own case of
    the omission-label contract (``tests/test_omission_labels.py``): the
    sentinel goes in ``rts`` only and ``choice_p`` sums to one."""
    cfg = ModelConfigBuilder.from_model("mtr_deadline")
    theta = np.array(cfg["default_params"], dtype=float)
    theta[cfg["params"].index("deadline")] = 0.9
    out = legacy_simulator(
        theta=np.tile(theta, (8000, 1)),
        model="mtr_deadline",
        n_samples=1,
        random_state=0,
    )
    choices = np.asarray(out["choices"]).ravel()
    rts = np.asarray(out["rts"]).ravel()
    assert (rts == OMISSION_SENTINEL).any()
    assert not (choices == OMISSION_SENTINEL).any()
    labels = SimulationPipeline._compute_auxiliary_labels(
        None,
        {
            "choices": choices,
            "rts": rts,
            "metadata": {"possible_choices": [0, 1, DONT_KNOW]},
        },
    )
    assert labels["omission_p"][0, 0] == pytest.approx(
        float((rts == OMISSION_SENTINEL).mean()), abs=1e-9
    )
    assert labels["choice_p"][0].sum() == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < labels["omission_p"][0, 0] < 1.0
