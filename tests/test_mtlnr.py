"""Tests for the multiple-threshold correlated Lognormal race (MTLNR).

Reynolds, Kvam, Osth & Heathcote (2020). ``mtlnr2_conf2`` / ``mtlnr2_conf3``
are registered statically in ``ssms/config/_modelconfig/__init__.py``; the
simulator is ``ssms.basic_simulators.mtlnr``. The analytic reference (paper
Eqs. 4, 6-7 and the SDT representation) is test support in
``tests/mtlnr_reference.py``.

The declared parameters, transforms and reductions of the contribution are
inlined below (``DECLARED``) so the exact declared settings and maps are
exercised through the public ``simulator``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ssms import OMISSION_SENTINEL, Simulator
from ssms.basic_simulators.mtlnr import decode_choices
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder, get_model_registry
from ssms.config._modelconfig import get_model_config
from ssms.config._modelconfig.validation import (
    get_invalid_configs,
    get_invalid_ndt_edge_shift_configs,
)
from tests import mtlnr_reference as R

_SWAP = "ssms.transforms.sampling.swap.SwapIfLessConstraint"
_BASE = [
    ("mu0", [-3.0, 2.0], -1.0),
    ("mu1", [-3.0, 2.0], -0.5),
    ("sigma0", [0.1, 2.0], 0.5),
    ("sigma1", [0.1, 2.0], 0.7),
    ("rho", [-0.999, 0.999], 0.0),
]
_RACE_MAP = {k: k for k in ["mu0", "mu1", "sigma0", "sigma1", "rho", "t"]}
_IND_MAP = {k: k for k in ["mu0", "mu1", "sigma0", "sigma1", "t"]}
_ZERO3 = {"d1acc0": 0.0, "d2acc0": 0.0, "d1acc1": 0.0, "d2acc1": 0.0}
_ZERO2 = {"d1acc0": 0.0, "d1acc1": 0.0}


def _params(thresholds):
    return [
        {"name": n, "bounds": b, "default": d}
        for n, b, d in [*_BASE, *thresholds, ("t", [0.0, 2.0], 0.3)]
    ]


DECLARED = {
    "models": [
        {
            "name": "mtlnr2_conf3",
            "parameters": _params(
                [
                    ("d1acc0", [0.0, 1.0], 0.5),
                    ("d2acc0", [0.0, 1.0], 0.8),
                    ("d1acc1", [0.0, 1.0], 0.5),
                    ("d2acc1", [0.0, 1.0], 0.8),
                ]
            ),
            "parameter_transforms": {
                "sampling": [
                    {"type": _SWAP, "param_a": "d2acc0", "param_b": "d1acc0"},
                    {"type": _SWAP, "param_a": "d2acc1", "param_b": "d1acc1"},
                ],
                "simulation": [],
            },
            "reductions": [
                {
                    "to": "lnr2_corr",
                    "settings": dict(_ZERO3),
                    "map": _RACE_MAP,
                    "expect": "identical",
                },
                {
                    "to": "lnr2",
                    "settings": {"rho": 0.0, **_ZERO3},
                    "map": _IND_MAP,
                    "expect": "identical",
                },
            ],
        },
        {
            "name": "mtlnr2_conf2",
            "parameters": _params(
                [("d1acc0", [0.0, 1.0], 0.5), ("d1acc1", [0.0, 1.0], 0.5)]
            ),
            "parameter_transforms": {"sampling": [], "simulation": []},
            "reductions": [
                {
                    "to": "lnr2_corr",
                    "settings": dict(_ZERO2),
                    "map": _RACE_MAP,
                    "expect": "identical",
                },
                {
                    "to": "lnr2",
                    "settings": {"rho": 0.0, **_ZERO2},
                    "map": _IND_MAP,
                    "expect": "identical",
                },
            ],
        },
    ],
}

MODELS = [m["name"] for m in DECLARED["models"]]
REDUCTIONS = [(m["name"], r) for m in DECLARED["models"] for r in m["reductions"]]

# Figure 7 of the paper: group parameters, accumulator 0 = "new", 1 = "old".
PAPER_PARAMS = {"mu_old": -0.95, "mu_new": -0.63, "sigma": 0.55, "d1": 0.3, "d2": 0.6}
CONDITION_RHO = {"neg_corr": -0.9, "independent": 0.0, "pos_corr": 0.9}


def _figure7_theta(condition):
    p = PAPER_PARAMS
    return {
        "mu0": p["mu_new"],
        "mu1": p["mu_old"],
        "sigma0": p["sigma"],
        "sigma1": p["sigma"],
        "rho": CONDITION_RHO[condition],
        "d1acc0": p["d1"],
        "d2acc0": p["d2"],
        "d1acc1": p["d1"],
        "d2acc1": p["d2"],
        "t": 0.0,
    }


THETA3 = {
    "mu0": -0.6,
    "mu1": -0.9,
    "sigma0": 0.55,
    "sigma1": 0.7,
    "rho": 0.5,
    "d1acc0": 0.35,
    "d2acc0": 0.7,
    "d1acc1": 0.45,
    "d2acc1": 0.85,
    "t": 0.25,
}
THETA2 = {
    k: THETA3[k]
    for k in ["mu0", "mu1", "sigma0", "sigma1", "rho", "d1acc0", "d1acc1", "t"]
}
THETA = {"mtlnr2_conf3": THETA3, "mtlnr2_conf2": THETA2}


def _d(theta):
    K = sum(1 for k in theta if k.endswith("acc0"))
    return [[theta[f"d{k}acc{i}"] for k in range(1, K + 1)] for i in range(2)]


# --------------------------------------------------------------------------
# Registration and the library's static validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_registered_and_valid(name):
    assert get_model_registry().has_model(name)
    cfg = ModelConfigBuilder.from_model(name)
    # The library's own static validation (run on import for built-ins).
    assert get_invalid_configs({name: cfg}) == []
    assert get_invalid_ndt_edge_shift_configs({name: cfg}) == []
    ok, errors = ModelConfigBuilder.validate_config(cfg, strict=True)
    assert ok, errors
    lo, hi = cfg["param_bounds"]
    assert all(a <= d <= b for a, d, b in zip(lo, cfg["default_params"], hi))


@pytest.mark.parametrize("decl", DECLARED["models"], ids=MODELS)
def test_declared_parameters_match_config(decl):
    cfg = ModelConfigBuilder.from_model(decl["name"])
    assert cfg["params"] == [p["name"] for p in decl["parameters"]]
    lo, hi = cfg["param_bounds"]
    for i, p in enumerate(decl["parameters"]):
        assert [lo[i], hi[i]] == p["bounds"]
        assert cfg["default_params"][i] == p["default"]
    tr = cfg["parameter_transforms"]
    declared = decl["parameter_transforms"]
    assert len(tr["simulation"]) == len(declared["simulation"]) == 0
    assert len(tr["sampling"]) == len(declared["sampling"])
    for obj, spec in zip(tr["sampling"], declared["sampling"]):
        assert f"{type(obj).__module__}.{type(obj).__name__}" == spec["type"]
        assert (obj.param_a, obj.param_b) == (spec["param_a"], spec["param_b"])


def test_registered_statically():
    """Both configs come from the static config table, deadline variants too."""
    configs = get_model_config()
    for name in MODELS:
        assert name in configs
        assert get_model_registry().has_model(name)
        cfg = ModelConfigBuilder.from_model(f"{name}_deadline")
        assert cfg["params"][-1] == "deadline"


# --------------------------------------------------------------------------
# Reductions (declared): identical samples through the public simulator
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,red", REDUCTIONS, ids=[f"{n}->{r['to']}" for n, r in REDUCTIONS]
)
@pytest.mark.parametrize("seed", [0, 17])
def test_declared_reduction(name, red, seed):
    theta = {**THETA[name], **red["settings"]}
    target = {red["map"][k]: v for k, v in theta.items() if k in red["map"]}
    a = simulator(model=name, theta=theta, n_samples=5000, random_state=seed)
    b = simulator(model=red["to"], theta=target, n_samples=5000, random_state=seed)
    assert red["expect"] == "identical"
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])


def test_marginal_choice_rt_equal_lnr2_corr_for_any_thresholds():
    """Confidence never changes the race: label % 2 and rts equal lnr2_corr's."""
    a = simulator(model="mtlnr2_conf3", theta=THETA3, n_samples=5000, random_state=4)
    th = {k: v for k, v in THETA3.items() if not k.startswith("d")}
    b = simulator(model="lnr2_corr", theta=th, n_samples=5000, random_state=4)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"] % 2, b["choices"])
    assert len(np.unique(a["choices"] // 2)) == 3


def test_conf3_with_zero_lower_threshold_is_conf2():
    th3 = {**THETA3, "d1acc0": 0.0, "d1acc1": 0.0}
    th2 = {**THETA2, "d1acc0": THETA3["d2acc0"], "d1acc1": THETA3["d2acc1"]}
    a = simulator(model="mtlnr2_conf3", theta=th3, n_samples=5000, random_state=2)
    b = simulator(model="mtlnr2_conf2", theta=th2, n_samples=5000, random_state=2)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])


def test_threshold_order_invariant_and_sampling_transform_orders():
    swapped = {**THETA3, "d1acc0": THETA3["d2acc0"], "d2acc0": THETA3["d1acc0"]}
    a = simulator(model="mtlnr2_conf3", theta=THETA3, n_samples=3000, random_state=1)
    b = simulator(model="mtlnr2_conf3", theta=swapped, n_samples=3000, random_state=1)
    np.testing.assert_array_equal(a["choices"], b["choices"])
    cfg = ModelConfigBuilder.from_model("mtlnr2_conf3")
    th = {k: np.array([0.9, 0.2]) for k in ["d1acc0", "d2acc0", "d1acc1", "d2acc1"]}
    th["d2acc0"] = np.array([0.1, 0.6])
    for tr in cfg["parameter_transforms"]["sampling"]:
        th = tr.apply(th)
    assert np.all(th["d2acc0"] > th["d1acc0"])


# --------------------------------------------------------------------------
# Correctness against the paper's equations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_label_probabilities_match_closed_form(name):
    th = THETA[name]
    mu, sg, d = [th["mu0"], th["mu1"]], [th["sigma0"], th["sigma1"]], _d(th)
    p_cf = R.label_probs_sdt(mu, sg, th["rho"], d)
    p_q = np.array(
        [
            R.joint_prob_quad(l % 2, l // 2, mu, sg, th["rho"], d)[0]
            for l in range(len(p_cf))
        ]
    )
    np.testing.assert_allclose(p_q, p_cf, atol=1e-10)  # Eqs. 6-7 == SDT form
    n = 200_000
    out = simulator(model=name, theta=th, n_samples=n, random_state=11)
    counts = np.bincount(out["choices"].ravel(), minlength=len(p_cf))
    _chi2, p = stats.chisquare(counts, p_cf * n)
    assert p > 1e-3, (counts / n, p_cf)
    np.testing.assert_allclose(out["choice_p"][0], counts / n)


def test_rt_distribution_per_confidence_matches_eqs_6_7():
    th = THETA3
    mu, sg, d = [th["mu0"], th["mu1"]], [th["sigma0"], th["sigma1"]], _d(th)
    out = simulator(model="mtlnr2_conf3", theta=th, n_samples=200_000, random_state=5)
    lab = out["choices"].ravel()
    dt = out["rts"].ravel() - th["t"]
    for l in range(6):
        x = dt[lab == l]
        if x.size < 2000:
            continue
        qs = np.quantile(x, [0.1, 0.5, 0.9])
        F = [R.conditional_cdf(q, l % 2, l // 2, mu, sg, th["rho"], d) for q in qs]
        se = np.sqrt(np.array([0.09, 0.25, 0.09]) / x.size)
        assert np.all(np.abs(np.array(F) - [0.1, 0.5, 0.9]) < 5 * se), (l, F)


def test_figure7_qualitative_claims():
    """Neg corr -> higher-confidence responses more frequent (paper p. 12)."""
    hi = {}
    for c in CONDITION_RHO:
        out = simulator(
            model="mtlnr2_conf3",
            theta=_figure7_theta(c),
            n_samples=100_000,
            random_state=3,
        )
        _, conf = decode_choices(out["choices"].ravel())
        hi[c] = (conf == 2).mean()
    assert hi["neg_corr"] > hi["independent"] > hi["pos_corr"]


# --------------------------------------------------------------------------
# Omission contract (max_t and the _deadline variant)
# --------------------------------------------------------------------------


def _check_omission_contract(out, n):
    rts, ch = out["rts"], out["choices"]
    omitted = rts == OMISSION_SENTINEL
    assert omitted.any()
    possible = out["metadata"]["possible_choices"]
    assert not np.any(ch == OMISSION_SENTINEL)
    assert np.all(np.isin(ch, possible))
    cp = out["choice_p"]
    assert np.all(np.isfinite(cp))
    np.testing.assert_allclose(cp.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(out["omission_p"][0, 0], omitted.mean(), atol=1e-12)
    return omitted


@pytest.mark.parametrize("name", MODELS)
def test_forced_omission_max_t(name):
    th = {**THETA[name], "mu0": 1.5, "mu1": 1.5, "sigma0": 1.0, "sigma1": 1.0}
    out = simulator(model=name, theta=th, n_samples=4000, max_t=2.0, random_state=5)
    omitted = _check_omission_contract(out, 4000)
    # latent confidence is kept for censored trials, several levels observed
    assert len(np.unique(out["choices"][omitted])) > 2


@pytest.mark.parametrize("name", MODELS)
def test_forced_omission_deadline_variant(name):
    th = {**THETA[name], "deadline": 0.6}
    out = simulator(model=f"{name}_deadline", theta=th, n_samples=4000, random_state=6)
    omitted = _check_omission_contract(out, 4000)
    assert np.all(out["rts"][~omitted] < 0.6)
    # all-forced case: every trial censored, labels still valid
    th_all = {**THETA[name], "deadline": 0.01}
    out = simulator(
        model=f"{name}_deadline", theta=th_all, n_samples=2000, random_state=6
    )
    assert np.all(out["rts"] == OMISSION_SENTINEL)
    _check_omission_contract(out, 2000)


# --------------------------------------------------------------------------
# Library integration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_simulator_class_and_defaults(name):
    cfg = ModelConfigBuilder.from_model(name)
    theta = dict(zip(cfg["params"], cfg["default_params"]))
    out = Simulator(name).simulate(theta=theta, n_samples=3000, random_state=0)
    assert np.all(out["rts"] > 0)
    assert set(np.unique(out["choices"])) == set(cfg["choices"])


def test_trialwise_parameters():
    th = {k: [v, v] for k, v in THETA3.items()}
    th["mu0"], th["mu1"] = [-1.5, -0.2], [-0.2, -1.5]
    out = simulator(model="mtlnr2_conf3", theta=th, n_samples=3000, random_state=1)
    assert out["rts"].shape == (3000, 2, 1)
    assert out["choice_p"][0, 0::2].sum() > 0.9
    assert out["choice_p"][1, 1::2].sum() > 0.9


# --------------------------------------------------------------------------
# Choice decoding and the training-data pipeline
# --------------------------------------------------------------------------


def test_decode_choices_and_figure7_rating_side():
    for c in CONDITION_RHO:
        out = simulator(
            model="mtlnr2_conf3",
            theta=_figure7_theta(c),
            n_samples=2000,
            random_state=0,
        )
        winner, conf = decode_choices(out["choices"].ravel())
        assert set(np.unique(winner)) <= {0, 1}
        assert set(np.unique(conf)) <= {0, 1, 2}
        np.testing.assert_array_equal(winner + 2 * conf, out["choices"].ravel())
        assert np.all(np.isfinite(out["rts"]))
        # 6-point old/new rating scale: side agrees with the winner
        rating = np.where(winner == 1, 4 + conf, 3 - conf)
        assert set(np.unique(rating)) <= set(range(1, 7))
        assert np.all((rating >= 4) == (winner == 1))


def test_lan_training_data_pipeline_applies_ordering(tmp_path):
    """The library's training-data generator runs and applies the swap."""
    from ssms.config import get_default_generator_config
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

    config = get_default_generator_config(model="mtlnr2_conf3")
    config["pipeline"]["n_parameter_sets"] = 10
    config["pipeline"]["n_subruns"] = 1
    config["pipeline"]["n_cpus"] = 1
    config["simulator"]["n_samples"] = 2000
    config["training"]["n_samples_per_param"] = 100
    # save=False still creates the output folder; keep it in pytest's tmp_path.
    config["output"]["folder"] = str(tmp_path / "lan_pipeline_out")
    data = TrainingDataGenerator(config).generate_data_training(save=False)
    assert data["theta"].shape == (10, 10)
    assert np.isfinite(data["lan_labels"]).all()
    params = ModelConfigBuilder.from_model("mtlnr2_conf3")["params"]
    th = data["theta"]
    for i in (0, 1):
        assert np.all(
            th[:, params.index(f"d2acc{i}")] >= th[:, params.index(f"d1acc{i}")]
        )


# --------------------------------------------------------------------------
# Omission-label contract (mirrors tests/test_omission_labels.py for this
# Python-censoring simulator; that file is left unchanged)
# --------------------------------------------------------------------------


def _omission_labels(choices, rts, possible):
    from ssms.dataset_generators.pipelines.simulation_pipeline import (
        SimulationPipeline,
    )

    return SimulationPipeline._compute_auxiliary_labels(
        None,
        {
            "choices": choices,
            "rts": rts,
            "metadata": {"possible_choices": list(possible)},
        },
    )


@pytest.mark.parametrize("name", MODELS)
def test_mtlnr_marks_omissions_in_rts_not_choices(name):
    cfg = ModelConfigBuilder.from_model(f"{name}_deadline")
    theta = {
        **THETA[name],
        "mu0": 0.0,
        "mu1": 0.3,
        "sigma0": 0.6,
        "sigma1": 0.6,
        "t": 0.2,
        "deadline": 0.8,
    }
    out = simulator(
        model=f"{name}_deadline", theta=theta, n_samples=8000, random_state=0
    )
    choices = np.asarray(out["choices"]).ravel()
    rts = np.asarray(out["rts"]).ravel()
    assert (rts == OMISSION_SENTINEL).any(), "no omissions: lower the deadline"
    assert not (choices == OMISSION_SENTINEL).any()
    labels = _omission_labels(choices, rts, possible=cfg["choices"])
    assert labels["omission_p"][0, 0] == pytest.approx(
        float((rts == OMISSION_SENTINEL).mean()), abs=1e-9
    )
    assert labels["choice_p"][0].sum() == pytest.approx(1.0, abs=1e-9)
    kept = choices[rts != OMISSION_SENTINEL]
    for i, c in enumerate(cfg["choices"]):
        assert labels["choice_p_no_omission"][0, i] == pytest.approx(
            float((kept == c).mean()), abs=1e-9
        )
