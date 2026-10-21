"""The library's omission contract, on the registered deadline variants.

A censored trial must put -999 in ``rts`` only: its latent choice stays a
legal choice, and ``choice_p`` stays finite and normalised over all returned
choices (omissions included).  ``ModelConfigBuilder`` derives ``mlba2_deadline``
and ``mlba3_deadline`` automatically from the registered configs, so both are
exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssms.basic_simulators.simulator import OMISSION_SENTINEL, simulator
from ssms.config import ModelConfigBuilder

DEADLINE_MODELS = ["mlba2_deadline", "mlba3_deadline"]


def _simulate(model: str, deadline: float, n_samples: int = 20000, seed: int = 5):
    cfg = ModelConfigBuilder.from_model(model)
    theta = dict(zip(cfg["params"], cfg["default_params"]))
    theta["deadline"] = deadline
    return cfg, simulator(
        theta=theta, model=model, n_samples=n_samples, random_state=seed
    )


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_deadline_variant_is_derivable(model):
    cfg = ModelConfigBuilder.from_model(model)
    assert cfg["params"][-1] == "deadline"
    assert cfg["deadline"] is True
    assert cfg["name"] == model


def _median_rt(model: str) -> float:
    """Median RT with no effective deadline, so the fixture can censor half."""
    _, out = _simulate(model, deadline=10.0, n_samples=5000, seed=1)
    return float(np.median(np.asarray(out["rts"]).ravel()))


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_forced_omissions_follow_the_contract(model):
    """A deadline at the median RT must censor roughly half the trials."""
    deadline = _median_rt(model)
    cfg, out = _simulate(model, deadline=deadline)
    rts = np.asarray(out["rts"]).ravel()
    choices = np.asarray(out["choices"]).ravel()
    omitted = rts == OMISSION_SENTINEL

    assert 0.3 < omitted.mean() < 0.7, f"unexpected omission rate {omitted.mean()}"

    # 1. the sentinel appears in rts only
    assert not np.any(choices == OMISSION_SENTINEL)
    # 2. the latent choice of a censored trial is still a legal choice
    assert set(np.unique(choices[omitted])) <= set(cfg["choices"])
    assert set(np.unique(choices)) <= set(cfg["choices"])
    # 3. surviving RTs are genuine times
    assert np.all(rts[~omitted] > 0)
    assert np.all(rts[~omitted] < deadline)
    # 4. choice_p is finite, normalised, and counts the censored trials too
    choice_p = out["choice_p"]
    assert np.all(np.isfinite(choice_p))
    assert choice_p.sum() == pytest.approx(1.0)
    for i, choice in enumerate(cfg["choices"]):
        assert choice_p[0, i] == pytest.approx((choices == choice).mean(), abs=1e-9)
    # 5. omission_p agrees with rts
    assert out["omission_p"][0, 0] == pytest.approx(omitted.mean(), abs=1e-9)
    # 6. the omission-free choice probabilities exclude the censored trials
    kept = choices[~omitted]
    for i, choice in enumerate(cfg["choices"]):
        assert out["choice_p_no_omission"][0, i] == pytest.approx(
            (kept == choice).mean(), abs=1e-9
        )


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_a_loose_deadline_censors_nothing(model):
    _, out = _simulate(model, deadline=10.0)
    rts = np.asarray(out["rts"]).ravel()
    assert not np.any(rts == OMISSION_SENTINEL)
    assert out["omission_p"][0, 0] == 0.0
    np.testing.assert_allclose(out["choice_p"], out["choice_p_no_omission"], atol=1e-12)


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_everything_censored_keeps_choice_p_finite(model):
    """The degenerate end of the contract: all trials time out."""
    cfg, out = _simulate(model, deadline=0.001, n_samples=2000)
    rts = np.asarray(out["rts"]).ravel()
    choices = np.asarray(out["choices"]).ravel()
    assert np.all(rts == OMISSION_SENTINEL)
    assert not np.any(choices == OMISSION_SENTINEL)
    assert set(np.unique(choices)) <= set(cfg["choices"])
    assert np.all(np.isfinite(out["choice_p"]))
    assert out["choice_p"].sum() == pytest.approx(1.0)
    assert out["omission_p"][0, 0] == 1.0
