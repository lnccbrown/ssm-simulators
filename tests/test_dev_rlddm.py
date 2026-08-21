"""Tests for the `dev_rlddm` model configuration.

Contributed from a paper via HSSMCortex. The generative claims below are the ones its verification
report checked; see the pull request body for the report and the rungs that ran.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config


N_SAMPLES = 20_000


def _theta(name):
    cfg = model_config[name]
    return dict(zip(cfg["params"], cfg["default_params"]))


def test_dev_rlddm_is_registered():
    """The config resolves through the registry the package exposes."""
    assert "dev_rlddm" in model_config
    cfg = model_config["dev_rlddm"]
    assert len(cfg["params"]) == cfg["n_params"] == len(cfg["default_params"])
    assert len(cfg["param_bounds"][0]) == len(cfg["param_bounds"][1]) == cfg["n_params"]


def test_dev_rlddm_simulates():
    """It runs, and produces response times no smaller than its own non-decision time."""
    cfg = model_config["dev_rlddm"]
    theta = _theta("dev_rlddm")
    out = simulator(model="dev_rlddm", theta=theta, n_samples=N_SAMPLES)
    rts = np.asarray(out["rts"]).ravel()
    assert np.isfinite(rts).all()
    assert set(np.unique(np.asarray(out["choices"]).ravel())) <= set(cfg["choices"])
    if "t" in theta:
        assert rts.min() >= theta["t"] - 1e-9


def test_dev_rlddm_reduces_to_ddm():
    """With a_mod = 0.0 the new mechanism is off and this must be `ddm`.

    The paper's own claim, and the one the contributing pipeline checked at rung 2. Compared as
    response-time distributions per choice rather than sample by sample, because the two models are
    only required to agree in law.
    """
    theta = _theta("dev_rlddm")
    theta['a_mod'] = 0.0
    ours = simulator(model="dev_rlddm", theta=theta, n_samples=N_SAMPLES, random_state=7)
    theirs = simulator(model="ddm", theta=_theta("ddm"), n_samples=N_SAMPLES,
                       random_state=7)

    for choice in model_config["ddm"]["choices"]:
        a = np.asarray(ours["rts"]).ravel()[np.asarray(ours["choices"]).ravel() == choice]
        b = np.asarray(theirs["rts"]).ravel()[np.asarray(theirs["choices"]).ravel() == choice]
        if min(len(a), len(b)) < 500:
            continue
        grid = np.quantile(np.concatenate([a, b]), np.linspace(0.01, 0.99, 99))
        gap = np.abs(
            np.searchsorted(np.sort(a), grid) / len(a)
            - np.searchsorted(np.sort(b), grid) / len(b)
        ).max()
        band = 3.0 * np.sqrt(2.0 / min(len(a), len(b)))
        assert gap <= band, f"choice {choice}: D = {gap:.4f} against a band of {band:.4f}"
