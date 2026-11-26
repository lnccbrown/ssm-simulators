"""The statically registered TRDM configs are exactly what the contribution declared."""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder, get_model_registry


def test_models_are_statically_registered(declarations):
    from ssms.config import model_config
    from ssms.config._modelconfig import get_model_config

    reg = get_model_registry()
    built_in = get_model_config()
    for m in declarations["models"]:
        assert reg.has_model(m["name"])
        assert m["name"] in built_in
        assert m["name"] in model_config


def test_config_matches_declaration(declarations):
    for m in declarations["models"]:
        cfg = ModelConfigBuilder.from_model(m["name"])
        names = [p["name"] for p in m["parameters"]]
        assert cfg["params"] == names
        assert cfg["n_params"] == len(names)
        lo, hi = cfg["param_bounds"]
        for i, p in enumerate(m["parameters"]):
            assert [lo[i], hi[i]] == pytest.approx(p["bounds"])
            assert cfg["default_params"][i] == pytest.approx(p["default"])
            assert lo[i] <= p["default"] <= hi[i]
            if p["neutral"] is not None:
                assert lo[i] <= p["neutral"] <= hi[i]
        assert cfg["parameter_transforms"] == {"sampling": [], "simulation": []}
        assert m["parameter_transforms"] == {"sampling": [], "simulation": []}


@pytest.mark.parametrize("model", ["trdm_2", "trdm_2_guess", "trdm_3"])
def test_defaults_simulate(model):
    cfg = ModelConfigBuilder.from_model(model)
    out = simulator(
        theta=np.array(cfg["default_params"]),
        model=model,
        n_samples=2000,
        random_state=0,
    )
    rts = np.asarray(out["rts"]).ravel()
    assert np.all(np.isfinite(rts)) and np.all(rts > 0)
    assert set(np.unique(out["choices"])) <= set(cfg["choices"])
    assert out["choice_p"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("model", ["trdm_2", "trdm_2_guess", "trdm_3"])
def test_deadline_variant_exists(model):
    cfg = ModelConfigBuilder.from_model(model + "_deadline")
    assert cfg["params"][-1] == "deadline"


def test_simulator_class_runs():
    from ssms import Simulator

    res = Simulator("trdm_2").simulate(
        theta={
            "v0": 2.0,
            "v1": 0.5,
            "b0": 1.0,
            "b1": 1.0,
            "t": 0.3,
            "vT": 1.0,
            "sT": 0.5,
            "tT": 0.05,
        },
        n_samples=500,
        random_state=1,
    )
    assert np.asarray(res["rts"]).shape[0] == 500


def test_vectorised_trials_and_seed_reproducibility():
    theta = np.tile([2.0, 0.5, 1.0, 1.0, 0.3, 1.0, 0.5, 0.05], (5, 1))
    theta[:, 5] = np.linspace(0.2, 2.0, 5)
    a = simulator(theta=theta, model="trdm_2", n_samples=300, random_state=3)
    b = simulator(theta=theta, model="trdm_2", n_samples=300, random_state=3)
    assert np.asarray(a["rts"]).shape == (300, 5, 1)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])
    # faster timer -> faster responses
    mrt = np.asarray(a["rts"])[:, :, 0].mean(axis=0)
    assert mrt[0] > mrt[-1]
