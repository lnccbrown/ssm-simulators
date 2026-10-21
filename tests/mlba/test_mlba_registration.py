"""The models register, build and simulate through the public ssms entry points.

Also checks that ``MLBA_SPEC`` is a faithful description of what actually
got registered: same parameters, in the same order, with the same bounds,
defaults and transform pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssms.basic_simulators import Simulator
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder, get_model_registry, model_config
from ssms.config._modelconfig.mlba import get_mlba2_config, get_mlba3_config

MODELS = ["mlba2", "mlba3"]


def _default_theta(model: str) -> dict:
    cfg = ModelConfigBuilder.from_model(model)
    return dict(zip(cfg["params"], cfg["default_params"]))


@pytest.mark.parametrize("model", MODELS)
def test_model_is_registered(model):
    assert get_model_registry().has_model(model)


@pytest.mark.parametrize(
    "model, factory", [("mlba2", get_mlba2_config), ("mlba3", get_mlba3_config)]
)
def test_model_is_statically_registered(model, factory):
    """The configs come from the static ``_modelconfig`` registry, no import glue."""
    assert model in model_config
    registered = model_config[model]
    built = factory()
    assert registered["params"] == built["params"]
    assert registered["default_params"] == built["default_params"]
    assert registered["param_bounds"] == built["param_bounds"]
    assert registered["simulator"] is built["simulator"]


@pytest.mark.parametrize("model", MODELS)
def test_config_is_valid(model):
    cfg = ModelConfigBuilder.from_model(model)
    is_valid, errors = ModelConfigBuilder.validate_config(cfg, strict=True)
    assert is_valid, errors
    assert cfg["n_particles"] == cfg["nchoices"]
    assert cfg["choices"] == list(range(cfg["nchoices"]))
    assert "param_bounds_dict" in cfg


@pytest.mark.parametrize("model", MODELS)
def test_declaration_matches_registered_config(model, declared_models):
    cfg = ModelConfigBuilder.from_model(model)
    declared = declared_models[model]["parameters"]

    assert cfg["params"] == [p["name"] for p in declared]
    assert cfg["default_params"] == pytest.approx([p["default"] for p in declared])
    lower, upper = cfg["param_bounds"]
    assert lower == pytest.approx([p["bounds"][0] for p in declared])
    assert upper == pytest.approx([p["bounds"][1] for p in declared])


@pytest.mark.parametrize("model", MODELS)
def test_declared_transform_pipeline_matches(model, declared_models):
    cfg = ModelConfigBuilder.from_model(model)
    declared = declared_models[model]["parameter_transforms"]

    for phase in ("sampling", "simulation"):
        registered = ModelConfigBuilder.get_transforms(cfg, phase)
        assert len(registered) == len(declared[phase])
        for instance, spec in zip(registered, declared[phase]):
            module, _, cls = spec["type"].rpartition(".")
            assert type(instance).__name__ == cls
            assert type(instance).__module__ == module
            for field, value in spec.items():
                if field == "type":
                    continue
                assert getattr(instance, field) == value, (phase, spec["type"], field)


@pytest.mark.parametrize("model", MODELS)
def test_simulates_through_function_and_class(model):
    theta = _default_theta(model)
    cfg = ModelConfigBuilder.from_model(model)

    out = simulator(theta=theta, model=model, n_samples=2000, random_state=11)
    assert out["rts"].shape == (2000, 1)
    assert set(np.unique(out["choices"])) <= set(cfg["choices"])
    assert np.all(out["rts"] > 0)
    assert out["choice_p"].shape == (1, cfg["nchoices"])
    assert out["choice_p"].sum() == pytest.approx(1.0)

    same = Simulator(model).simulate(theta, n_samples=2000, random_state=11)
    np.testing.assert_array_equal(np.asarray(same["rts"]).ravel(), out["rts"].ravel())
    np.testing.assert_array_equal(
        np.asarray(same["choices"]).ravel(), out["choices"].ravel()
    )


@pytest.mark.parametrize("model", MODELS)
def test_random_state_is_honoured(model):
    """The seeded wrapper makes repeated runs reproducible.

    cssm.lba_vanilla itself ignores random_state (see ssms/basic_simulators/lba_seeded.py),
    which is why the built-in lba models are not reproducible this way.
    """
    theta = _default_theta(model)
    a = simulator(theta=theta, model=model, n_samples=500, random_state=1234)
    b = simulator(theta=theta, model=model, n_samples=500, random_state=1234)
    c = simulator(theta=theta, model=model, n_samples=500, random_state=4321)
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])
    assert not np.array_equal(a["rts"], c["rts"])


@pytest.mark.parametrize("model", MODELS)
def test_per_trial_drift_sd_is_respected(model):
    """A trial-varying s must bite on every trial, not only the first.

    ssms routes a theta entry named 's' through make_noise_vec, which
    broadcasts only the first trial's value; MLBAFrontEnd rewrites the sd
    array per trial so this does not silently flatten.  A is set to 0 so that
    all of the RT spread comes from drift variability.
    """
    s_values = [0.05, 0.5, 1.5]
    theta = {
        k: np.array([v] * len(s_values), dtype=np.float32)
        for k, v in _default_theta(model).items()
    }
    theta["A"] = np.zeros(len(s_values), dtype=np.float32)
    theta["s"] = np.array(s_values, dtype=np.float32)
    out = simulator(theta=theta, model=model, n_samples=20000, random_state=3)
    spread = out["rts"].std(axis=0).ravel()
    assert spread[0] < spread[1] < spread[2], spread
    assert spread[2] > 5 * spread[0], spread


@pytest.mark.parametrize("model", MODELS)
def test_per_trial_attributes_are_respected(model):
    theta = {
        k: np.array([v, v], dtype=np.float32) for k, v in _default_theta(model).items()
    }
    theta["P0"] = np.array([theta["P0"][0], theta["P0"][0] * 0.5], dtype=np.float32)
    out = simulator(theta=theta, model=model, n_samples=4000, random_state=3)
    assert out["choice_p"][0, 0] != out["choice_p"][1, 0]


@pytest.mark.parametrize("model", MODELS)
def test_sigma_noise_argument_is_refused(model):
    """s is a registered parameter, so the simulate-time noise argument is not free."""
    with pytest.raises(ValueError, match="sigma_noise"):
        simulator(
            theta=_default_theta(model), model=model, n_samples=10, sigma_noise=1.0
        )


@pytest.mark.parametrize("model", MODELS)
def test_training_data_pipeline_accepts_the_model(model, tmp_path):
    """The LAN training-data generator runs end to end on the registered model.

    This exercises the sampling transforms, the uniform parameter sampler over
    param_bounds, and the KDE likelihood estimator, i.e. the path ssms exists
    for beyond single simulations.
    """
    from ssms.config import get_default_generator_config
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

    config = get_default_generator_config("lan", model=model)
    config["pipeline"].update(
        {"n_parameter_sets": 4, "n_subruns": 1, "n_cpus": 1, "n_samples": 2000}
    )
    config["training"]["n_samples_per_param"] = 200
    config["output"]["folder"] = str(tmp_path)
    out = TrainingDataGenerator(
        config=config, model_config=ModelConfigBuilder.from_model(model)
    ).generate_data_training(save=False, verbose=False)

    n_params = len(ModelConfigBuilder.from_model(model)["params"])
    assert out["theta"].shape == (4, n_params)
    assert out["lan_data"].shape == (800, n_params + 2)
    assert np.isfinite(out["lan_data"]).all()
    assert np.isfinite(out["lan_labels"]).all()
