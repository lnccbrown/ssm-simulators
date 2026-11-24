"""TRDM has two onsets; shifting every RT by the evidence onset loses data."""

import numpy as np
import pytest

from ssms.config import (
    KDE_NO_DISPLACE_T,
    ModelConfigBuilder,
    get_lan_kde_config,
    model_config,
)
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator


@pytest.mark.parametrize("model", ["trdm_2", "trdm_2_guess", "trdm_3"])
@pytest.mark.parametrize("deadline", [False, True])
def test_generator_disables_t_displacement(model, deadline, tmp_path):
    assert model in KDE_NO_DISPLACE_T
    name = model + ("_deadline" if deadline else "")
    cfg = get_lan_kde_config(name)
    cfg["pipeline"]["n_cpus"] = 1
    cfg["output"]["folder"] = str(tmp_path)
    cfg["estimator"]["displace_t"] = True
    model_cfg = ModelConfigBuilder.from_model(name)
    with pytest.warns(UserWarning, match="Overriding setting to False"):
        generator = TrainingDataGenerator(cfg, model_cfg)
    assert generator.generator_config["estimator"]["displace_t"] is False
    assert generator._generation_pipeline.estimator_builder.displace_t is False
    # The caller's config is unchanged.
    assert cfg["estimator"]["displace_t"] is True


def test_early_timer_density_survives_generator_override(tmp_path):
    cfg = get_lan_kde_config("trdm_2")
    cfg["pipeline"]["n_cpus"] = 1
    cfg["output"]["folder"] = str(tmp_path)
    cfg["estimator"]["displace_t"] = True
    with pytest.warns(UserWarning, match="Overriding setting to False"):
        generator = TrainingDataGenerator(cfg, model_config["trdm_2"])
    pipeline = generator._generation_pipeline
    theta = dict(
        zip(model_config["trdm_2"]["params"], [2.0, 1.0, 1.0, 1.0, 0.8, 4.0, 0.5, 0.01])
    )
    simulations = pipeline.simulator.simulate(theta, n_samples=4000, random_state=12)
    rt = np.asarray(simulations["rts"]).ravel()
    choice = np.asarray(simulations["choices"]).ravel()
    early = (rt > 0) & (rt < theta["t"])
    assert early.sum() > 3000
    estimator = pipeline.estimator_builder.build(theta, simulations)
    assert np.isfinite(estimator.evaluate(rt[early][:20], choice[early][:20])).all()
