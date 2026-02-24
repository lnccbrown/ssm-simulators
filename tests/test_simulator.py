from copy import deepcopy
import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def sim_input_data():
    data = dict()

    # Prepare input data for each model
    for key, config in model_config.items():
        # Get model parameter names
        model_param_list = config["params"]

        # Dictionary with all scalar values
        theta_dict_all_scalars = {
            param: config["default_params"][i]
            for i, param in enumerate(model_param_list)
        }

        # Dictionary with all vectors
        theta_dict_all_vectors = {
            param: np.tile(
                np.array(config["default_params"][i]),
                100,
            )
            for i, param in enumerate(model_param_list)
        }

        # Dictionary with mix of scalars and vectors
        theta_dict_sca_vec = deepcopy(theta_dict_all_vectors)
        cnt = 0
        for tmp_key in theta_dict_all_scalars:
            theta_dict_sca_vec[tmp_key] = theta_dict_all_scalars[tmp_key]
            if cnt > 0:
                break
            cnt += 1

        # Dictionary with vectors of uneven length
        theta_dict_uneven = deepcopy(theta_dict_all_vectors)

        cnt = 0
        for tmp_key in theta_dict_all_scalars:
            if cnt > 0:
                break
            theta_dict_uneven[tmp_key] = np.concatenate(
                [theta_dict_all_vectors[tmp_key], np.zeros(2)]
            )
            cnt += 1

        # Input is list
        theta_list = [
            config["default_params"][i] for i, param in enumerate(model_param_list)
        ]

        # Input is numpy array
        theta_nparray = np.array(theta_list)

        # Input is pd.DataFrame
        theta_pd_1 = pd.DataFrame([theta_nparray], columns=model_param_list)

        theta_pd_n = pd.DataFrame(
            np.tile(theta_nparray, (100, 1)), columns=model_param_list
        )

        data[key] = {
            "theta_dict_all_scalars": theta_dict_all_scalars,
            "theta_dict_all_vectors": theta_dict_all_vectors,
            "theta_dict_sca_vec": theta_dict_sca_vec,
            "theta_dict_uneven": theta_dict_uneven,
            "theta_list": theta_list,
            "theta_nparray": theta_nparray,
            "theta_pd_1": theta_pd_1,
            "theta_pd_n": theta_pd_n,
        }

    return data


def test_simulator_runs(sim_input_data):
    """Test that simulator runs for all models"""
    # Go over model names
    for key in model_config:
        # Go over different types of input data
        # (listed above in sim_input_data() fixture)
        for subkey in sim_input_data[key]:
            logger.debug(f"{key} -> {subkey}")

            # Go over different number of samples
            if subkey == "theta_dict_uneven":
                for n_samples in [1, 10]:
                    with pytest.raises(ValueError):
                        simulator(
                            model=key,
                            theta=sim_input_data[key][subkey],
                            n_samples=n_samples,
                        )
            else:
                for n_samples in [1, 10]:
                    logger.debug("input data: %s", sim_input_data[key][subkey])
                    logger.debug("n_samples: %s", n_samples)
                    out = simulator(
                        model=key,
                        theta=sim_input_data[key][subkey],
                        n_samples=n_samples,
                    )
                    assert isinstance(out, dict)
                    assert "metadata" in out
                    assert "rts" in out
                    assert "choices" in out


@pytest.mark.parametrize(
    "model",
    [
        # ddm_models.pyx — ddm_flexbound (constant boundary)
        "ddm",
        # ddm_models.pyx — ddm_flexbound (angle boundary)
        "angle",
        # ddm_models.pyx — full_ddm
        "full_ddm",
        # levy_models.pyx — levy_flexbound
        "levy",
        # ornstein_models.pyx — ornstein_uhlenbeck
        "ornstein",
        # parallel_models.pyx — ddm_flexbound_par2
        "ddm_par2",
        # sequential_models.pyx — ddm_flexbound_seq2 (true parallel path)
        "ddm_seq2",
        # sequential_models.pyx — ddm_flexbound_mic2_ornstein (warning-only)
        "ddm_mic2_adj",
        # ddm_models.pyx — ddm_flexbound_tradeoff (warning-only)
        "tradeoff_no_bias",
        # race_models.pyx — race_model (warning-only)
        "race_no_bias_2",
        # lba_models.pyx — lba_vanilla
        "lba2",
        # race_models.pyx — lca
        "lca_3",
        # sequential_models.pyx — rlwm_lba_pw_v1 (warning-only)
        "dev_rlwm_lba_pw_v1",
    ],
)
def test_simulator_runs_parallel(model):
    """Smoke test: n_threads=4 does not crash, covering every distinct Cython backend."""
    cfg = model_config[model]
    theta = {p: cfg["default_params"][i] for i, p in enumerate(cfg["params"])}
    out = simulator(model=model, theta=theta, n_samples=10, n_threads=4)
    assert isinstance(out, dict)
    assert "rts" in out
    assert "choices" in out
    assert "metadata" in out


def test_simulator_sigma_noise_conflict_raises():
    """ValueError when 's' is in theta and sigma_noise is also explicitly passed."""
    with pytest.raises(ValueError, match="sigma_noise parameter should be None"):
        simulator(
            model="ddm",
            theta={"v": 0.0, "a": 1.0, "z": 0.5, "t": 0.3, "s": 1.0},
            sigma_noise=0.5,
            n_samples=5,
        )


def test_simulator_no_noise_flag():
    """no_noise=True suppresses diffusion noise (sigma_noise set to 0.0 internally)."""
    out = simulator(
        model="ddm",
        theta={"v": 0.0, "a": 1.0, "z": 0.5, "t": 0.3},
        no_noise=True,
        n_samples=50,
    )
    assert "rts" in out
    assert "choices" in out


def test_simulator_bad_return_type_raises():
    """TypeError when the underlying simulator callable returns a non-dict."""
    from ssms.config import ModelConfigBuilder

    real_config = ModelConfigBuilder.from_model("ddm")
    bad_config = dict(real_config)
    bad_config["simulator"] = lambda **kwargs: [1, 2, 3]

    # ModelConfigBuilder is imported locally inside simulator(), so patch at the source.
    with patch("ssms.config.ModelConfigBuilder.from_model", return_value=bad_config):
        with pytest.raises(
            TypeError, match="Expected simulator to return a dictionary"
        ):
            simulator(
                model="ddm",
                theta={"v": 0.0, "a": 1.0, "z": 0.5, "t": 0.3},
                n_samples=5,
            )


def test_make_boundary_dict_registry_path():
    """make_boundary_dict uses boundary registry when 'boundary' key is not callable."""
    from ssms.basic_simulators.simulator import make_boundary_dict

    # Old-style config: no callable "boundary" key, only "boundary_name"
    # This exercises the else-branch (lines 283-291) that does a registry lookup.
    config = {"boundary_name": "constant"}
    theta = {"a": 1.0, "z": 0.5, "v": 1.0}
    result = make_boundary_dict(config, theta)
    assert "boundary_fun" in result
    assert callable(result["boundary_fun"])
    assert "boundary_params" in result
    # "constant" boundary has params=["a"], so only "a" should be extracted
    assert result["boundary_params"] == {"a": 1.0}
