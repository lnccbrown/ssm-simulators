from copy import deepcopy
import logging

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
                    raised_value_error = 0

                    try:
                        out = simulator(
                            model=key,
                            theta=sim_input_data[key][subkey],
                            n_samples=n_samples,
                        )
                    except ValueError:
                        raised_value_error = 1

                    assert raised_value_error
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
