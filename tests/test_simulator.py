import pytest

import numpy as np
import pandas as pd
from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config
from copy import deepcopy


@pytest.fixture(scope="module")
def sim_input_data():
    data = dict()

    # Prepare input data for each model
    for key in model_config.keys():
        # Get model parameter names
        model_param_list = model_config[key]["params"]

        # Dictionary with all scalar values
        theta_dict_all_scalars = {
            param: model_config[key]["default_params"][i]
            for i, param in enumerate(model_param_list)
        }

        # Dictionary with all vectors
        theta_dict_all_vectors = {
            param: np.tile(
                np.array(model_config[key]["default_params"][i]),
                100,
            )
            for i, param in enumerate(model_param_list)
        }

        # Dictionary with mix of scalars and vectors
        theta_dict_sca_vec = deepcopy(theta_dict_all_vectors)
        cnt = 0
        for tmp_key in theta_dict_all_scalars.keys():
            theta_dict_sca_vec[tmp_key] = theta_dict_all_scalars[tmp_key]
            if cnt > 0:
                break
            cnt += 1

        # Dictionary with vectros of uneven length
        theta_dict_uneven = deepcopy(theta_dict_all_vectors)

        cnt = 0
        for tmp_key in theta_dict_all_scalars.keys():
            if cnt > 0:
                break
            theta_dict_uneven[tmp_key] = np.concatenate(
                [theta_dict_all_vectors[tmp_key], np.zeros(2)]
            )
            cnt += 1

        # Input is list
        theta_list = [
            model_config[key]["default_params"][i]
            for i, param in enumerate(model_param_list)
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
    for key in model_config.keys():
        # Go over different types of input data
        # (listed above in sim_input_data() fixture)
        for subkey in sim_input_data[key].keys():
            print(key, " -> ", subkey)

            # Go over different number of samples
            if subkey == "theta_dict_uneven":
                for n_samples in [1, 10]:
                    print("input data: ", sim_input_data[key][subkey])
                    print("n_samples: ", n_samples)
                    try:
                        out = simulator(
                            model=key,
                            theta=sim_input_data[key][subkey],
                            n_samples=n_samples,
                        )
                    except ValueError:
                        raised_val_error = 1

                    assert raised_val_error
            else:
                for n_samples in [1, 10]:
                    print("input data: ", sim_input_data[key][subkey])
                    print("n_samples: ", n_samples)
                    out = simulator(
                        model=key,
                        theta=sim_input_data[key][subkey],
                        n_samples=n_samples,
                    )
                    assert isinstance(out, dict)
                    assert "metadata" in out.keys()
                    assert "rts" in out.keys()
                    assert "choices" in out.keys()
