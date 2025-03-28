from copy import deepcopy

import numpy as np
import pytest

from ssms.dataset_generators.lan_mlp import data_generator
from ssms.config import model_config, data_generator_config


def test_data_generator(tmp_path):
    # Check included models
    assert list(model_config.keys())[:10] == [
        "ddm",
        "ddm_legacy",
        "angle",
        "weibull",
        "levy",
        "levy_angle",
        "full_ddm",
        "full_ddm_rv",
        "ddm_st",
        "ddm_truncnormt",
    ]

    # Take an example config for a given model
    model_conf = model_config["ddm"]

    assert type(model_conf["simulator"]).__name__ == "cython_function_or_method"

    assert callable(model_conf["simulator"])
    assert callable(model_conf["boundary"])

    # Initialize the generator config (for MLP LANs)
    gen_config = data_generator_config["lan"]
    generator_config = deepcopy(gen_config)
    # Specify generative model (one from the list of included models mentioned above)
    generator_config["dgp_list"] = "angle"
    # Specify number of parameter sets to simulate
    generator_config["n_parameter_sets"] = 100
    # Specify how many samples a simulation run should entail
    generator_config["n_samples"] = 1000

    # set ouput folder
    generator_config["output_folder"] = str(tmp_path)

    # Now let's define our corresponding `model_config`.
    angle_model_config = model_config["angle"]

    with pytest.raises(ValueError):
        data_generator(
        generator_config=generator_config, model_config=None
    )
    
    with pytest.raises(ValueError):
        data_generator(
        generator_config=None, model_config=angle_model_config
    )


    my_dataset_generator = data_generator(
        generator_config=generator_config, model_config=angle_model_config
    )
    training_data = my_dataset_generator.generate_data_training_uniform(save=True)
    
    new_data_file = list(tmp_path.iterdir())[0]
    assert new_data_file.exists()
    assert new_data_file.suffix == '.pickle'


    # Because randomly generated arrays may differ across OS and versions of Python,
    # even when setting a random seed, we check for array shape
    td_array_shapes = {
        k: v.shape for k, v in training_data.items() if isinstance(v, np.ndarray)
    }

    expected_shapes = {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    }

    assert td_array_shapes == expected_shapes

    assert list(training_data.keys()) == [
        "cpn_data",
        "cpn_labels",
        "cpn_no_omission_data",
        "cpn_no_omission_labels",
        "opn_data",
        "opn_labels",
        "gonogo_data",
        "gonogo_labels",
        "thetas",
        "lan_data",
        "lan_labels",
        "binned_128",
        "binned_256",
        "generator_config",
        "model_config",
    ]

    assert training_data["model_config"]["constrained_param_space"] == {
        "a": (0.3, 3.0),
        "t": (0.001, 2.0),
        "theta": (-0.1, 1.3),
        "v": (-3.0, 3.0),
        "z": (0.1, 0.9),
    }

    del training_data["model_config"]["constrained_param_space"]
    assert training_data["model_config"] == angle_model_config
