import random
from copy import deepcopy

import numpy as np
import pytest
from expected_shapes import expected_shapes

from ssms.config import get_lan_config, model_config
from ssms.dataset_generators.lan_mlp import data_generator

gen_config = get_lan_config()
# Specify number of parameter sets to simulate
gen_config["n_parameter_sets"] = 100
gen_config["n_training_samples_by_parameter_set"] = 100
# Specify how many samples a simulation run should entail
gen_config["n_samples"] = 10


def infer_constrained_param_space(conf: dict) -> dict:
    """Infer a conservative constrained parameter space from a model config.

    Supports two formats for ``param_bounds``:
    1. ``[[low_1, ...], [high_1, ...]]`` aligned with ``params`` order.
    2. ``{param: (low, high)}`` mapping.

    Silently skips malformed / mismatched entries.
    """
    params = conf.get("params")
    bounds = conf.get("param_bounds")
    if bounds is None:
        return {}
    out = {}
    if (
        isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and not isinstance(bounds[0], (int, float))
    ):
        if not params:
            return {}
        lows, highs = bounds
        if not (hasattr(lows, "__len__") and hasattr(highs, "__len__")):
            return {}
        for i, p in enumerate(params):
            if i < len(lows) and i < len(highs):
                out[p] = (lows[i], highs[i])
        return out
    if isinstance(bounds, dict):
        keys = params if params else list(bounds.keys())
        for p in keys:
            if p in bounds:
                out[p] = bounds[p]
        return out
    return {}


def test_data_persistance(tmp_path):
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["output_folder"] = str(tmp_path)
    generator_config["n_subruns"] = 1

    my_dataset_generator = data_generator(
        generator_config=generator_config, model_config=model_conf
    )
    my_dataset_generator.generate_data_training_uniform(save=True)
    new_data_file = list(tmp_path.iterdir())[0]
    assert new_data_file.exists()
    assert new_data_file.suffix == ".pickle"


@pytest.mark.parametrize("model_name", list(model_config.keys()))
def test_model_config(model_name):
    # Take an example config for a given model
    model_conf = deepcopy(model_config[model_name])

    assert type(model_conf["simulator"]).__name__ == "cython_function_or_method"

    assert callable(model_conf["simulator"])
    assert callable(model_conf["boundary"])


def test_bad_inputs():
    model_conf = model_config["ddm"]

    with pytest.raises(ValueError):
        data_generator(generator_config=gen_config, model_config=None)

    with pytest.raises(ValueError):
        data_generator(generator_config=None, model_config=model_conf)


# TODO: Remove this once #114 is fixed
models_to_skip = [
    "lba_3_vs_constraint",
    "lba_angle_3_vs_constraint",
    "dev_rlwm_lba_race_v2",
]
ok_model_config = [
    item for item in model_config.items() if item[0] not in models_to_skip
]
# TODO: Remove this once data generator is optimized for slow models (#113)
subset_size = 1 + len(ok_model_config) // 10
ok_model_config = random.sample(ok_model_config, subset_size)


@pytest.mark.parametrize("model_name,model_conf", ok_model_config)
def test_data_generator(model_name, model_conf):
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = model_name
    generator_config["n_subruns"] = 1

    my_dataset_generator = data_generator(
        generator_config=generator_config, model_config=model_conf
    )
    training_data = my_dataset_generator.generate_data_training_uniform(save=False)

    # Because randomly generated arrays may differ across OS and versions of Python,
    # even when setting a random seed, we check for array shape only
    td_array_shapes = {
        k: v.shape for k, v in training_data.items() if isinstance(v, np.ndarray)
    }

    assert td_array_shapes == expected_shapes[model_name]

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

    # Check that the inferred constrained parameter space from model config matches those in the generated training data
    assert training_data["model_config"][
        "constrained_param_space"
    ] == infer_constrained_param_space(model_config[model_name])
