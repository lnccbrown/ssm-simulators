import random
from copy import deepcopy

import numpy as np
import pytest
from expected_shapes import get_expected_shapes

from ssms.config import get_lan_config, model_config
from ssms.dataset_generators.lan_mlp import data_generator

N_PARAMETER_SETS = random.randint(2, 10)
N_TRAINING_SAMPLES_BY_PARAMETER_SET = random.randint(2, 10)
N_TRAINING_SAMPLES_BY_PARAMETER_SET = random.randint(
    6, 10
)  # seems to need to be at least 6 for n_paramsets = 1
N_SAMPLES = 4  # lowerbound seems to be 4 -- breaks if random number is chosen


def infer_constrained_param_space(conf: dict) -> dict:
    """Infer a conservative constrained parameter space from a model config.

    Supports two formats for ``param_bounds``:
    1. ``[[low_1, ...], [high_1, ...]]`` aligned with ``params`` order.
    2. ``{param: (low, high)}`` mapping.

    Silently skips malformed / mismatched entries.
    """
    params = conf.get("params")
    bounds = conf.get("param_bounds")

    if isinstance(bounds, dict):
        return bounds.copy()

    if (
        params
        and isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and isinstance(bounds[0], (list, tuple))
        and isinstance(bounds[1], (list, tuple))
    ):
        lows, highs = bounds
        return {p: (low, high) for p, low, high in zip(params, lows, highs)}

    return {}


def _make_gen_config(
    n_parameter_sets=N_PARAMETER_SETS,
    n_training_samples_by_parameter_set=N_TRAINING_SAMPLES_BY_PARAMETER_SET,
    n_samples=10,
    n_subruns=1,
):
    return {
        "n_parameter_sets": n_parameter_sets,
        "n_training_samples_by_parameter_set": n_training_samples_by_parameter_set,
        "n_samples": n_samples,
        "n_subruns": n_subruns,
    }


EXPECTED_KEYS = [
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

# TODO: Remove this once #114 is fixed
broken_models = [
    "lba_3_vs_constraint",  # broken
    "lba_angle_3_vs_constraint",  # broken
    "dev_rlwm_lba_race_v2",  # broken
]

# Ultra slow models, likely broken?
slow_prefixes = (
    "race",
    "dev_rlwm",
    "lba3",
    "lba_angle_3",
    "lca",
    "ddm_par2",
    "ddm_seq2",
    "ddm_mic2",
    "tradeoff",
)

gen_config = get_lan_config()
gen_config.update(
    _make_gen_config(
        N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET, N_SAMPLES, 1
    )
)

def test_data_persistance(tmp_path):
    model_conf = model_config["ddm"]
    my_dataset_generator = data_generator(
        generator_config=gen_config, model_config=model_conf
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


@pytest.mark.parametrize("model_name,model_conf", model_config.items())
def test_data_generator(model_name, model_conf):
    if model_name in broken_models:
        pytest.skip(f"Skipping broken model: {model_name}")
    if model_name.startswith(slow_prefixes):
        pytest.skip(f"Skipping slow model: {model_name}")

    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = model_name

    my_dataset_generator = data_generator(
        generator_config=generator_config, model_config=model_conf
    )
    training_data = my_dataset_generator.generate_data_training_uniform(save=False)

    # Because randomly generated arrays may differ across OS and versions of Python,
    # even when setting a random seed, we check for array shape only
    td_array_shapes = {
        k: v.shape for k, v in training_data.items() if isinstance(v, np.ndarray)
    }

    assert td_array_shapes == get_expected_shapes(
        model_conf, N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET
    )

    assert training_data["model_config"][
        "constrained_param_space"
    ] == infer_constrained_param_space(model_config[model_name])
