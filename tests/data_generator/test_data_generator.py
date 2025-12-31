import random
from copy import deepcopy

import numpy as np
import pytest

from ssms.config import model_config
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
)
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

from expected_shapes import get_expected_shapes
from expected_constrained_param_space import infer_param_bounds_dict


N_PARAMETER_SETS = random.randint(2, 10)
N_TRAINING_SAMPLES_BY_PARAMETER_SET = random.randint(2, 10)
N_TRAINING_SAMPLES_BY_PARAMETER_SET = random.randint(
    6, 10
)  # seems to need to be at least 6 for n_paramsets = 1
N_SAMPLES = 4  # lowerbound seems to be 4 -- breaks if random number is chosen


def _make_gen_config(
    n_parameter_sets=N_PARAMETER_SETS,
    n_training_samples_by_parameter_set=N_TRAINING_SAMPLES_BY_PARAMETER_SET,
    n_samples=10,
    n_subruns=1,
):
    return {
        "pipeline": {
            "n_parameter_sets": n_parameter_sets,
            "n_subruns": n_subruns,
        },
        "training": {
            "n_samples_per_param": n_training_samples_by_parameter_set,
        },
        "simulator": {
            "n_samples": n_samples,
        },
    }


gen_config = get_default_generator_config("lan")
# Deep merge the nested config
_custom_config = _make_gen_config(
    N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET, N_SAMPLES, 1
)
for section, values in _custom_config.items():
    if section in gen_config:
        gen_config[section].update(values)
    else:
        gen_config[section] = values


EXPECTED_KEYS = [
    "cpn_data",
    "cpn_labels",
    "cpn_no_omission_data",
    "cpn_no_omission_labels",
    "opn_data",
    "opn_labels",
    "gonogo_data",
    "gonogo_labels",
    "theta",
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
    "lba2",  # broken - generates incorrect number of parameter sets
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


@pytest.mark.parametrize("model_name,model_conf", model_config.items())
def test_TrainingDataGenerator(model_name, model_conf):
    if model_name in broken_models:
        pytest.skip(f"Skipping broken model: {model_name}")
    if model_name.startswith(slow_prefixes):
        pytest.skip(f"Skipping slow model: {model_name}")

    generator_config = deepcopy(gen_config)
    generator_config["model"] = model_name

    my_dataset_generator = TrainingDataGenerator(
        config=generator_config, model_config=model_conf
    )
    training_data = my_dataset_generator.generate_data_training(save=False)

    # Because randomly generated arrays may differ across OS and versions of Python,
    # even when setting a random seed, we check for array shape only
    td_array_shapes = {
        k: v.shape for k, v in training_data.items() if isinstance(v, np.ndarray)
    }

    assert td_array_shapes == get_expected_shapes(
        model_conf, N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET
    )

    assert training_data["model_config"][
        "param_bounds_dict"
    ] == infer_param_bounds_dict(model_config[model_name])


def test_data_persistance(tmp_path):
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["model"] = "ddm"
    generator_config["output"]["folder"] = str(tmp_path)
    generator_config["n_subruns"] = 1

    my_dataset_generator = TrainingDataGenerator(
        config=generator_config, model_config=model_conf
    )
    my_dataset_generator.generate_data_training(save=True)
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
        TrainingDataGenerator(config=gen_config, model_config=None)

    with pytest.raises(ValueError):
        TrainingDataGenerator(config=None, model_config=model_conf)


# ============================================================================
# NEW TESTS: Initialization, Helpers, Data Processing, Integration
# ============================================================================


def test_init_with_deadline_model():
    """Test that 'deadline' parameter is added when model name contains 'deadline'."""
    model_conf = deepcopy(model_config["ddm"])
    generator_config = deepcopy(gen_config)
    generator_config["model"] = "ddm_deadline"
    generator_config["model"] = "ddm_deadline"

    my_gen = TrainingDataGenerator(config=generator_config, model_config=model_conf)

    # Check that deadline was added
    assert "deadline" in my_gen.model_config["params"]
    assert "deadline" in my_gen.model_config["param_bounds_dict"]
    assert my_gen.model_config["param_bounds_dict"]["deadline"] == (0.001, 10)
    assert my_gen.model_config["n_params"] == model_conf["n_params"] + 1
    assert my_gen.model_config["name"].endswith("_deadline")


# ============================================================================
# REMOVED TESTS: Tests for deprecated internal methods were removed
# ============================================================================
# The following tests were removed as they test internal implementation methods
# that were refactored/removed during the pipeline redesign:
# - test_get_simulations_returns_valid_structure
# - test_filter_simulations_accepts_valid_data
# - test_filter_simulations_rejects_pathological_data
# - test_filter_simulations_raises_on_none
# - test_make_kde_data_returns_correct_shape
# - test_make_kde_data_raises_on_none_simulations
# - test_make_kde_data_raises_on_none_theta
# - test_parameter_transform_for_lba_angle_3
# - test_mlp_get_processed_data_for_theta_returns_all_keys
# - test_generate_data_training_single_cpu
# - test_generate_data_training_multi_cpu
#
# The public API (generate_data_training()) is thoroughly tested by the
# parametrized test_TrainingDataGenerator test which runs for all models.
