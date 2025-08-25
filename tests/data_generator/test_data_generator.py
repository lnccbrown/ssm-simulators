import random
from copy import deepcopy

import numpy as np
import pytest
from expected_shapes import get_expected_shapes

from ssms.config import get_lan_config, model_config
from ssms.dataset_generators.lan_mlp import data_generator

N_PARAMETER_SETS = random.randint(2, 10)
N_TRAINING_SAMPLES_BY_PARAMETER_SET = random.randint(2, 10)
N_PARAMETER_SETS = 1
N_TRAINING_SAMPLES_BY_PARAMETER_SET = (
    6  # seems to need to be at least 6 for nparamsets = 1
)
N_SAMPLES = 4  # lowerbound seems to be 4


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


gen_config = get_lan_config()
gen_config.update(
    _make_gen_config(
        N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET, N_SAMPLES, 1
    )
)

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

    assert (
        td_array_shapes
        == get_expected_shapes(N_PARAMETER_SETS, N_TRAINING_SAMPLES_BY_PARAMETER_SET)[
            model_name
        ]
    )
