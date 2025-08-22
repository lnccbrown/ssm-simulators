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
models_to_skip = [
    "lba_3_vs_constraint",  # broken
    "lba_angle_3_vs_constraint",  # broken
    "dev_rlwm_lba_race_v2",  # broken
]

slow_models = ["race_3", "race_no_bias_3", "race_no_z_3"]
slow_prefixes = (
    "race",  # slow
    "dev_rlwm",
    "lba3",
    "lba_angle_3",
    "lca",
    "ddm_par2",
    "ddm_seq2",
    "ddm_mic2",
    "tradeoff",
)

ok_model_config = [
    item for item in model_config.items() if item[0] not in models_to_skip
]
# TODO: Remove this once data generator is optimized for slow models (#113)
# subset_size = 1 + len(ok_model_config) // 10
# ok_model_config = random.sample(ok_model_config, subset_size)


@pytest.mark.parametrize("model_name,model_conf", ok_model_config)
def test_data_generator(model_name, model_conf):
    if model_name in slow_models or model_name.startswith(slow_prefixes):
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
