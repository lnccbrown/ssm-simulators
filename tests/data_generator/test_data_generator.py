import random
from copy import deepcopy

import numpy as np
import pytest

from ssms.config import get_lan_config, model_config
from ssms.dataset_generators.lan_mlp import DataGenerator

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
def test_DataGenerator(model_name, model_conf):
    if model_name in broken_models:
        pytest.skip(f"Skipping broken model: {model_name}")
    if model_name.startswith(slow_prefixes):
        pytest.skip(f"Skipping slow model: {model_name}")

    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = model_name

    my_dataset_generator = DataGenerator(
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
        "param_bounds_dict"
    ] == infer_param_bounds_dict(model_config[model_name])


def test_data_persistance(tmp_path):
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["output_folder"] = str(tmp_path)
    generator_config["n_subruns"] = 1

    my_dataset_generator = DataGenerator(
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
        DataGenerator(generator_config=gen_config, model_config=None)

    with pytest.raises(ValueError):
        DataGenerator(generator_config=None, model_config=model_conf)


# ============================================================================
# NEW TESTS: Initialization, Helpers, Data Processing, Integration
# ============================================================================


def test_init_with_deadline_model():
    """Test that 'deadline' parameter is added when model name contains 'deadline'."""
    model_conf = deepcopy(model_config["ddm"])
    generator_config = deepcopy(gen_config)
    generator_config["model"] = "ddm_deadline"
    generator_config["dgp_list"] = "ddm_deadline"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    # Check that deadline was added
    assert "deadline" in my_gen.model_config["params"]
    assert "deadline" in my_gen.model_config["param_bounds_dict"]
    assert my_gen.model_config["param_bounds_dict"]["deadline"] == (0.001, 10)
    assert my_gen.model_config["n_params"] == model_conf["n_params"] + 1
    assert my_gen.model_config["name"].endswith("_deadline")


def test_get_simulations_returns_valid_structure():
    """Test get_simulations() returns dict with expected keys."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    theta_dict = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3}
    sim = my_gen.get_simulations(theta=theta_dict, random_seed=42)

    assert isinstance(sim, dict)
    assert "rts" in sim
    assert "choices" in sim
    assert "metadata" in sim
    assert isinstance(sim["rts"], np.ndarray)
    assert isinstance(sim["choices"], np.ndarray)


def test_filter_simulations_accepts_valid_data():
    """Test _filter_simulations() returns keep=1 for valid simulations with good parameters."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    # Use config with increased n_samples to get better statistics
    generator_config["n_samples"] = 1000

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    # Use parameters known to produce good variability
    theta_dict = {"v": 2.0, "a": 1.5, "z": 0.5, "t": 0.2}
    sim = my_gen.get_simulations(theta=theta_dict, random_seed=123)

    keep, stats = my_gen._filter_simulations(sim)
    # With these parameters and more samples, should pass filters
    assert keep == 1
    assert isinstance(stats, np.ndarray)
    assert stats.shape == (6,)


def test_filter_simulations_rejects_pathological_data():
    """Test _filter_simulations() returns keep=0 for pathological data."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    # Create fake simulation with all identical RTs (high mode count)
    fake_sim = {
        "rts": np.full(100, 0.5, dtype=np.float32),
        "choices": np.ones(100, dtype=np.int32),
        "metadata": {"possible_choices": [-1, 1]},
    }

    keep, stats = my_gen._filter_simulations(fake_sim)
    assert keep == 0  # Should reject due to mode_cnt_rel being too high


def test_filter_simulations_raises_on_none():
    """Test _filter_simulations() raises ValueError when simulations is None."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    with pytest.raises(ValueError, match="No simulations provided"):
        my_gen._filter_simulations(simulations=None)


def test_make_kde_data_returns_correct_shape():
    """Test _make_kde_data() returns correct shape."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["n_training_samples_by_parameter_set"] = 10

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    # _make_kde_data expects theta_dict values to be arrays, not scalars
    theta_dict = {
        "v": np.array([1.0], dtype=np.float32),
        "a": np.array([1.5], dtype=np.float32),
        "z": np.array([0.5], dtype=np.float32),
        "t": np.array([0.3], dtype=np.float32),
    }
    sim = my_gen.get_simulations(theta=theta_dict, random_seed=42)

    kde_data = my_gen._make_kde_data(simulations=sim, theta=theta_dict)

    # Expected shape: (n_training_samples_by_parameter_set, n_params + 3)
    # For ddm: 4 params + 3 (rt, choice, likelihood) = 7
    assert kde_data.shape == (10, 7)
    assert kde_data.dtype == np.float32


def test_make_kde_data_raises_on_none_simulations():
    """Test _make_kde_data() raises ValueError when simulations is None."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    theta_dict = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3}

    # Phase 1 refactoring: Error message now comes from estimator builder
    with pytest.raises(ValueError, match="KDE estimator requires simulations"):
        my_gen._make_kde_data(simulations=None, theta=theta_dict)


def test_make_kde_data_raises_on_none_theta():
    """Test _make_kde_data() raises ValueError when theta is None."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    theta_dict = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.3}
    sim = my_gen.get_simulations(theta=theta_dict, random_seed=42)

    with pytest.raises(ValueError, match="No theta provided"):
        my_gen._make_kde_data(simulations=sim, theta=None)


def test_parameter_transform_for_lba_angle_3():
    """Test parameter_transform swaps a and z when a <= z for lba_angle_3."""
    model_conf = deepcopy(model_config["lba_angle_3"])
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "lba_angle_3"

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    # Create theta with a < z
    theta_dict = {"v0": 1.0, "v1": 1.2, "v2": 1.1, "a": 0.8, "z": 1.5, "t": 0.3}
    transformed = my_gen.parameter_transform_for_data_gen(theta_dict)

    # After transform, a should be >= z
    assert transformed["a"] >= transformed["z"]
    # Specifically, they should have swapped
    assert transformed["a"] == 1.5
    assert transformed["z"] == 0.8


def test_mlp_get_processed_data_for_theta_returns_all_keys():
    """Test _mlp_get_processed_data_for_theta() returns all expected keys."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["n_training_samples_by_parameter_set"] = 6

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    result = my_gen._mlp_get_processed_data_for_theta((42, 43))

    expected_keys = {
        "lan_data",
        "lan_labels",
        "cpn_data",
        "cpn_labels",
        "cpn_no_omission_data",
        "cpn_no_omission_labels",
        "opn_data",
        "opn_labels",
        "gonogo_data",
        "gonogo_labels",
        "binned_128",
        "binned_256",
        "theta",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["lan_data"], np.ndarray)
    assert isinstance(result["theta"], np.ndarray)


# Phase 4 Note: Tests for _cpn_get_processed_data_for_theta() and cpn_only
# functionality were removed as this functionality was deprecated and removed
# in Phase 4 refactoring to simplify the interface.


def test_generate_data_training_uniform_single_cpu():
    """Test generate_data_training_uniform() with n_cpus=1."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["n_parameter_sets"] = 2
    generator_config["n_subruns"] = 1
    generator_config["n_cpus"] = 1

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    result = my_gen.generate_data_training_uniform(save=False, verbose=False)

    assert "lan_data" in result
    assert "cpn_data" in result
    assert "thetas" in result
    assert result["thetas"].shape[0] == 2


def test_generate_data_training_uniform_multi_cpu():
    """Test generate_data_training_uniform() with n_cpus=2."""
    model_conf = model_config["ddm"]
    generator_config = deepcopy(gen_config)
    generator_config["dgp_list"] = "ddm"
    generator_config["n_parameter_sets"] = 2
    generator_config["n_subruns"] = 1
    generator_config["n_cpus"] = 2

    my_gen = DataGenerator(generator_config=generator_config, model_config=model_conf)

    result = my_gen.generate_data_training_uniform(save=False, verbose=False)

    assert "lan_data" in result
    assert "cpn_data" in result
    assert "thetas" in result
    assert result["thetas"].shape[0] == 2
