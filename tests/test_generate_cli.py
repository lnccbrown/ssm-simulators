import io
import yaml

import pytest

from ssms.cli.generate import (
    try_gen_folder,
    make_data_generator_configs,
    collect_data_generator_config,
)


@pytest.fixture
def yaml_config():
    return {
        "GENERATOR_APPROACH": "lan",
        "MODEL": "ddm",
        "PIPELINE": {
            "N_PARAMETER_SETS": 10,
            "N_SUBRUNS": 1,
        },
        "SIMULATOR": {
            "N_SAMPLES": 1000,
            "DELTA_T": 0.1,
        },
        "TRAINING": {
            "N_SAMPLES_PER_PARAM": 100,
        },
    }


def test_try_gen_folder(tmp_path):
    # Test creating a folder
    test_folder = tmp_path / "test_folder"
    try_gen_folder(test_folder)
    assert test_folder.exists()
    assert test_folder.is_dir()

    # Test creating nested folders
    test_nested_folder = tmp_path / "parent" / "child"
    try_gen_folder(test_nested_folder)
    assert test_nested_folder.exists()
    assert test_nested_folder.is_dir()

    # Test error when folder is None
    with pytest.raises(ValueError, match="Folder path cannot be None or empty."):
        try_gen_folder(None)

    # Test warning for absolute path when not allowed
    with pytest.warns(UserWarning, match="Absolute folder path provided"):
        try_gen_folder(tmp_path.resolve(), allow_abs_path_folder_generation=False)


def test_make_data_generator_configs(tmp_path):
    # Test default configuration
    result = make_data_generator_configs()
    assert isinstance(result, dict), "Default configuration should return a dictionary"
    assert "model_config" in result
    assert "data_config" in result

    # Test with custom arguments (nested structure)
    custom_config = make_data_generator_configs(
        model="ddm",
        generator_approach="lan",
        data_generator_nested_dict={"simulator": {"n_samples": 1000}},
        model_config_arg_dict={"drift": 0.5},
        save_name="test_config.pkl",
        save_folder=tmp_path,
    )
    assert custom_config["data_config"]["simulator"]["n_samples"] == 1000
    assert custom_config["model_config"]["drift"] == 0.5
    assert (tmp_path / "test_config.pkl").exists()


def test_collect_data_generator_config(tmp_path, yaml_config):
    # Use StringIO to create an in-memory file-like object
    yaml_buffer = io.StringIO()
    yaml.dump(yaml_config, yaml_buffer)
    yaml_buffer.seek(0)  # Reset buffer position to the start

    # Test configuration retrieval
    config_dict = collect_data_generator_config(
        yaml_config_path=yaml_buffer, base_path=tmp_path
    )

    data_config = config_dict["data_config"]
    assert data_config["simulator"]["n_samples"] == 1000
    assert data_config["model"] == "ddm"
    assert data_config["simulator"]["delta_t"] == 0.1


# Phase 2: Tests for estimator_type configuration


def test_collect_config_with_estimator_type_yaml(tmp_path):
    """Test that estimator_type from YAML is correctly parsed."""
    yaml_config = {
        "GENERATOR_APPROACH": "lan",
        "MODEL": "ddm",
        "PIPELINE": {
            "N_PARAMETER_SETS": 10,
            "N_SUBRUNS": 1,
        },
        "SIMULATOR": {
            "N_SAMPLES": 1000,
            "DELTA_T": 0.1,
        },
        "TRAINING": {
            "N_SAMPLES_PER_PARAM": 100,
        },
        "ESTIMATOR": {
            "TYPE": "kde",
        },
    }

    yaml_buffer = io.StringIO()
    yaml.dump(yaml_config, yaml_buffer)
    yaml_buffer.seek(0)

    config_dict = collect_data_generator_config(
        yaml_config_path=yaml_buffer, base_path=tmp_path
    )

    assert "estimator" in config_dict["data_config"]
    assert config_dict["data_config"]["estimator"]["type"] == "kde"


def test_collect_config_without_estimator_type(tmp_path):
    """Test that config works when estimator_type is not specified."""
    yaml_config = {
        "GENERATOR_APPROACH": "lan",
        "MODEL": "ddm",
        "PIPELINE": {
            "N_PARAMETER_SETS": 10,
            "N_SUBRUNS": 1,
        },
        "SIMULATOR": {
            "N_SAMPLES": 1000,
            "DELTA_T": 0.1,
        },
        "TRAINING": {
            "N_SAMPLES_PER_PARAM": 100,
        },
    }

    yaml_buffer = io.StringIO()
    yaml.dump(yaml_config, yaml_buffer)
    yaml_buffer.seek(0)

    config_dict = collect_data_generator_config(
        yaml_config_path=yaml_buffer, base_path=tmp_path
    )

    # estimator section should still exist (from defaults), but might not have custom type
    # The default config will have estimator section, just not overridden from YAML
    assert "estimator" in config_dict["data_config"]


def test_collect_config_estimator_type_case_insensitive(tmp_path):
    """Test that estimator_type is converted to lowercase."""
    yaml_config = {
        "GENERATOR_APPROACH": "lan",
        "MODEL": "ddm",
        "PIPELINE": {
            "N_PARAMETER_SETS": 10,
            "N_SUBRUNS": 1,
        },
        "SIMULATOR": {
            "N_SAMPLES": 1000,
            "DELTA_T": 0.1,
        },
        "TRAINING": {
            "N_SAMPLES_PER_PARAM": 100,
        },
        "ESTIMATOR": {
            "TYPE": "KDE",  # Uppercase
        },
    }

    yaml_buffer = io.StringIO()
    yaml.dump(yaml_config, yaml_buffer)
    yaml_buffer.seek(0)

    config_dict = collect_data_generator_config(
        yaml_config_path=yaml_buffer, base_path=tmp_path
    )

    # Should be lowercased
    assert config_dict["data_config"]["estimator"]["type"] == "kde"


# TODO: test app object and CLI commands with --estimator-type flag.
# This requires using typer.testing.CliRunner, which is harder to do than with argparse
