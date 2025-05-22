import io
import yaml
from unittest.mock import patch, MagicMock

import pytest

from ssms.cli.generate import (
    try_gen_folder,
    make_data_generator_configs,
    collect_data_generator_config,
    main,
)


@pytest.fixture
def yaml_config():
    return {
        "GENERATOR_APPROACH": "lan",
        "N_SAMPLES": 1000,
        "DELTA_T": 0.1,
        "MODEL": "ddm",
        "N_PARAMETER_SETS": 10,
        "N_TRAINING_SAMPLES_BY_PARAMETER_SET": 100,
        "N_SUBRUNS": 1,
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

    # Test with custom arguments
    custom_config = make_data_generator_configs(
        model="ddm",
        generator_approach="lan",
        data_generator_arg_dict={"n_samples": 1000},
        model_config_arg_dict={"drift": 0.5},
        save_name="test_config.pkl",
        save_folder=tmp_path,
    )
    assert custom_config["data_config"]["n_samples"] == 1000
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
    assert data_config["n_samples"] == 1000
    assert data_config["model"] == "ddm"
    assert data_config["delta_t"] == 0.1


@patch("ssms.dataset_generators.lan_mlp.data_generator")
def test_main(mock_data_generator, tmp_path, yaml_config):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f)

    # Prepare mock data generator
    mock_generator_instance = MagicMock()
    mock_data_generator.return_value = mock_generator_instance

    # Call main with test configuration
    with patch("typer.echo"), patch("typer.Exit"):
        # Patch the yaml_file parameter to be None
        with patch("ssms.cli.generate.main", wraps=main) as mock_main:
            mock_main(
                config_path=config_path,
                output=tmp_path,
                log_level="info",
            )

    # Verify data generator was called
    mock_data_generator.assert_called_once()
    mock_generator_instance.generate_data_training_uniform.assert_called_once_with(
        save=True, cpn_only=False
    )
    