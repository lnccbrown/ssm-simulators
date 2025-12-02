import io
import yaml
from pathlib import Path
import pytest

from ssms.cli.generate import (
    make_data_generator_configs,
    collect_data_generator_config,
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
        save_folder=str(tmp_path),
    )
    assert custom_config["data_config"]["n_samples"] == 1000
    assert custom_config["model_config"]["drift"] == 0.5
    assert (Path(tmp_path) / "test_config.pkl").exists()


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


# TODO: test app object and CLI commands. Harder to do than with argparse
