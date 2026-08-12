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
        save_folder=str(tmp_path),
    )
    assert custom_config["data_config"]["simulator"]["n_samples"] == 1000
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


def test_generator_approach_persisted_in_data_config():
    """The selecting approach must survive into data_config (provenance)."""
    from ssms.cli.generate import make_data_generator_configs

    config = make_data_generator_configs(model="ddm", generator_approach="lan")
    assert config["data_config"]["generator_approach"] == "lan"

    config_re = make_data_generator_configs(
        model="ddm", generator_approach="ratio_estimator"
    )
    assert config_re["data_config"]["generator_approach"] == "ratio_estimator"


class TestParseMlflowTags:
    def test_single_and_multiple_tags(self):
        from ssms.cli.generate import parse_mlflow_tags

        assert parse_mlflow_tags(["a=1"]) == {"a": "1"}
        assert parse_mlflow_tags(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}

    def test_empty_value_allowed(self):
        from ssms.cli.generate import parse_mlflow_tags

        assert parse_mlflow_tags(["flag="]) == {"flag": ""}

    def test_missing_separator_rejected(self):
        import typer

        from ssms.cli.generate import parse_mlflow_tags

        with pytest.raises(typer.BadParameter):
            parse_mlflow_tags(["notagvalue"])

    def test_empty_key_rejected(self):
        import typer

        from ssms.cli.generate import parse_mlflow_tags

        with pytest.raises(typer.BadParameter):
            parse_mlflow_tags(["=value"])

    def test_reserved_schema_tags_rejected(self):
        import typer

        from ssms.cli.generate import parse_mlflow_tags

        for reserved in ("schema_version=2", "phase=custom"):
            with pytest.raises(typer.BadParameter):
                parse_mlflow_tags([reserved])


# n_cpus: the pipeline section used to be forwarded key-by-key, which silently
# discarded every field except n_parameter_sets and n_subruns.


def _config_from(yaml_dict, tmp_path):
    buffer = io.StringIO()
    yaml.dump(yaml_dict, buffer)
    buffer.seek(0)
    return collect_data_generator_config(yaml_config_path=buffer, base_path=tmp_path)


def test_pipeline_n_cpus_from_yaml_reaches_the_generator_config(tmp_path, yaml_config):
    """The regression this fixes: N_CPUS was accepted and thrown away."""
    yaml_config["PIPELINE"]["N_CPUS"] = 4
    config = _config_from(yaml_config, tmp_path)
    assert config["data_config"]["pipeline"]["n_cpus"] == 4


def test_pipeline_defaults_survive_a_partial_yaml(tmp_path, yaml_config):
    # Forwarding the whole section must not wipe keys the YAML omits.
    config = _config_from(yaml_config, tmp_path)
    pipeline = config["data_config"]["pipeline"]
    assert pipeline["n_parameter_sets"] == 10
    assert pipeline["n_subruns"] == 1
    assert pipeline["n_cpus"] == "all"  # packaged default, untouched


def test_unknown_pipeline_keys_are_forwarded_not_dropped(tmp_path, yaml_config):
    # The point of forwarding the section wholesale: a new key needs no second
    # edit in the CLI bridge to become reachable.
    yaml_config["PIPELINE"]["N_PARAMETER_SETS_REJECTED"] = 7
    config = _config_from(yaml_config, tmp_path)
    assert config["data_config"]["pipeline"]["n_parameter_sets_rejected"] == 7


@pytest.mark.parametrize("value,expected", [("all", "all"), ("8", 8), (None, None)])
def test_parse_n_cpus_accepts_valid_values(value, expected):
    from ssms.cli.generate import parse_n_cpus

    assert parse_n_cpus(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "eight", "2.5"])
def test_parse_n_cpus_rejects_invalid_values(value):
    import typer

    from ssms.cli.generate import parse_n_cpus

    with pytest.raises(typer.BadParameter):
        parse_n_cpus(value)
