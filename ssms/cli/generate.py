#!/usr/bin/env -S uv run --script

import logging
import pickle
import warnings
import yaml
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from pprint import pformat

import typer

import ssms
from ssms.config import get_default_generator_config, model_config as _model_config

app = typer.Typer()


def try_gen_folder(
    folder: str | Path | None = None, allow_abs_path_folder_generation: bool = True
) -> None:
    """Function to generate a folder from a string. If the folder already exists, it will not be generated.

    Arguments
    ---------
        folder (str):
            The folder string to generate.
        allow_abs_path_folder_generation (bool):
            If True, the folder string is treated as an absolute path.
            If False, the folder string is treated as a relative path.
    """
    if not folder:
        raise ValueError("Folder path cannot be None or empty.")

    folder_path = Path(folder)

    # Check if the path is absolute and if absolute path generation is allowed
    if folder_path.is_absolute() and not allow_abs_path_folder_generation:
        warnings.warn(
            "Absolute folder path provided, but allow_abs_path_folder_generation is False. "
            "No folders will be generated."
        )
        return

    try:
        # Create the folder and any necessary parent directories
        folder_path.mkdir(parents=True, exist_ok=True)
        logging.info("Folder %s created or already exists.", folder_path)
    except Exception as e:
        logging.error("Error creating folder '%s': %s", folder, e)


def make_data_generator_configs(
    model="ddm",
    generator_approach="lan",
    data_generator_arg_dict={},
    model_config_arg_dict={},
    save_name=None,
    save_folder="",
):
    # Load copy of the respective model's config dict from ssms
    _no_deadline_model = model.split("_deadline")[0]
    model_config = deepcopy(_model_config[_no_deadline_model])

    # Load data_generator_config dicts
    data_config = get_default_generator_config(generator_approach)
    data_config["model"] = model
    data_config.update(data_generator_arg_dict)
    model_config.update(model_config_arg_dict)

    config_dict = {"model_config": model_config, "data_config": data_config}

    if save_name:
        try_gen_folder(save_folder)
        output_file = Path(save_folder) / save_name
        logging.info("Saving config to: %s", output_file)
        with open(output_file, "wb") as f:
            pickle.dump(config_dict, f)
        logging.info("Config saved successfully.")
    return config_dict


def parse_dict_as_namedtuple(d: dict, to_lowercase: bool = True):
    """Convert a dictionary to a named tuple."""
    d = {k.lower() if to_lowercase else k: v for k, v in d.items()}
    return namedtuple("Config", d.keys())(**d)


def _make_data_folder_path(base_path: str | Path, basic_config: namedtuple) -> Path:
    training_data_folder = (
        Path(base_path)
        / "data/training_data"
        / basic_config.generator_approach
        / f"training_data_n_samples_{basic_config.n_samples}_dt_{basic_config.delta_t}"
        / basic_config.model
    )

    return training_data_folder


def get_basic_config_from_yaml(
    yaml_config_path: str | Path, base_path: str | Path = None
):
    """Load the basic configuration from a YAML file."""
    # Handle both file paths and file-like objects (makes mock testing easier)
    if hasattr(yaml_config_path, "read"):
        # If it's a file-like object, read directly
        basic_config_from_yaml = yaml.safe_load(yaml_config_path)
    else:
        # If it's a file path, open and read
        with open(yaml_config_path, "rb") as f:
            basic_config_from_yaml = yaml.safe_load(f)
    bc = parse_dict_as_namedtuple(basic_config_from_yaml)
    training_data_folder = _make_data_folder_path(base_path=base_path, basic_config=bc)
    return bc, training_data_folder


def collect_data_generator_config(
    yaml_config_path=None, base_path=None, extra_configs={}
):
    """Get the data generator configuration from a YAML file."""
    bc, training_data_folder = get_basic_config_from_yaml(
        yaml_config_path, base_path=base_path
    )

    data_generator_arg_dict = {
        "output_folder": training_data_folder,
        "model": bc.model,
        "n_samples": bc.n_samples,
        "n_parameter_sets": bc.n_parameter_sets,
        "delta_t": bc.delta_t,
        "n_training_samples_by_parameter_set": bc.n_training_samples_by_parameter_set,
        "n_subruns": bc.n_subruns,
        "cpn_only": True if (bc.generator_approach == "cpn") else False,
    }

    config_dict = make_data_generator_configs(
        model=bc.model,  # TODO: model is already set in data_generator_arg_dict
        generator_approach=bc.generator_approach,
        data_generator_arg_dict=data_generator_arg_dict,
        model_config_arg_dict=extra_configs,
        save_name=None,
        save_folder=None,
    )
    return config_dict


log_level_option = typer.Option(
    "WARNING",
    "--log-level",
    "-l",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    case_sensitive=False,
    show_default=True,
    rich_help_panel="Logging",
    metavar="LEVEL",
    autocompletion=lambda: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)


@app.command()
def main(
    config_path: Path = typer.Option(..., help="Path to the YAML configuration file."),
    output: Path = typer.Option(..., help="Path to the output directory."),
    log_level: str = log_level_option,
):
    """
    Generate data using the specified configuration.
    """
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Casting config_path to str for now
    # TODO: Fix this in the future
    config_path = str(config_path)
    output = str(output)

    config_dict = collect_data_generator_config(
        yaml_config_path=config_path, base_path=output
    )

    logger.debug("GENERATOR CONFIG")
    logger.debug(pformat(config_dict["data_config"]))

    logger.debug("MODEL CONFIG")
    logger.debug(pformat(config_dict["model_config"]))

    # Make the generator
    logger.info("Generating data")
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=config_dict["data_config"],
        model_config=config_dict["model_config"],
    )

    is_cpn = config_dict["data_config"].get("cpn_only", False)
    my_dataset_generator.generate_data_training_uniform(save=True, cpn_only=is_cpn)

    logger.info("Data generation finished")


if __name__ == "__main__":
    app()
