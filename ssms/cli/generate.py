#!/usr/bin/env -S uv run --script

import logging
import os
import pickle
from collections import namedtuple
from copy import deepcopy
from importlib.resources import files, as_file
from pathlib import Path
from pprint import pformat

import tqdm
import typer
import yaml

import ssms
from ssms.config import get_default_generator_config, model_config as _model_config

app = typer.Typer(add_completion=False)


# def try_gen_folder(
#     folder: str | Path | None = None, allow_abs_path_folder_generation: bool = True
# ) -> None:
#     """Function to generate a folder from a string. If the folder already exists, it will not be generated.

#     Arguments
#     ---------
#         folder (str):
#             The folder string to generate.
#         allow_abs_path_folder_generation (bool):
#             If True, the folder string is treated as an absolute path.
#             If False, the folder string is treated as a relative path.
#     """
#     if not folder:
#         raise ValueError("Folder path cannot be None or empty.")

#     folder_path = Path(folder)

#     # Check if the path is absolute and if absolute path generation is allowed
#     if folder_path.is_absolute() and not allow_abs_path_folder_generation:
#         warnings.warn(
#             "Absolute folder path provided, but allow_abs_path_folder_generation is False. "
#             "No folders will be generated."
#         )
#         return

#     try:
#         # Create the folder and any necessary parent directories
#         folder_path.mkdir(parents=True, exist_ok=True)
#         logging.info("Folder %s created or already exists.", folder_path)
#     except Exception as e:
#         logging.error("Error creating folder '%s': %s", folder, e)


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
        Path(save_folder).mkdir(parents=True, exist_ok=True)
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

epilog = "Example: `generate --config-path myconfig.yaml --output ./output --n-files 10 --log-level INFO`"


@app.command(epilog=epilog)
def main(  # pragma: no cover
    config_path: Path = typer.Option(None, help="Path to the YAML configuration file."),
    output: Path = typer.Option(..., help="Path to the output directory."),
    n_files: int = typer.Option(
        1,
        "--n-files",
        "-n",
        help="Number of files to generate.",
        min=1,
        show_default=True,
    ),
    mlflow_run_name: str = typer.Option(
        None,
        "--mlflow-run-name",
        help="MLflow Run ID to resume. "
        "If provided, metrics and artifacts will be logged to this run.",
    ),
    mlflow_experiment_name: str = typer.Option(
        None,
        "--mlflow-experiment-name",
        help="MLflow Experiment Name to log to. "
        "If provided, metrics and artifacts will be logged to this experiment.",
    ),
    mlflow_tracking_uri: str = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (e.g., 'sqlite:///path/to/mlflow.db' or 'http://mlflow-server:5000'). "
        "Defaults to MLFLOW_TRACKING_URI env var, then 'sqlite:///mlflow.db'.",
    ),
    mlflow_artifact_location: str = typer.Option(
        None,
        "--mlflow-artifact-location",
        help="Root directory for MLflow artifacts. "
        "Defaults to MLFLOW_ARTIFACT_LOCATION env var, then './mlruns'.",
    ),
    log_level: str = log_level_option,
):
    """
    Generate data using the specified configuration.
    """
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Setup MLflow
    mlflow_active = False
    if mlflow_run_name:
        try:
            import mlflow

            # Set tracking URI with priority: CLI arg > env var > default
            if mlflow_tracking_uri:
                tracking_uri = mlflow_tracking_uri
            else:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI: %s", tracking_uri)

            # Set experiment with optional artifact location
            if mlflow_experiment_name:
                # Determine artifact location with priority: CLI arg > env var > default
                if mlflow_artifact_location:
                    artifact_location = mlflow_artifact_location
                else:
                    artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)

                if artifact_location:
                    # Ensure artifact location is absolute path
                    artifact_location = str(Path(artifact_location).absolute())

                    # Try to get existing experiment, or create new one with artifact location
                    try:
                        experiment = mlflow.get_experiment_by_name(
                            mlflow_experiment_name
                        )
                        if experiment is None:
                            # Create new experiment with artifact location
                            mlflow.create_experiment(
                                mlflow_experiment_name,
                                artifact_location=artifact_location,
                            )
                            logger.info(
                                "Created MLflow experiment: %s (artifacts: %s)",
                                mlflow_experiment_name,
                                artifact_location,
                            )
                        mlflow.set_experiment(mlflow_experiment_name)
                    except Exception as e:
                        logger.warning(
                            "Could not set artifact location: %s. Using default.", e
                        )
                        mlflow.set_experiment(mlflow_experiment_name)
                else:
                    mlflow.set_experiment(mlflow_experiment_name)
                    logger.info(
                        "Set MLflow experiment: %s (default artifact location)",
                        mlflow_experiment_name,
                    )

            mlflow.start_run(run_name=mlflow_run_name)
            mlflow_active = True
            logger.info("Started new MLflow run: %s", mlflow_run_name)
            logger.info("MLflow run ID: %s", mlflow.active_run().info.run_id)
        except ImportError as e:
            logger.warning(
                "MLflow package not installed but --mlflow-run-name provided. Ignoring --mlflow-run-name. Error: %s",
                e,
            )

    if config_path is None:
        logger.warning("No config path provided, using default configuration.")
        with as_file(
            files("ssms.cli") / "config_data_generation.yaml"
        ) as default_config:
            config_path = default_config

    config_dict = collect_data_generator_config(
        yaml_config_path=config_path, base_path=output
    )

    logger.debug("GENERATOR CONFIG")
    logger.debug(pformat(config_dict["data_config"]))

    logger.debug("MODEL CONFIG")
    logger.debug(pformat(config_dict["model_config"]))

    if mlflow_active:
        mlflow.log_params(
            {f"data_{k}": v for k, v in config_dict["data_config"].items()}
        )
        mlflow.log_dict(config_dict["model_config"], "model_config.json")

    # Make the generator
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=config_dict["data_config"],
        model_config=config_dict["model_config"],
    )

    # In generate.py, BEFORE the generation loop (around line 246)
    output_folder = Path(config_dict["data_config"]["output_folder"])
    # training_data_path = output_folder / "data" / "training_data"

    # Capture existing files before generation
    existing_files = set()
    if output_folder.exists():
        existing_files = set(output_folder.rglob("*.pickle"))
        logger.info("Existing files: %s", existing_files)

    is_cpn = config_dict["data_config"].get("cpn_only", False)

    for _ in tqdm.tqdm(
        range(n_files), desc="Generating simulated data files", unit="file"
    ):
        my_dataset_generator.generate_data_training_uniform(save=True, cpn_only=is_cpn)

    # After generation loop, find new files
    newly_generated_files = []
    if output_folder.exists():
        current_files = set(output_folder.rglob("*.pickle"))
        new_files = current_files - existing_files  # Set difference
        newly_generated_files = sorted(new_files)
        logger.info("Newly generated files: %s", newly_generated_files)
    else:
        logger.warning(
            "Output folder does not exist: %s. "
            "Something went wrong with generation, please check the logs",
            output_folder,
        )

    if mlflow_active:
        output_folder = Path(config_dict["data_config"]["output_folder"])

        # Log the output folder path
        mlflow.log_param("data_output_folder", str(output_folder))

        # Capture the list of generated files
        generated_files = []
        if output_folder.exists():
            # Get all pickle files in the training data directory
            generated_files = [
                {
                    "filename": f.name,
                    "relative_path": str(output_folder / str(f.name)),
                    "size_bytes": f.stat().st_size,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                }
                for f in newly_generated_files
            ]
        else:
            logger.warning(
                "No generated files found in output folder: %s", output_folder
            )

        # Log file inventory as a JSON artifact
        file_inventory = {
            "num_files": len(generated_files),
            "total_size_mb": round(sum(f["size_mb"] for f in generated_files), 2),
            "files": generated_files,
        }
        mlflow.log_dict(file_inventory, "generated_files_inventory.json")

        # Log summary metrics
        mlflow.log_metric("num_files_generated", len(generated_files))
        mlflow.log_metric("total_data_size_mb", file_inventory["total_size_mb"])

        # Log configuration files for reproducibility
        mlflow.log_dict(config_dict["data_config"], "data_config.json")
        mlflow.log_dict(config_dict["model_config"], "model_config.json")
        logger.info("Logged %d files to MLflow inventory", len(generated_files))

        mlflow.end_run()

    logger.info("Data generation finished")


if __name__ == "__main__":
    app()
