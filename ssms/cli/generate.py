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


def make_data_generator_configs(
    model="ddm",
    generator_approach="lan",
    data_generator_nested_dict={},
    model_config_arg_dict={},
    save_name=None,
    save_folder="",
):
    # Load copy of the respective model's config dict from ssms
    _no_deadline_model = model.split("_deadline")[0]
    model_config = deepcopy(_model_config[_no_deadline_model])

    # Load data_generator_config dicts (already nested)
    data_config = get_default_generator_config(generator_approach)
    data_config["model"] = model

    # Deep merge nested config sections
    for section, values in data_generator_nested_dict.items():
        if section in data_config and isinstance(values, dict):
            data_config[section].update(values)
        else:
            data_config[section] = values

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
    """Convert a dictionary to a named tuple, handling nested dicts."""

    def convert_value(v):
        if isinstance(v, dict):
            # Recursively convert nested dicts to namedtuples
            v = {
                k.lower() if to_lowercase else k: convert_value(val)
                for k, val in v.items()
            }
            return namedtuple("Config", v.keys())(**v)
        return v

    d = {k.lower() if to_lowercase else k: convert_value(v) for k, v in d.items()}
    return namedtuple("Config", d.keys())(**d)


def _make_data_folder_path(base_path: str | Path, basic_config: namedtuple) -> Path:
    training_data_folder = (
        Path(base_path)
        / "data/training_data"
        / basic_config.generator_approach
        / f"training_data_n_samples_{basic_config.simulator.n_samples}_dt_{basic_config.simulator.delta_t}"
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
    """Get the data generator configuration from a YAML file with nested structure."""
    bc, training_data_folder = get_basic_config_from_yaml(
        yaml_config_path, base_path=base_path
    )

    # Build nested config from YAML sections
    data_generator_nested_dict = {
        "output": {"folder": str(training_data_folder)},
        "model": bc.model,
        "cpn_only": True if (bc.generator_approach == "cpn") else False,
    }

    # Add pipeline config
    if hasattr(bc, "pipeline"):
        data_generator_nested_dict["pipeline"] = {
            "n_parameter_sets": bc.pipeline.n_parameter_sets,
            "n_subruns": bc.pipeline.n_subruns,
        }

    # Add simulator config
    if hasattr(bc, "simulator"):
        data_generator_nested_dict["simulator"] = {
            "n_samples": bc.simulator.n_samples,
            "delta_t": bc.simulator.delta_t,
        }

    # Add training config
    if hasattr(bc, "training"):
        data_generator_nested_dict["training"] = {
            "n_samples_per_param": bc.training.n_samples_per_param,
        }

    # Add estimator config
    if hasattr(bc, "estimator"):
        data_generator_nested_dict["estimator"] = {
            "type": bc.estimator.type.lower(),
        }

    config_dict = make_data_generator_configs(
        model=bc.model,
        generator_approach=bc.generator_approach,
        data_generator_nested_dict=data_generator_nested_dict,
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


def setup_mlflow(  # pragma: no cover
    mlflow_run_name: str,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    mlflow_artifact_location: str,
    logger: logging.Logger,
) -> bool:
    """
    Setup MLflow tracking and start a run.

    Returns:
        bool: True if MLflow is active, False otherwise.
    """
    if not mlflow_run_name:
        return False

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
                    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
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
        logger.info("Started new MLflow run: %s", mlflow_run_name)
        logger.info("MLflow run ID: %s", mlflow.active_run().info.run_id)
        return True
    except ImportError as e:
        logger.warning(
            "MLflow package not installed but --mlflow-run-name provided. Ignoring --mlflow-run-name. Error: %s",
            e,
        )
        return False


def load_config(  # pragma: no cover
    config_path: Path,
    output: Path,
    estimator_type: str,
    logger: logging.Logger,
) -> dict:
    """
    Load and prepare the generator configuration.

    Returns:
        dict: Configuration dictionary with 'data_config' and 'model_config' keys.
    """
    if config_path is None:
        logger.warning("No config path provided, using default configuration.")
        with as_file(
            files("ssms.cli") / "config_data_generation.yaml"
        ) as default_config:
            config_path = default_config

    config_dict = collect_data_generator_config(
        yaml_config_path=config_path, base_path=output
    )

    # Override estimator_type if specified via CLI
    if estimator_type is not None:
        logger.info(f"Overriding estimator_type from CLI: {estimator_type}")
        config_dict["data_config"]["estimator_type"] = estimator_type.lower()

    logger.debug("GENERATOR CONFIG")
    logger.debug(pformat(config_dict["data_config"]))

    logger.debug("MODEL CONFIG")
    logger.debug(pformat(config_dict["model_config"]))

    return config_dict


def create_estimator(  # pragma: no cover
    config_dict: dict,
    logger: logging.Logger,
):
    """
    Create the estimator builder based on the configuration.

    Returns:
        EstimatorBuilder: The configured estimator builder.

    Raises:
        typer.Exit: If estimator creation fails.
    """
    from ssms.dataset_generators.estimator_builders import create_estimator_builder

    try:
        estimator_builder = create_estimator_builder(
            config_dict["data_config"], config_dict["model_config"]
        )
        logger.info(f"Using estimator: {type(estimator_builder).__name__}")
        return estimator_builder
    except (NotImplementedError, ImportError, ValueError) as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


def generate_data(  # pragma: no cover
    config_dict: dict,
    n_files: int,
    dry_run: bool,
    logger: logging.Logger,
) -> tuple[list[Path], Path]:
    """
    Generate training data files.

    Args:
        config_dict: Configuration dictionary
        n_files: Number of files to generate
        dry_run: If True, validate setup but don't save files
        logger: Logger instance

    Returns:
        tuple: (newly_generated_files, output_folder)
    """
    # Make the generator with the new API
    my_dataset_generator = ssms.dataset_generators.lan_mlp.TrainingDataGenerator(
        config=config_dict["data_config"],
        model_config=config_dict["model_config"],
    )

    # Capture existing files before generation (for MLflow tracking)
    output_folder = Path(config_dict["data_config"]["output"]["folder"])
    existing_files = set()
    if output_folder.exists():
        existing_files = set(output_folder.rglob("*.pickle"))
        logger.info("Existing files before generation: %d", len(existing_files))

    if dry_run:
        logger.info("DRY RUN: Validating data generation pipeline...")
        # Generate one dataset without saving to validate the pipeline
        try:
            _ = my_dataset_generator.generate_data_training(save=False)
            logger.info("✓ DRY RUN: Pipeline validation successful!")
            logger.info(
                "✓ DRY RUN: Would generate %d file(s) to: %s", n_files, output_folder
            )
            return [], output_folder
        except Exception as e:
            logger.error("✗ DRY RUN: Pipeline validation failed: %s", e)
            raise typer.Exit(code=1)

    # Generate data (normal mode)
    for _ in tqdm.tqdm(
        range(n_files), desc="Generating simulated data files", unit="file"
    ):
        my_dataset_generator.generate_data_training(save=True)

    # After generation loop, find new files
    newly_generated_files = []
    if output_folder.exists():
        current_files = set(output_folder.rglob("*.pickle"))
        new_files = current_files - existing_files  # Set difference
        newly_generated_files = sorted(new_files)
        logger.info("Newly generated files: %d", len(newly_generated_files))
    else:
        logger.warning(
            "Output folder does not exist: %s. "
            "Something went wrong with generation, please check the logs",
            output_folder,
        )

    return newly_generated_files, output_folder


def log_to_mlflow(  # pragma: no cover
    config_dict: dict,
    newly_generated_files: list[Path],
    output_folder: Path,
    logger: logging.Logger,
):
    """
    Log generation results to MLflow.

    Args:
        config_dict: Configuration dictionary
        newly_generated_files: List of newly generated file paths
        output_folder: Output folder path
        logger: Logger instance
    """
    import mlflow

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
        logger.warning("No generated files found in output folder: %s", output_folder)

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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the pipeline without saving any data. Useful for testing configurations.",
        is_flag=True,
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
    estimator_type: str = typer.Option(
        None,
        "--estimator-type",
        "-e",
        help="Likelihood estimator type ('kde' or 'pyddm'). Overrides YAML config if specified.",
        case_sensitive=False,
    ),
    log_level: str = log_level_option,
):
    """
    Generate data using the specified configuration.

    The estimator-type option allows you to choose between different likelihood
    estimation methods:

    - 'kde': Kernel Density Estimation (default, always available)

    - 'pyddm': Analytical PDF from PyDDM (Phase 3, coming soon)

    If not specified, the estimator type is read from the YAML config file.
    If neither the CLI flag nor YAML config specifies it, defaults to 'kde'.
    """
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Setup MLflow
    mlflow_active = setup_mlflow(
        mlflow_run_name,
        mlflow_experiment_name,
        mlflow_tracking_uri,
        mlflow_artifact_location,
        logger,
    )

    # Load and prepare configuration
    config_dict = load_config(config_path, output, estimator_type, logger)

    # Log initial config to MLflow
    if mlflow_active:
        import mlflow

        mlflow.log_params(
            {f"data_{k}": v for k, v in config_dict["data_config"].items()}
        )
        mlflow.log_dict(config_dict["model_config"], "model_config.json")
        if dry_run:
            mlflow.log_param("dry_run", True)

    # Create and validate estimator
    estimator_builder = create_estimator(config_dict, logger)

    if mlflow_active:
        import mlflow

        mlflow.log_param("estimator_type", type(estimator_builder).__name__)

    # Generate data
    newly_generated_files, output_folder = generate_data(
        config_dict, n_files, dry_run, logger
    )

    # Log results to MLflow (skip in dry-run mode)
    if mlflow_active and not dry_run:
        log_to_mlflow(config_dict, newly_generated_files, output_folder, logger)

    if mlflow_active:
        import mlflow

        mlflow.end_run()

    if dry_run:
        logger.info("DRY RUN complete - no data was saved")
    else:
        logger.info("Data generation finished")


if __name__ == "__main__":
    app()
