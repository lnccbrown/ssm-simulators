"""Tests for MLflow integration in the data generation CLI."""

import json
import shutil
from pathlib import Path

import pytest

# Try to import mlflow, skip all tests if not available
mlflow = pytest.importorskip("mlflow")

from ssms.cli.generate import make_data_generator_configs
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator


def set_experiment_with_artifact_location(experiment_name, artifact_location):
    """Helper to set experiment with artifact location (MLflow 3.6+ compatible)."""
    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Create new experiment with artifact location
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Clean up any MLflow artifacts after each test."""
    # Store original tracking URI
    original_uri = mlflow.get_tracking_uri()

    yield

    # Clean up any stray mlruns directory
    mlruns_path = Path.cwd() / "mlruns"
    if mlruns_path.exists():
        shutil.rmtree(mlruns_path)

    # Reset to original URI (or default)
    try:
        mlflow.set_tracking_uri(original_uri)
    except Exception:
        pass


@pytest.fixture
def test_mlflow_dir(tmp_path):
    """Create a temporary MLflow tracking directory with SQLite backend."""
    # Create isolated directories with unique names to avoid confusion
    mlflow_root = tmp_path / "mlflow_test"
    mlflow_root.mkdir()

    mlflow_db = mlflow_root / "tracking.db"
    artifact_dir = mlflow_root / "artifacts"
    artifact_dir.mkdir()

    # Use absolute path to prevent MLflow from scanning current directory
    tracking_uri = f"sqlite:///{mlflow_db.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Store artifact location for tests that need it
    artifact_location = str(artifact_dir.absolute())

    # Return tracking URI, artifact location, and tmp_path for cleanup
    yield {
        "tracking_uri": tracking_uri,
        "artifact_location": artifact_location,
        "tmp_path": tmp_path,
    }
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal configuration for fast testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    config = make_data_generator_configs(
        model="ddm",
        generator_approach="lan",
        data_generator_nested_dict={
            "simulator": {
                "n_samples": 100,  # Small for speed
            },
            "pipeline": {
                "n_parameter_sets": 2,
                "n_subruns": 1,
                "n_cpus": 1,  # Avoid psutil issues
            },
            "training": {
                "n_samples_per_param": 10,
            },
            "output": {
                "folder": str(output_dir),
            },
        },
    )
    return config


@pytest.fixture
def mlflow_experiment(test_mlflow_dir):
    """Set up MLflow experiment and return experiment name."""
    # tracking_uri already set by test_mlflow_dir fixture
    experiment_name = "test-data-generation"
    mlflow.set_experiment(experiment_name)
    return experiment_name


class TestMLflowIntegration:
    """Test suite for MLflow integration with data generation."""

    def test_data_generation_without_mlflow(self, minimal_config):
        """Test that data generation works without MLflow (baseline)."""
        gen = TrainingDataGenerator(
            config=minimal_config["data_config"],
            model_config=minimal_config["model_config"],
        )

        data = gen.generate_data_training(save=True, verbose=False)

        # Verify data structure
        assert isinstance(data, dict)
        assert "theta" in data  # Updated key name
        assert "lan_data" in data

        # Verify file was created
        output_folder = Path(minimal_config["data_config"]["output"]["folder"])
        pickle_files = list(output_folder.rglob("*.pickle"))
        assert len(pickle_files) == 1, "Expected exactly one pickle file"

    def test_mlflow_run_creation(self, test_mlflow_dir):
        """Test that MLflow runs can be created and logged."""
        # test_mlflow_dir is now a dict with tracking_uri and artifact_location
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]

        set_experiment_with_artifact_location("test-experiment", artifact_location)

        with mlflow.start_run(run_name="test-run") as run:
            mlflow.log_param("test_param", "value")
            mlflow.log_metric("test_metric", 42)
            run_id = run.info.run_id

        # Verify run was created
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        retrieved_run = client.get_run(run_id)

        assert retrieved_run.data.params["test_param"] == "value"
        assert retrieved_run.data.metrics["test_metric"] == 42

    def test_data_generation_with_mlflow_logging(self, minimal_config, test_mlflow_dir):
        """Test data generation with MLflow artifact logging."""
        # test_mlflow_dir is now a dict with tracking_uri and artifact_location
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]

        set_experiment_with_artifact_location("test-experiment", artifact_location)

        # Get output directory from config
        test_output_dir = Path(minimal_config["data_config"]["output"]["folder"])

        # Track files before generation
        existing_files = set(test_output_dir.rglob("*.pickle"))

        # Generate data within MLflow run
        with mlflow.start_run(run_name="test-data-gen") as run:
            run_id = run.info.run_id

            # Log configuration
            mlflow.log_dict(minimal_config["data_config"], "data_config.json")
            mlflow.log_dict(minimal_config["model_config"], "model_config.json")
            mlflow.log_param("data_output_folder", str(test_output_dir))

            # Generate data
            gen = TrainingDataGenerator(
                config=minimal_config["data_config"],
                model_config=minimal_config["model_config"],
            )
            gen.generate_data_training(save=True, verbose=False)

            # Find newly generated files
            current_files = set(test_output_dir.rglob("*.pickle"))
            new_files = sorted(current_files - existing_files)

            # Create file inventory
            generated_files = [
                {
                    "filename": f.name,
                    "relative_path": str(f.relative_to(test_output_dir)),
                    "size_bytes": f.stat().st_size,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                }
                for f in new_files
            ]

            file_inventory = {
                "num_files": len(generated_files),
                "total_size_mb": round(sum(f["size_mb"] for f in generated_files), 2),
                "files": generated_files,
            }

            # Log inventory
            mlflow.log_dict(file_inventory, "generated_files_inventory.json")
            mlflow.log_metric("num_files_generated", len(generated_files))
            mlflow.log_metric("total_data_size_mb", file_inventory["total_size_mb"])

        # Verify artifacts were logged
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]

        assert "data_config.json" in artifact_names
        assert "model_config.json" in artifact_names
        assert "generated_files_inventory.json" in artifact_names

        # Verify metrics were logged
        retrieved_run = client.get_run(run_id)
        assert "num_files_generated" in retrieved_run.data.metrics
        assert retrieved_run.data.metrics["num_files_generated"] == 1

    def test_mlflow_artifact_retrieval(self, minimal_config, test_mlflow_dir):
        """Test that logged artifacts can be retrieved via MLflow API."""
        # test_mlflow_dir is now a dict with tracking_uri and artifact_location
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]

        set_experiment_with_artifact_location("test-experiment", artifact_location)

        # Generate and log data
        with mlflow.start_run(run_name="test-retrieval") as run:
            run_id = run.info.run_id

            gen = TrainingDataGenerator(
                config=minimal_config["data_config"],
                model_config=minimal_config["model_config"],
            )
            gen.generate_data_training(save=True, verbose=False)

            # Log a simple inventory
            file_inventory = {
                "num_files": 1,
                "files": [{"filename": "test.pickle"}],
            }
            mlflow.log_dict(file_inventory, "generated_files_inventory.json")

        # Retrieve via MLflow API
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        downloaded_path = client.download_artifacts(
            run_id, "generated_files_inventory.json"
        )

        # Verify contents
        with open(downloaded_path, "r") as f:
            retrieved_inventory = json.load(f)

        assert retrieved_inventory["num_files"] == 1
        assert len(retrieved_inventory["files"]) == 1

    def test_file_inventory_accuracy(self, minimal_config):
        """Test that file inventory accurately tracks generated files."""
        # Get output directory from config
        test_output_dir = Path(minimal_config["data_config"]["output"]["folder"])

        # Track before
        existing_files = set(test_output_dir.rglob("*.pickle"))

        # Generate multiple files
        gen = TrainingDataGenerator(
            config=minimal_config["data_config"],
            model_config=minimal_config["model_config"],
        )

        n_files = 3
        for _ in range(n_files):
            gen.generate_data_training(save=True, verbose=False)

        # Track after
        current_files = set(test_output_dir.rglob("*.pickle"))
        new_files = current_files - existing_files

        # Verify correct number
        assert len(new_files) == n_files

        # Verify all files are readable
        for file_path in new_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 0

    def test_mlflow_experiment_separation(self, test_mlflow_dir):
        """Test that different experiments are properly separated."""
        # test_mlflow_dir is now a dict with tracking_uri and artifact_location
        artifact_location = test_mlflow_dir["artifact_location"]

        # Create two experiments
        exp1_name = "experiment-1"
        exp2_name = "experiment-2"

        set_experiment_with_artifact_location(exp1_name, artifact_location)
        with mlflow.start_run(run_name="run-1"):
            mlflow.log_param("experiment", "1")

        set_experiment_with_artifact_location(exp2_name, artifact_location)
        with mlflow.start_run(run_name="run-2"):
            mlflow.log_param("experiment", "2")

        # Verify separation
        exp1 = mlflow.get_experiment_by_name(exp1_name)
        exp2 = mlflow.get_experiment_by_name(exp2_name)

        assert exp1.experiment_id != exp2.experiment_id

        runs1 = mlflow.search_runs(experiment_ids=[exp1.experiment_id])
        runs2 = mlflow.search_runs(experiment_ids=[exp2.experiment_id])

        assert len(runs1) == 1
        assert len(runs2) == 1
        assert runs1.iloc[0]["params.experiment"] == "1"
        assert runs2.iloc[0]["params.experiment"] == "2"

    def test_mlflow_nested_directory_structure(self, test_mlflow_dir):
        """Test that MLflow SQLite backend stores data correctly."""
        # test_mlflow_dir is now a dict with tracking_uri and artifact_location
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        tmp_path = test_mlflow_dir["tmp_path"]

        set_experiment_with_artifact_location("test-experiment", artifact_location)

        with mlflow.start_run(run_name="test-structure") as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            mlflow.log_param("test", "value")

        # With SQLite backend, verify data is stored correctly via API
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        retrieved_run = client.get_run(run_id)

        # Verify run data is accessible
        assert retrieved_run.info.run_id == run_id
        assert retrieved_run.info.experiment_id == experiment_id
        assert retrieved_run.data.params["test"] == "value"

        # Verify SQLite database file exists
        mlflow_root = tmp_path / "mlflow_test"
        db_file = mlflow_root / "tracking.db"
        assert db_file.exists(), "SQLite database should exist"


# Integration test that mimics the full CLI workflow
def test_full_cli_workflow_simulation(tmp_path):
    """Simulate the full CLI workflow with MLflow integration."""
    # Create isolated directories with unique names to avoid confusion
    mlflow_root = tmp_path / "mlflow_test"
    mlflow_root.mkdir()

    mlflow_db = mlflow_root / "tracking.db"
    artifact_dir = mlflow_root / "artifacts"
    output_dir = tmp_path / "output"

    artifact_dir.mkdir()
    output_dir.mkdir()

    # Use absolute path to prevent MLflow from scanning current directory
    tracking_uri = f"sqlite:///{mlflow_db.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "integration-test"
    artifact_location = str(artifact_dir.absolute())
    set_experiment_with_artifact_location(experiment_name, artifact_location)

    # Simulate orchestrator creating parent run
    with mlflow.start_run(run_name="parent-run"):
        mlflow.log_param("job_type", "data_generation")

    # Simulate CLI resuming/creating child run
    with mlflow.start_run(run_name="data-gen-worker"):
        # Create minimal config
        config = make_data_generator_configs(
            model="ddm",
            generator_approach="lan",
            data_generator_nested_dict={
                "simulator": {
                    "n_samples": 50,
                },
                "pipeline": {
                    "n_parameter_sets": 1,
                    "n_subruns": 1,
                    "n_cpus": 1,
                },
                "training": {
                    "n_samples_per_param": 5,
                },
                "output": {
                    "folder": str(output_dir),
                },
            },
        )

        # Generate data
        gen = TrainingDataGenerator(
            config=config["data_config"],
            model_config=config["model_config"],
        )
        gen.generate_data_training(save=True, verbose=False)

        # Log as CLI would
        mlflow.log_param("data_output_folder", str(output_dir))
        mlflow.log_dict(config["data_config"], "data_config.json")
        mlflow.log_dict(config["model_config"], "model_config.json")

    # Verify both runs exist
    runs = mlflow.search_runs()
    assert len(runs) == 2

    # Verify artifacts
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    # Access DataFrame column using bracket notation
    worker_runs = runs[runs["tags.mlflow.runName"] == "data-gen-worker"]
    assert len(worker_runs) == 1

    artifacts = client.list_artifacts(worker_runs.iloc[0]["run_id"])
    artifact_names = [a.path for a in artifacts]
    assert "data_config.json" in artifact_names
