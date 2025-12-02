# MLflow Configuration Guide

## Overview

The `ssm-simulators` package provides **optional MLflow integration** with full control over tracking and artifact storage through CLI arguments, environment variables, and sensible defaults.

## Installation

MLflow is an **optional dependency**. To use MLflow tracking features:

```bash
# Install with MLflow support
pip install ssm-simulators[mlflow]

# Or with uv
uv pip install ssm-simulators[mlflow]
```

Without the `[mlflow]` extra, the package works normally but MLflow tracking will be disabled even if `--mlflow-run-name` is provided.

## Configuration Priority

For all MLflow settings, the priority order is:
1. **CLI arguments** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

## Available Options

### 1. Tracking URI (`--mlflow-tracking-uri`)

Controls where MLflow stores **metadata** (runs, parameters, metrics).

**Priority:**
- CLI: `--mlflow-tracking-uri`
- Env: `MLFLOW_TRACKING_URI`
- Default: `sqlite:///mlflow.db` (SQLite database in current directory)

**Examples:**
```bash
# SQLite (local, relative path)
--mlflow-tracking-uri "sqlite:///mlflow.db"

# SQLite (absolute path, recommended for shared filesystems)
--mlflow-tracking-uri "sqlite:////shared/project/mlflow.db"

# Remote MLflow server
--mlflow-tracking-uri "http://mlflow-server:5000"

# PostgreSQL (production)
--mlflow-tracking-uri "postgresql://user:pass@host:5432/mlflow"
```

### 2. Artifact Location (`--mlflow-artifact-location`)

Controls where MLflow stores **artifacts** (large files, models, logs).

**Priority:**
- CLI: `--mlflow-artifact-location`
- Env: `MLFLOW_ARTIFACT_LOCATION`
- Default: `./mlruns` (relative to current directory)

**Examples:**
```bash
# Local directory (relative)
--mlflow-artifact-location "./mlflow_artifacts"

# Shared filesystem (absolute path, recommended for clusters)
--mlflow-artifact-location "/shared/project/mlflow_artifacts"

# S3 bucket (cloud storage)
--mlflow-artifact-location "s3://my-bucket/mlflow-artifacts"

# Azure Blob Storage
--mlflow-artifact-location "wasbs://container@account.blob.core.windows.net/mlflow"
```

### 3. Experiment Name (`--mlflow-experiment-name`)

Logical grouping for related runs.

```bash
--mlflow-experiment-name "ddm-data-generation"
```

### 4. Run Name (`--mlflow-run-name`)

Identifier for a specific execution.

```bash
--mlflow-run-name "ddm-run-001"
```

## Usage Examples

### Local Development (Default)

```bash
ssms-generate \
  --output ./data \
  --n-files 10 \
  --mlflow-run-name "local-test" \
  --mlflow-experiment-name "dev-testing"
```

**Result:**
- Metadata: `./mlflow.db` (SQLite)
- Artifacts: `./mlruns/`

### Cluster with Shared Filesystem

```bash
ssms-generate \
  --output /shared/data \
  --n-files 1000 \
  --mlflow-tracking-uri "sqlite:////shared/mlflow/tracking.db" \
  --mlflow-artifact-location "/shared/mlflow/artifacts" \
  --mlflow-run-name "cluster-job-${SLURM_JOB_ID}" \
  --mlflow-experiment-name "production-ddm"
```

**Result:**
- Metadata: `/shared/mlflow/tracking.db` (SQLite on shared FS)
- Artifacts: `/shared/mlflow/artifacts/`
- Multiple nodes can read/write simultaneously

### Using Environment Variables

```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/mlflow/artifacts"

ssms-generate \
  --output ./data \
  --n-files 10 \
  --mlflow-run-name "env-configured" \
  --mlflow-experiment-name "my-experiment"
```

### Remote MLflow Server

```bash
ssms-generate \
  --output ./data \
  --n-files 10 \
  --mlflow-tracking-uri "http://mlflow.example.com:5000" \
  --mlflow-artifact-location "s3://my-mlflow-bucket/artifacts" \
  --mlflow-run-name "remote-run" \
  --mlflow-experiment-name "cloud-experiment"
```

## Best Practices

### For Local Development
- ✅ Use defaults (SQLite + `./mlruns`)
- ✅ Quick setup, no configuration needed
- ⚠️ Not suitable for teams or clusters

### For Shared Clusters (Slurm, etc.)
- ✅ Use **absolute paths** for both tracking URI and artifact location
- ✅ Place both on **shared filesystem** visible to all nodes
- ✅ SQLite works well for moderate concurrency
- ✅ Example:
  ```bash
  --mlflow-tracking-uri "sqlite:////nfs/project/mlflow.db"
  --mlflow-artifact-location "/nfs/project/mlflow_artifacts"
  ```

### For Production (High Concurrency)
- ✅ Use **database backend** (PostgreSQL, MySQL) for tracking
- ✅ Use **cloud storage** (S3, Azure Blob) for artifacts
- ✅ Set up dedicated MLflow server
- ✅ Example:
  ```bash
  --mlflow-tracking-uri "postgresql://mlflow:pass@db-server/mlflow"
  --mlflow-artifact-location "s3://company-mlflow/artifacts"
  ```

## Architecture

### With SQLite Backend
```
MLflow Setup:
├── Tracking URI: sqlite:///path/to/mlflow.db
│   └── Stores: experiments, runs, parameters, metrics, tags
│
└── Artifact Location: /path/to/artifacts/
    └── Stores: large files, models, logs, datasets
```

### Why Separate Storage?

1. **Performance**: SQLite is fast for queries, filesystem/cloud is better for large files
2. **Scalability**: Databases handle concurrent writes better than filesystem
3. **Flexibility**: Mix and match (e.g., local DB + cloud artifacts)

### Implementation Details

When you specify `--mlflow-artifact-location`, the CLI:
1. Checks if the experiment already exists
2. If not, creates it with `mlflow.create_experiment(name, artifact_location=...)`
3. Sets the experiment as active with `mlflow.set_experiment(name)`

This ensures artifacts are stored in your specified location rather than the default `./mlruns`

## Testing

The test suite now properly isolates MLflow artifacts in temporary directories:

```python
@pytest.fixture
def test_mlflow_dir(tmp_path):
    """Create isolated MLflow environment for testing."""
    mlflow_db = tmp_path / "mlflow_test" / "tracking.db"
    artifact_dir = tmp_path / "mlflow_test" / "artifacts"

    tracking_uri = f"sqlite:///{mlflow_db.absolute()}"
    artifact_location = str(artifact_dir.absolute())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test", artifact_location=artifact_location)
```

**Benefits:**
- ✅ No `./mlruns` pollution
- ✅ Automatic cleanup via `tmp_path`
- ✅ Complete isolation between tests

## Troubleshooting

### `./mlruns` directory appears after running tests

**Cause:** Some code is creating MLflow runs without explicitly setting artifact location.

**Solution:** The test suite includes an `autouse` fixture that automatically cleans up any stray `mlruns` directories.

### "Permission denied" on cluster

**Cause:** Using relative paths on shared filesystem, or SQLite database on node-local storage.

**Solution:** Always use **absolute paths** pointing to shared storage:
```bash
--mlflow-tracking-uri "sqlite:////nfs/shared/mlflow.db"
--mlflow-artifact-location "/nfs/shared/mlflow_artifacts"
```

### Concurrent writes fail with SQLite

**Cause:** SQLite has limited concurrent write support.

**Solutions:**
1. For moderate concurrency: Use SQLite on NFS with retry logic (built-in)
2. For high concurrency: Migrate to PostgreSQL/MySQL
3. For very high concurrency: Use remote MLflow server

## Migration from Filesystem Backend

The old filesystem backend (deprecated) used `./mlruns` for everything.

**Old (deprecated):**
```bash
# No configuration, everything in ./mlruns
ssms-generate --output ./data --n-files 10
```

**New (recommended):**
```bash
# Explicit SQLite + artifact location
ssms-generate \
  --output ./data \
  --n-files 10 \
  --mlflow-tracking-uri "sqlite:///mlflow.db" \
  --mlflow-artifact-location "./mlflow_artifacts"
```

**Benefits of migration:**
- ✅ No deprecation warnings
- ✅ Better query performance
- ✅ Easier to backup metadata separately from artifacts
- ✅ Forward compatible with cloud deployments

## Summary

The new configuration system provides:
- ✅ **Full control** over tracking and artifact storage
- ✅ **Flexible configuration** via CLI, env vars, or defaults
- ✅ **Production ready** with support for databases and cloud storage
- ✅ **Clean testing** with proper isolation and cleanup
- ✅ **Future proof** using recommended MLflow practices
