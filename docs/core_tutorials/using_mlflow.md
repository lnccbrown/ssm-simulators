# MLflow Tutorial for SSM-Simulators

Track and manage your data generation experiments with MLflow.

## 📚 What is MLflow?

MLflow helps you:
- 📊 Track experiments and parameters
- 🔍 Compare configurations
- 📁 Organize generated datasets
- 🔄 Reproduce runs exactly


## 🚀 Quick Start (5 minutes)

**1. Install:**
```bash
pip install ssm-simulators[mlflow]
```

**2. Generate with tracking:**
```bash
generate \
  --output ./data \
  --n-files 10 \
  --mlflow-run-name "my-first-run" \
  --mlflow-experiment-name "ddm-experiments"
```

**3. View results:**
```bash
mlflow ui
# Open http://localhost:5000
```

## 💡 What Gets Tracked?

**Automatically logged:**
- Configuration: model type, samples, parameter sets, estimator
- Results: number of files, total size
- Artifacts: data config, model config, file inventory

## 📖 Usage Examples

### Example 1: Compare Models

```bash
# Test different models
generate --output ./data --n-files 5 \
  --mlflow-run-name "ddm-baseline" \
  --mlflow-experiment-name "model-comparison"

generate --output ./data --n-files 5 \
  --estimator-type kde \
  --mlflow-run-name "ornstein-kde" \
  --mlflow-experiment-name "model-comparison"

# Compare in UI
mlflow ui
```

### Example 2: Dry-Run Validation

```bash
# Validate config without saving data
generate --config-path config.yaml --output ./data \
  --dry-run \
  --mlflow-run-name "validation" \
  --mlflow-experiment-name "testing"

# Then run for real
generate --config-path config.yaml --output ./data \
  --n-files 100 \
  --mlflow-run-name "production" \
  --mlflow-experiment-name "production"
```

### Example 3: Cluster with Shared Filesystem

```bash
#!/bin/bash
#SBATCH --job-name=ssm-datagen

# Use shared filesystem
export MLFLOW_TRACKING_URI="sqlite:////nfs/project/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/nfs/project/mlflow/artifacts"

generate \
  --output /nfs/project/data \
  --n-files 1000 \
  --mlflow-run-name "cluster-job-${SLURM_JOB_ID}" \
  --mlflow-experiment-name "production-data"
```

**Why absolute paths?** All nodes can access the same tracking database and artifacts.

## 🔧 Configuration

Three layers of configuration (priority: CLI > Environment > Defaults):

**1. Defaults (no configuration):**
```bash
generate --output ./data --mlflow-run-name "test"
# Uses: sqlite:///mlflow.db and ./mlruns/
```

**2. Environment variables (set once):**
```bash
export MLFLOW_TRACKING_URI="sqlite:///~/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="~/mlflow/artifacts"
```

**3. CLI arguments (per-run override):**
```bash
generate \
  --mlflow-tracking-uri "sqlite:////shared/mlflow.db" \
  --mlflow-artifact-location "/shared/artifacts" \
  --output ./data \
  --mlflow-run-name "run-001"
```

## 📊 Using the MLflow UI

```bash
mlflow ui
# Opens http://localhost:5000

# Sets up UI with tracking from .db
mlflow server --backend-store-uri <sqlite:////path/to/tracking.db>
```


## 💾 File Storage

**MLflow stores two types of data:**

| Type | What | Location |
|------|------|----------|
| **Metadata** | Experiment/run names, parameters, metrics | `--mlflow-tracking-uri` (SQLite DB) |
| **Artifacts** | Config files, file inventories | `--mlflow-artifact-location` |
| **Data files** | Your .pickle files | `--output` (NOT in MLflow) |

**Example structure:**
```
project/
├── mlflow/
│   ├── tracking.db          ← Metadata (lightweight)
│   └── artifacts/            ← Configs, inventories
└── data/                     ← Your .pickle files
    ├── training_data_001.pickle
    └── training_data_002.pickle
```

## 🗄️ Working with the SQLite Database

### View and Query

**Python API:**
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Search all runs
runs = mlflow.search_runs()
print(runs)

# Search specific experiment
runs = mlflow.search_runs(experiment_names=["my-project"])

# Filter by parameters
runs = mlflow.search_runs(
    filter_string="params.data_model = 'ddm'"
)

# Export to CSV
runs.to_csv("experiment_history.csv")
```

**Command line:**
```bash
# Direct SQLite queries (advanced)
sqlite3 mlflow.db "SELECT name FROM experiments;"
```

### Backup and Migration

```bash
# Backup database
cp mlflow.db mlflow-backup-$(date +%Y%m%d).db

# Move to new machine
tar -czf mlflow-export.tar.gz mlflow/
scp mlflow-export.tar.gz newmachine:~/project/
# Extract and set MLFLOW_TRACKING_URI on new machine
```

## 🎯 Common Use Cases

### Find Runs with Specific Config

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(
    filter_string="params.data_model = 'ddm' AND params.data_n_samples > '1000'"
)
print(f"Found {len(runs)} matching runs")
```

### Recreate a Previous Run

```python
# Find the run
runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'best-run'")
run = runs.iloc[0]

# Extract command
print(f"generate --output {run['params.data_output_folder']} \\")
print(f"  --n-files {int(run['metrics.num_files_generated'])}")
```

### Track Training Pipeline Versions

```bash
# Version 1
generate --output ./train/v1 --n-files 50 \
  --mlflow-run-name "dataset-v1.0" \
  --mlflow-experiment-name "training-pipeline"

# Version 2 (improved)
generate --output ./train/v2 --n-files 50 \
  --mlflow-run-name "dataset-v2.0" \
  --mlflow-experiment-name "training-pipeline"

# Compare in UI to see improvements
```

## ⚙️ Best Practices

### Project Organization

**Recommended structure:**
```bash
# Create organized directories
mkdir -p ~/projects/my-project/mlflow/artifacts

# Set environment (add to ~/.bashrc)
export MLFLOW_TRACKING_URI="sqlite:////$HOME/projects/my-project/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="$HOME/projects/my-project/mlflow/artifacts"
```

### Naming Conventions

- **Experiments**: Group related work (`"ddm-training-v2"` not `"exp1"`)
- **Runs**: Include version/iteration (`"baseline-v1.0"`)
- **Use dry-run**: Validate before large runs

### Cluster Usage

```bash
# Always use absolute paths on shared filesystems
export MLFLOW_TRACKING_URI="sqlite:////nfs/shared/mlflow.db"  # 4 slashes!
export MLFLOW_ARTIFACT_LOCATION="/nfs/shared/artifacts"
```

## 🚀 Quick Reference

```bash
# Minimal command
generate --output ./data --mlflow-run-name "test"

# Full command with all options
generate \
  --config-path config.yaml \
  --output ./data \
  --n-files 10 \
  --dry-run \
  --mlflow-run-name "experiment-001" \
  --mlflow-experiment-name "my-project" \
  --mlflow-tracking-uri "sqlite:///mlflow.db" \
  --mlflow-artifact-location "./mlflow_artifacts" \
  --estimator-type kde

# View experiments
mlflow ui

# Sets up UI with tracking from .db
mlflow server --backend-store-uri <sqlite:////path/to/tracking.db>

# Python queries
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
print(mlflow.search_runs())
"

# Backup
cp mlflow.db mlflow-backup-$(date +%Y%m%d).db
```

## 📚 Complete Workflow Example

```bash
# Setup
export MLFLOW_TRACKING_URI="sqlite:///project_mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"

# 1. Validate config
generate --config-path config.yaml --output ./data \
  --dry-run \
  --mlflow-run-name "validation" \
  --mlflow-experiment-name "my-project"

# 2. Generate training set
generate --config-path config.yaml --output ./data/train \
  --n-files 80 \
  --mlflow-run-name "train-v1" \
  --mlflow-experiment-name "my-project"

# 3. Generate validation set
generate --config-path config.yaml --output ./data/val \
  --n-files 10 \
  --mlflow-run-name "val-v1" \
  --mlflow-experiment-name "my-project"

# 4. Review in UI
mlflow --backend-store-uri sqlite:///project_mlflow.db"
```

## Troubleshooting

```bash
alembic.util.exc.CommandError: Can't locate revision identified by <revision number>
```

### Solution:
```bash
pip install --upgrade mlflow
```
