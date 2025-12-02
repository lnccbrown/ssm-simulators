#!/bin/bash

cd /Users/afengler/Library/CloudStorage/OneDrive-Personal/proj_ssm_simulators/ssm-simulators

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=./test_mlruns

TEST_RUN_ID=v0-test-0
TEST_EXPERIMENT_NAME=v0-test-experiment

# Test data generation WITHOUT MLflow first
echo "=== Test 1: Generate data without MLflow ==="
uv run python -m ssms.cli.generate \
  --output ./test_mlruns_output/ \
  --n-files 2 \
  --log-level INFO \
  --mlflow-run-name $TEST_RUN_ID \
  --mlflow-experiment-name $TEST_EXPERIMENT_NAME

# Check what was generated
echo "=== Checking generated files ==="
ls -la ./test_output/data/training_data/

# Verify MLflow logged the run
echo "=== Checking MLflow artifacts ==="
ls -la test_mlruns/0/$TEST_RUN_ID/
ls -la test_mlruns/0/$TEST_RUN_ID/artifacts/

echo "=== Tests complete ==="
