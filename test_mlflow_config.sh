#!/bin/bash
# Quick test script to verify MLflow configuration works

set -e

echo "=== Testing MLflow Configuration ==="
echo ""

# Create test directory
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

# Test 1: Run with explicit tracking URI and artifact location
echo ""
echo "Test 1: Explicit configuration via CLI"
cd "$TEST_DIR"
python -m ssms.cli.generate \
  --output "$TEST_DIR/data" \
  --n-files 1 \
  --mlflow-tracking-uri "sqlite:///$TEST_DIR/test_mlflow.db" \
  --mlflow-artifact-location "$TEST_DIR/test_artifacts" \
  --mlflow-run-name "test-run-1" \
  --mlflow-experiment-name "test-experiment"

# Check that files were created in the right places
if [ -f "$TEST_DIR/test_mlflow.db" ]; then
    echo "✅ SQLite database created at specified location"
else
    echo "❌ SQLite database NOT found"
    exit 1
fi

if [ -d "$TEST_DIR/test_artifacts" ]; then
    echo "✅ Artifact directory created at specified location"
else
    echo "❌ Artifact directory NOT found"
    exit 1
fi

if [ -d "$TEST_DIR/mlruns" ]; then
    echo "❌ WARNING: mlruns directory was created (should not happen)"
else
    echo "✅ No mlruns directory (correct)"
fi

echo ""
echo "Test 2: Configuration via environment variables"
export MLFLOW_TRACKING_URI="sqlite:///$TEST_DIR/env_mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="$TEST_DIR/env_artifacts"

python -m ssms.cli.generate \
  --output "$TEST_DIR/data2" \
  --n-files 1 \
  --mlflow-run-name "test-run-2" \
  --mlflow-experiment-name "test-experiment-env"

if [ -f "$TEST_DIR/env_mlflow.db" ]; then
    echo "✅ SQLite database created from env var"
else
    echo "❌ SQLite database from env var NOT found"
    exit 1
fi

if [ -d "$TEST_DIR/env_artifacts" ]; then
    echo "✅ Artifact directory created from env var"
else
    echo "❌ Artifact directory from env var NOT found"
    exit 1
fi

# Cleanup
echo ""
echo "Cleaning up test directory: $TEST_DIR"
rm -rf "$TEST_DIR"

echo ""
echo "=== All tests passed! ==="
