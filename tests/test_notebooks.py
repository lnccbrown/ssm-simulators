"""Test that tutorial notebooks execute without errors.

These tests are skipped by default. Run them with:
    pytest --run-notebooks
    pytest --run-notebooks tests/test_notebooks.py  # just notebooks
    pytest --run-notebooks -k "tutorial_configs"    # specific notebook

Note: Notebooks should use `save=False` when generating data to avoid
creating files during tests. Any files created in the notebook's directory
during execution will be cleaned up automatically.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Find project root and notebook directories
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# Directories containing tutorial notebooks
NOTEBOOK_DIRS = [
    DOCS_DIR / "basic_tutorial",
    DOCS_DIR / "core_tutorials",
]

# Timeout for notebook execution (seconds)
NOTEBOOK_TIMEOUT = 600


def get_notebook_paths() -> list[Path]:
    """Discover all notebooks in the docs tutorial directories."""
    notebooks = []
    for notebook_dir in NOTEBOOK_DIRS:
        if not notebook_dir.exists():
            continue
        for nb in notebook_dir.rglob("*.ipynb"):
            if ".ipynb_checkpoints" not in str(nb):
                notebooks.append(nb)
    return sorted(notebooks)


def get_existing_files(directory: Path) -> set[Path]:
    """Get set of all files currently in a directory (recursively)."""
    if not directory.exists():
        return set()
    return {f for f in directory.rglob("*") if f.is_file()}


def cleanup_new_files(directory: Path, files_before: set[Path]) -> None:
    """Remove files created after files_before snapshot.

    Only removes files within the given directory. Safe to call even if
    no new files were created.
    """
    files_after = get_existing_files(directory)
    new_files = files_after - files_before

    # Delete new files
    for new_file in new_files:
        try:
            new_file.unlink()
        except OSError:
            pass  # Ignore cleanup errors (file may already be gone)

    # Clean up empty directories that were created
    # (process deepest first so parent directories can be removed)
    dirs_before = {f.parent for f in files_before} | {directory}
    dirs_after = {f.parent for f in files_after}
    new_dirs = sorted(
        dirs_after - dirs_before,
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for new_dir in new_dirs:
        try:
            if new_dir.is_dir() and not any(new_dir.iterdir()):
                new_dir.rmdir()
        except OSError:
            pass  # Ignore cleanup errors


def run_notebook(
    notebook_path: Path, timeout: int = NOTEBOOK_TIMEOUT
) -> tuple[bool, str]:
    """Execute a notebook using jupyter nbconvert.

    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success, error_message)
    """
    # Use a temporary file for output to avoid modifying the original notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=True) as tmp:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    "--output",
                    tmp.name,
                    f"--ExecutePreprocessor.timeout={timeout}",
                    str(notebook_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # Extra buffer for nbconvert overhead
                check=False,
            )
            if result.returncode != 0:
                return False, f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"Notebook execution timed out after {timeout} seconds"
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e)


# Get notebooks at module load time for parametrization
NOTEBOOKS = get_notebook_paths()


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[nb.stem for nb in NOTEBOOKS],
)
def test_notebook_execution(notebook_path: Path):
    """Execute a notebook and verify it runs without errors.

    Uses jupyter nbconvert to execute the notebook. The original notebook
    is not modified - output goes to a temporary file.

    After execution, any files created in the notebook's directory are
    cleaned up to prevent lingering artifacts.
    """
    notebook_dir = notebook_path.parent

    # Snapshot existing files before running
    files_before = get_existing_files(notebook_dir)

    try:
        success, error = run_notebook(notebook_path)

        if not success:
            pytest.fail(f"Notebook {notebook_path.name} failed to execute:\n{error}")
    finally:
        # Clean up any files created during notebook execution
        cleanup_new_files(notebook_dir, files_before)
