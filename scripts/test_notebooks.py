#!/usr/bin/env python
"""Run all Jupyter notebooks in a directory to verify they execute without errors.

Usage:
    python scripts/test_notebooks.py                    # Run all notebooks in docs/
    python scripts/test_notebooks.py docs/core_tutorials  # Run notebooks in specific folder
    python scripts/test_notebooks.py --timeout 600      # Custom timeout (seconds)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_notebook(notebook_path: Path, timeout: int = 300) -> tuple[bool, str]:
    """Execute a notebook and return (success, error_message)."""
    try:
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                "--ExecutePreprocessor.timeout=" + str(timeout),
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Extra buffer for nbconvert overhead
        )
        if result.returncode != 0:
            return False, result.stderr
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds"
    except Exception as e:
        return False, str(e)


def find_notebooks(directory: Path) -> list[Path]:
    """Find all .ipynb files in directory, excluding checkpoints."""
    notebooks = []
    for nb in directory.rglob("*.ipynb"):
        if ".ipynb_checkpoints" not in str(nb):
            notebooks.append(nb)
    return sorted(notebooks)


def main():
    parser = argparse.ArgumentParser(
        description="Test Jupyter notebooks execute correctly"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="docs",
        help="Directory containing notebooks (default: docs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per notebook in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List notebooks without running them",
    )
    args = parser.parse_args()

    # Find project root (where pyproject.toml is)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    target_dir = project_root / args.directory

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    notebooks = find_notebooks(target_dir)

    if not notebooks:
        print(f"No notebooks found in {target_dir}")
        sys.exit(0)

    print(f"Found {len(notebooks)} notebook(s) in {target_dir}:")
    for nb in notebooks:
        print(f"  - {nb.relative_to(project_root)}")
    print()

    if args.dry_run:
        print("Dry run - not executing notebooks")
        sys.exit(0)

    # Run notebooks
    failed = []
    passed = []

    for i, nb in enumerate(notebooks, 1):
        rel_path = nb.relative_to(project_root)
        print(f"[{i}/{len(notebooks)}] Running {rel_path}...", end=" ", flush=True)

        success, error = run_notebook(nb, timeout=args.timeout)

        if success:
            print("OK")
            passed.append(rel_path)
        else:
            print("FAILED")
            print(
                f"    Error: {error[:200]}..."
                if len(error) > 200
                else f"    Error: {error}"
            )
            failed.append((rel_path, error))

    # Summary
    print()
    print("=" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")

    if failed:
        print("\nFailed notebooks:")
        for nb, error in failed:
            print(f"  - {nb}")
        sys.exit(1)
    else:
        print("\nAll notebooks executed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
