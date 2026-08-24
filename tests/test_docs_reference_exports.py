"""Regression checks for public exports promised by the docs reference."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _all_exports(source_path: str) -> tuple[str, ...]:
    """Return the literal ``__all__`` declared by a package module."""
    tree = ast.parse((ROOT / source_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exports = ast.literal_eval(node.value)
            return tuple(exports)
    raise AssertionError(f"{source_path} does not declare a literal __all__")


def _reference_text(*doc_paths: str) -> str:
    return "\n".join(
        (ROOT / doc_path).read_text(encoding="utf-8") for doc_path in doc_paths
    )


@pytest.mark.parametrize(
    ("source_path", "doc_paths"),
    [
        ("ssms/basic_simulators/__init__.py", ("docs/api/basic_simulators.md",)),
        ("ssms/config/__init__.py", ("docs/api/config.md",)),
        (
            "ssms/dataset_generators/strategies/__init__.py",
            ("docs/api/dataset_generators.md",),
        ),
        ("ssms/external_simulators/__init__.py", ("docs/api/external_simulators.md",)),
        ("ssms/rl/__init__.py", ("docs/api/rlssm.md",)),
        ("ssms/support_utils/__init__.py", ("docs/api/support_utils.md",)),
        (
            "ssms/basic_simulators/parameter_adapters/base.py",
            ("docs/api/parameter_adapters.md",),
        ),
    ],
)
def test_exported_names_appear_in_reference(
    source_path: str, doc_paths: tuple[str, ...]
) -> None:
    """Keep selected public ``__all__`` surfaces visible in rendered reference."""
    reference = _reference_text(*doc_paths)
    missing = [name for name in _all_exports(source_path) if name not in reference]
    assert not missing, f"Missing reference entries for {source_path}: {missing}"


def test_non_root_public_surfaces_appear_in_reference() -> None:
    """Protect public integration, environment, and validation surfaces."""
    expectations = {
        "docs/api/hssm_support.md": ("::: ssms.hssm_support",),
        "docs/api/rlssm.md": (
            "TaskEnvironment",
            "DiscreteChoiceEnvironment",
            "TaskEnvironmentBuilder",
            "Bandit",
            "TaskConfig",
            "register_task",
            "registered_tasks",
            "DataValidationIssue",
            "DataValidationReport",
            "validate_rlssm_data",
        ),
        "docs/api/basic_simulators.md": (
            "OBSERVATION_SCHEMA_VERSION",
            "OMISSION_SENTINEL",
        ),
    }
    for doc_path, names in expectations.items():
        reference = _reference_text(doc_path)
        missing = [name for name in names if name not in reference]
        assert not missing, f"Missing reference entries in {doc_path}: {missing}"


def test_reference_pages_are_in_navigation() -> None:
    """Prevent complete API pages from becoming navigation-only omissions."""
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for page in (
        "api/basic_simulators.md",
        "api/config.md",
        "api/dataset_generators.md",
        "api/external_simulators.md",
        "api/hssm_support.md",
        "api/parameter_adapters.md",
        "api/rlssm.md",
        "api/support_utils.md",
    ):
        assert page in nav
