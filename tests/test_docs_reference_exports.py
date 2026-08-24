"""Regression checks for public exports promised by the docs reference."""

from __future__ import annotations

import ast
import importlib
import inspect
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.structure.files import get_files


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"

LEGACY_CONFIG_IDS = frozenset(
    {
        "ssms.config",
        "ssms.config.CopyOnAccessDict",
        "ssms.config.boundary_registry",
        "ssms.config.boundary_registry.BoundaryRegistry",
        "ssms.config.boundary_registry.BoundaryRegistry.get",
        "ssms.config.boundary_registry.BoundaryRegistry.is_registered",
        "ssms.config.boundary_registry.BoundaryRegistry.list_boundaries",
        "ssms.config.boundary_registry.BoundaryRegistry.register",
        "ssms.config.boundary_registry.get_boundary_registry",
        "ssms.config.boundary_registry.register_boundary",
        "ssms.config.config_utils",
        "ssms.config.config_utils.get_nested_config",
        "ssms.config.config_utils.has_nested_structure",
        "ssms.config.drift_registry",
        "ssms.config.drift_registry.DriftRegistry",
        "ssms.config.drift_registry.DriftRegistry.get",
        "ssms.config.drift_registry.DriftRegistry.is_registered",
        "ssms.config.drift_registry.DriftRegistry.list_drifts",
        "ssms.config.drift_registry.DriftRegistry.register",
        "ssms.config.drift_registry.get_drift_registry",
        "ssms.config.drift_registry.register_drift",
        "ssms.config.model_config_builder",
        "ssms.config.model_config_builder.ModelConfigBuilder",
        "ssms.config.model_config_builder.ModelConfigBuilder.add_boundary",
        "ssms.config.model_config_builder.ModelConfigBuilder.add_drift",
        "ssms.config.model_config_builder.ModelConfigBuilder.from_model",
        "ssms.config.model_config_builder.ModelConfigBuilder.from_scratch",
        "ssms.config.model_config_builder.ModelConfigBuilder.get_sampling_transforms",
        "ssms.config.model_config_builder.ModelConfigBuilder.get_simulation_transforms",
        "ssms.config.model_config_builder.ModelConfigBuilder.get_transforms",
        "ssms.config.model_config_builder.ModelConfigBuilder.minimal_config",
        "ssms.config.model_config_builder.ModelConfigBuilder.validate_config",
        "ssms.config.model_config_builder.ModelConfigBuilder.with_deadline",
        "ssms.config.model_registry",
        "ssms.config.model_registry.ModelConfigRegistry",
        "ssms.config.model_registry.ModelConfigRegistry.get",
        "ssms.config.model_registry.ModelConfigRegistry.has_model",
        "ssms.config.model_registry.ModelConfigRegistry.list_models",
        "ssms.config.model_registry.ModelConfigRegistry.register_config",
        "ssms.config.model_registry.ModelConfigRegistry.register_factory",
        "ssms.config.model_registry.get_model_registry",
        "ssms.config.model_registry.register_model_config",
        "ssms.config.model_registry.register_model_config_factory",
    }
)

LEGACY_RLSSM_IDS = frozenset(
    {
        "rlssm-simulation-ssmsrl",
        "quick-start",
        "public-api",
        "model-configuration",
        "derived-decision-process-config-_ssm_config",
        "built-in-rescorla-wagner-learning-processes",
        "built-in-presets",
        "task-environment-protocols",
        "participant-wise-parameters",
        "simulation-modes",
        "data-validation",
        "choice-only-inverse-temperature-softmax-presets",
        "context-fields",
        "assembled-model-inference-integration",
        "hssm-bridge",
        "module-reference",
    }
)

CONFIG_OBJECT_IDS = frozenset(
    {
        "ssms.config.CopyOnAccessDict",
        "ssms.config.boundary_registry.BoundaryRegistry",
        "ssms.config.boundary_registry.BoundaryRegistry.get",
        "ssms.config.boundary_registry.BoundaryRegistry.is_registered",
        "ssms.config.boundary_registry.BoundaryRegistry.list_boundaries",
        "ssms.config.boundary_registry.BoundaryRegistry.register",
        "ssms.config.config_utils",
        "ssms.config.config_utils.get_nested_config",
        "ssms.config.config_utils.has_nested_structure",
        "ssms.config.drift_registry.DriftRegistry",
        "ssms.config.drift_registry.DriftRegistry.get",
        "ssms.config.drift_registry.DriftRegistry.is_registered",
        "ssms.config.drift_registry.DriftRegistry.list_drifts",
        "ssms.config.drift_registry.DriftRegistry.register",
        "ssms.config.model_registry.ModelConfigRegistry",
        "ssms.config.model_registry.ModelConfigRegistry.get",
        "ssms.config.model_registry.ModelConfigRegistry.has_model",
        "ssms.config.model_registry.ModelConfigRegistry.list_models",
        "ssms.config.model_registry.ModelConfigRegistry.register_config",
        "ssms.config.model_registry.ModelConfigRegistry.register_factory",
        "data_generator_config.get_default_generator_config",
        "data_generator_config.get_kde_simulation_filters",
        "data_generator_config.get_lan_kde_config",
        "data_generator_config.get_lan_config",
        "data_generator_config.get_defective_detector_config",
        "data_generator_config.get_ratio_estimator_config",
    }
)

NON_ALL_OBJECT_IDS = frozenset(
    {
        "ssms.hssm_support",
        "ssms.rl.env.TaskEnvironment",
        "ssms.rl.env.DiscreteChoiceEnvironment",
        "ssms.rl.env.TaskEnvironmentBuilder",
        "ssms.rl.env.Bandit",
        "ssms.rl.env.TaskConfig",
        "ssms.rl.env.register_task",
        "ssms.rl.env.registered_tasks",
        "ssms.rl.validation.DataValidationIssue",
        "ssms.rl.validation.DataValidationReport",
        "ssms.rl.validation.validate_rlssm_data",
    }
)


@dataclass(frozen=True)
class PublicAllModule:
    """A public module with a literal top-level ``__all__`` declaration."""

    source_path: Path
    module_name: str
    exports: tuple[str, ...]


@dataclass(frozen=True)
class RenderedPage:
    """Anchor inventory for one generated API page."""

    all_ids: frozenset[str]
    object_ids: frozenset[str]


class _ReferenceHTMLParser(HTMLParser):
    """Collect content and mkdocstrings-object IDs from rendered HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._article_depth = 0
        self.all_ids: set[str] = set()
        self.object_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article":
            self._article_depth += 1
        if not self._article_depth:
            return
        element_id = attributes.get("id")
        if not element_id:
            return
        self.all_ids.add(element_id)
        classes = set((attributes.get("class") or "").split())
        if "doc-heading" in classes:
            self.object_ids.add(element_id)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._article_depth:
            self._article_depth -= 1


def _literal_all(path: Path) -> tuple[str, ...] | None:
    """Return a literal top-level ``__all__``, or ``None`` when absent."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            assignments.append(value)

    if not assignments:
        return None
    assert len(assignments) == 1, f"{path} declares __all__ more than once"
    try:
        exports = ast.literal_eval(assignments[0])
    except (ValueError, TypeError) as error:
        raise AssertionError(f"{path} must keep __all__ literal") from error
    assert isinstance(exports, (list, tuple)), f"{path} __all__ must be a list/tuple"
    assert all(isinstance(name, str) for name in exports), (
        f"{path} __all__ must contain only strings"
    )
    assert len(exports) == len(set(exports)), f"{path} __all__ contains duplicates"
    return tuple(exports)


def _discover_public_all_modules() -> tuple[PublicAllModule, ...]:
    """Discover every non-private ``ssms`` module with a literal ``__all__``."""
    modules: list[PublicAllModule] = []
    for path in sorted((ROOT / "ssms").rglob("*.py")):
        relative_module = path.relative_to(ROOT).with_suffix("")
        parts = list(relative_module.parts)
        if parts[-1] == "__init__":
            parts.pop()
        if any(part.startswith("_") for part in parts):
            continue
        exports = _literal_all(path)
        if exports is not None:
            modules.append(PublicAllModule(path, ".".join(parts), exports))
    assert modules, "No public literal __all__ modules discovered"
    return tuple(modules)


PUBLIC_ALL_MODULES = _discover_public_all_modules()


def _walk_nav(node: Any) -> tuple[str, ...]:
    """Return page destinations from arbitrarily nested MkDocs nav data."""
    if isinstance(node, str):
        return (node,)
    if isinstance(node, list):
        return tuple(destination for item in node for destination in _walk_nav(item))
    if isinstance(node, dict):
        return tuple(
            destination for item in node.values() for destination in _walk_nav(item)
        )
    raise AssertionError(f"Unexpected MkDocs nav node: {node!r}")


@pytest.fixture(scope="session")
def mkdocs_config(tmp_path_factory: pytest.TempPathFactory):
    """Load the real configuration with a temporary output directory."""
    return load_config(
        config_file=str(ROOT / "mkdocs.yml"),
        site_dir=str(tmp_path_factory.mktemp("reference-site")),
        strict=True,
    )


@pytest.fixture(scope="session")
def rendered_api(mkdocs_config) -> dict[str, RenderedPage]:
    """Build once and return fresh anchor inventories for all API pages."""
    files = get_files(mkdocs_config)
    destinations = {file.src_uri: file.dest_uri for file in files}
    build(mkdocs_config)

    pages: dict[str, RenderedPage] = {}
    for source in sorted(DOCS.glob("api/**/*.md")):
        source_uri = source.relative_to(DOCS).as_posix()
        output = Path(mkdocs_config.site_dir) / destinations[source_uri]
        parser = _ReferenceHTMLParser()
        parser.feed(output.read_text(encoding="utf-8"))
        pages[source_uri] = RenderedPage(
            all_ids=frozenset(parser.all_ids),
            object_ids=frozenset(parser.object_ids),
        )
    return pages


@lru_cache(maxsize=1)
def _identity_anchor_candidates() -> dict[int, frozenset[str]]:
    """Group public callable/class aliases by their imported object identity."""
    objects: dict[int, object] = {}
    candidates: dict[int, set[str]] = {}
    for module_record in PUBLIC_ALL_MODULES:
        module = importlib.import_module(module_record.module_name)
        for name in module_record.exports:
            exported = getattr(module, name)
            if not (inspect.isclass(exported) or callable(exported)):
                continue
            object_id = id(exported)
            objects[object_id] = exported
            candidates.setdefault(object_id, set()).add(
                f"{module_record.module_name}.{name}"
            )

    for object_id, exported in objects.items():
        defining_module = getattr(exported, "__module__", None)
        qualified_name = getattr(exported, "__qualname__", None)
        if defining_module and qualified_name and "<locals>" not in qualified_name:
            candidates[object_id].add(f"{defining_module}.{qualified_name}")
    return {object_id: frozenset(names) for object_id, names in candidates.items()}


def _export_is_rendered(
    module_record: PublicAllModule,
    name: str,
    all_ids: frozenset[str],
    object_ids: frozenset[str],
) -> bool:
    """Check exact stable anchors or an anchored alias of the same object."""
    export_id = f"{module_record.module_name}.{name}"
    if export_id in all_ids:
        return True

    module = importlib.import_module(module_record.module_name)
    exported = getattr(module, name)
    if isinstance(exported, ModuleType):
        prefix = f"{exported.__name__}."
        return exported.__name__ in object_ids or any(
            anchor.startswith(prefix) for anchor in object_ids
        )
    if not (inspect.isclass(exported) or callable(exported)):
        return False
    candidates = _identity_anchor_candidates().get(id(exported), frozenset())
    return not candidates.isdisjoint(object_ids)


@pytest.mark.parametrize(
    "module_record",
    PUBLIC_ALL_MODULES,
    ids=lambda record: record.module_name,
)
def test_all_public_exports_have_rendered_reference_coverage(
    module_record: PublicAllModule,
    rendered_api: dict[str, RenderedPage],
) -> None:
    """Require every discovered public export to resolve to a rendered object/anchor."""
    all_ids = frozenset(
        anchor for page in rendered_api.values() for anchor in page.all_ids
    )
    object_ids = frozenset(
        anchor for page in rendered_api.values() for anchor in page.object_ids
    )
    missing = [
        name
        for name in module_record.exports
        if not _export_is_rendered(module_record, name, all_ids, object_ids)
    ]
    assert not missing, (
        f"Missing rendered reference objects/anchors for "
        f"{module_record.module_name}: {missing}"
    )


def test_configuration_reference_renders_public_objects(
    rendered_api: dict[str, RenderedPage],
) -> None:
    """Keep config registries, utilities, and factories as generated objects."""
    object_ids = rendered_api["api/config.md"].object_ids
    assert CONFIG_OBJECT_IDS <= object_ids, (
        f"Missing generated configuration objects: {sorted(CONFIG_OBJECT_IDS - object_ids)}"
    )


def test_non_all_public_surfaces_render_as_objects(
    rendered_api: dict[str, RenderedPage],
) -> None:
    """Protect integration, RL environment, and validation object references."""
    object_ids = frozenset(
        anchor for page in rendered_api.values() for anchor in page.object_ids
    )
    assert NON_ALL_OBJECT_IDS <= object_ids, (
        f"Missing generated non-__all__ objects: {sorted(NON_ALL_OBJECT_IDS - object_ids)}"
    )


def test_rlssm_reference_assigns_derived_fields_to_their_public_owner() -> None:
    """Keep ModelConfig and AssembledModel field ownership accurate."""
    reference = (DOCS / "api/rlssm.md").read_text(encoding="utf-8")
    config_module = importlib.import_module("ssms.rl.config")
    assembled_module = importlib.import_module("ssms.rl.assembled")

    config_fields = set(config_module.ModelConfig.__dataclass_fields__)
    assembled_fields = set(assembled_module.AssembledModel.__dataclass_fields__)
    assert {
        "list_params",
        "bounds",
        "params_default",
        "response_to_choice",
    } <= config_fields
    assert "computed_params" not in config_fields
    assert "computed_params" in assembled_fields
    assert "`AssembledModel.computed_params` exposes" in reference
    assert "`bounds`, `computed_params`" not in reference


@pytest.mark.parametrize(
    ("page", "legacy_ids"),
    [
        ("api/config.md", LEGACY_CONFIG_IDS),
        ("api/rlssm.md", LEGACY_RLSSM_IDS),
    ],
)
def test_legacy_reference_fragments_remain_addressable(
    page: str,
    legacy_ids: frozenset[str],
    rendered_api: dict[str, RenderedPage],
) -> None:
    """Keep previously published deep links valid after reference rewrites."""
    rendered_ids = rendered_api[page].all_ids
    assert legacy_ids <= rendered_ids, (
        f"Missing compatibility anchors in {page}: {sorted(legacy_ids - rendered_ids)}"
    )


def test_reference_pages_are_in_navigation(mkdocs_config) -> None:
    """Walk the parsed nav and require each API page exactly once."""
    destinations = Counter(_walk_nav(mkdocs_config.nav))
    reference_pages = {
        path.relative_to(DOCS).as_posix() for path in DOCS.glob("api/**/*.md")
    }
    nav_reference_pages = {
        destination
        for destination in destinations
        if destination.startswith("api/") and destination.endswith(".md")
    }
    assert nav_reference_pages == reference_pages
    duplicates = {
        page: destinations[page] for page in reference_pages if destinations[page] != 1
    }
    assert not duplicates, f"API reference nav entries must occur once: {duplicates}"
