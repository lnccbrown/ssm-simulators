"""Tests for the ssm_configs plugin system.

A plugin is an installed distribution named ``<registry prefix>-<model name>``
that implements the ``ssm_configs_config_path`` hook. These tests cover the
manual registration path, the real entry-point path, prefix routing, laziness,
and each warn-and-skip failure mode.
"""

import importlib
import json
import sys
import textwrap
from pathlib import Path

import pluggy
import pytest

import ssm_configs
from ssm_configs import _plugins
from ssm_configs.registry import BaseModelRegistry, HSSMRegistry, RLSSMRegistry

hookimpl = pluggy.HookimplMarker("ssm_configs")


@pytest.fixture(autouse=True)
def clean_plugin_state():
    """Restore discovery state so tests cannot leak plugins into each other."""
    registries = list(BaseModelRegistry.registries_by_prefix.values())
    saved = {cls: dict(cls._external_models) for cls in registries}
    saved_loaded = _plugins._loaded
    saved_plugins = list(_plugins.plugin_manager.get_plugins())

    yield

    for plugin in _plugins.plugin_manager.get_plugins():
        if plugin not in saved_plugins:
            _plugins.plugin_manager.unregister(plugin)
    for cls, external in saved.items():
        cls._external_models.clear()
        cls._external_models.update(external)
    _plugins._loaded = saved_loaded


def make_config(tmp_path: Path, name: str) -> Path:
    """Write a minimal valid model config JSON and return its path."""
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"name": name, "description": "a test model"}))
    return path


def make_plugin(config_path: Path | str, raises: bool = False):
    """Build a plugin object exposing the config-path hook."""

    class Plugin:
        @hookimpl
        def ssm_configs_config_path(self):
            if raises:
                raise RuntimeError("boom")
            return config_path

    return Plugin()


def register(plugin, dist_name: str) -> None:
    """Register a plugin by hand and re-run discovery.

    Manually registered plugins have no distribution metadata, so the name they
    are registered under stands in for the distribution name.
    """
    _plugins.plugin_manager.register(plugin, name=dist_name)
    ssm_configs.load_plugins(force=True)


def test_manual_plugin_round_trips(tmp_path):
    """A registered plugin becomes an external model that load_config can read."""
    config = make_config(tmp_path, "my_cool_model")
    register(make_plugin(config), "hssm-my-cool-model")

    assert ssm_configs.hssm_registry.is_external("my_cool_model")
    assert ssm_configs.hssm_registry.is_supported("my_cool_model")
    assert not ssm_configs.hssm_registry.is_internal("my_cool_model")

    loaded = ssm_configs.hssm_registry.load_config("my_cool_model")
    assert loaded.name == "my_cool_model"
    assert loaded.description == "a test model"


def test_string_paths_are_accepted(tmp_path):
    """The hook may return a str as well as a Path."""
    config = make_config(tmp_path, "stringy")
    register(make_plugin(str(config)), "hssm-stringy")

    assert ssm_configs.hssm_registry.load_config("stringy").name == "stringy"


def test_prefix_routes_to_the_matching_registry(tmp_path):
    """The distribution prefix picks the registry; others are untouched."""
    register(make_plugin(make_config(tmp_path, "rl_thing")), "rlssm-rl-thing")

    assert ssm_configs.rlssm_registry.is_external("rl_thing")
    assert not ssm_configs.hssm_registry.is_external("rl_thing")


def test_dist_name_is_normalized(tmp_path):
    """Underscores, dots and case in the distribution name all normalize."""
    register(make_plugin(make_config(tmp_path, "cfg")), "HSSM_My.Cool_Model")

    assert ssm_configs.hssm_registry.is_external("my_cool_model")


def test_unknown_prefix_warns_and_skips(tmp_path):
    """A distribution matching no registry prefix is skipped."""
    plugin = make_plugin(make_config(tmp_path, "cfg"))

    with pytest.warns(UserWarning, match="matches no registry prefix"):
        register(plugin, "totally-unrelated-model")

    assert "model" not in ssm_configs.hssm_registry.external_models


def test_internal_model_wins_over_plugin(tmp_path):
    """A plugin cannot shadow a built-in model."""
    internal = ssm_configs.hssm_registry.internal_models[0]
    plugin = make_plugin(make_config(tmp_path, internal))

    with pytest.warns(UserWarning, match="already a built-in model"):
        register(plugin, f"hssm-{internal.replace('_', '-')}")

    assert not ssm_configs.hssm_registry.is_external(internal)
    assert ssm_configs.hssm_registry.is_internal(internal)


def test_duplicate_model_name_warns_and_keeps_the_first(tmp_path):
    """The second plugin claiming a name is skipped, not silently preferred.

    Two distributions can only collide when their names normalize to the same
    thing, e.g. ``hssm-duplicated`` and ``hssm_duplicated``.
    """
    first = make_config(tmp_path, "duplicated")
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = make_config(second_dir, "duplicated")

    register(make_plugin(first), "hssm-duplicated")
    with pytest.warns(UserWarning, match="already registered from"):
        register(make_plugin(second), "hssm_duplicated")

    assert ssm_configs.hssm_registry.external_models["duplicated"] == first


def test_re_running_discovery_is_idempotent(tmp_path):
    """Re-registering the same plugin path does not warn about itself."""
    register(make_plugin(make_config(tmp_path, "stable")), "hssm-stable")
    ssm_configs.load_plugins(force=True)

    assert ssm_configs.hssm_registry.external_models["stable"].name == "stable.json"


def test_raising_hook_warns_and_skips():
    """A plugin whose hook raises is skipped, not propagated."""
    plugin = make_plugin("unused", raises=True)

    with pytest.warns(UserWarning, match="raised RuntimeError"):
        register(plugin, "hssm-exploding")

    assert not ssm_configs.hssm_registry.is_external("exploding")


def test_missing_config_file_warns_and_skips(tmp_path):
    """A plugin pointing at a file that is not there is skipped."""
    plugin = make_plugin(tmp_path / "nope.json")

    with pytest.warns(UserWarning, match="does not exist"):
        register(plugin, "hssm-missing")

    assert not ssm_configs.hssm_registry.is_external("missing")


def test_one_broken_plugin_does_not_block_the_others(tmp_path):
    """Discovery keeps going after a failure."""
    _plugins.plugin_manager.register(make_plugin("unused", raises=True), "hssm-broken")
    _plugins.plugin_manager.register(
        make_plugin(make_config(tmp_path, "healthy")), "hssm-healthy"
    )

    with pytest.warns(UserWarning):
        ssm_configs.load_plugins(force=True)

    assert ssm_configs.hssm_registry.is_external("healthy")


def test_discovery_is_lazy():
    """Importing ssm_configs scans nothing; touching a registry does."""
    source = textwrap.dedent(
        """
        import ssm_configs
        from ssm_configs import _plugins

        assert _plugins._loaded is False, "import should not trigger discovery"
        ssm_configs.hssm_registry.is_external("anything")
        assert _plugins._loaded is True, "registry access should trigger discovery"
        print("ok")
        """
    )
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_entry_point_discovery(tmp_path, monkeypatch):
    """An installed distribution is discovered without anyone importing it."""
    package = tmp_path / "hssm_model_from_entry_point"
    package.mkdir()
    (package / "model_from_entry_point.json").write_text(
        json.dumps({"name": "model_from_entry_point"})
    )
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import pluggy

            hookimpl = pluggy.HookimplMarker("ssm_configs")

            @hookimpl
            def ssm_configs_config_path():
                return Path(__file__).parent / "model_from_entry_point.json"
            """
        )
    )

    dist_info = tmp_path / "hssm_model_from_entry_point-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: hssm-model-from-entry-point\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[ssm_configs]\nmodel_from_entry_point = hssm_model_from_entry_point\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert "hssm_model_from_entry_point" not in sys.modules
    ssm_configs.load_plugins(force=True)

    try:
        assert ssm_configs.hssm_registry.is_external("model_from_entry_point")
        assert (
            ssm_configs.hssm_registry.load_config("model_from_entry_point").name
            == "model_from_entry_point"
        )
    finally:
        sys.modules.pop("hssm_model_from_entry_point", None)


def test_registries_are_registered_by_prefix():
    """Every concrete registry is discoverable by its prefix."""
    assert BaseModelRegistry.registries_by_prefix["hssm"] is HSSMRegistry
    assert BaseModelRegistry.registries_by_prefix["rlssm"] is RLSSMRegistry
