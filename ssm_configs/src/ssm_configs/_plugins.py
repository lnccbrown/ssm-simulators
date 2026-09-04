"""Plugin discovery for third-party model configurations.

A plugin is an installed distribution whose name leads with the prefix of one of
the registries in :mod:`ssm_configs.registry` -- ``hssm-my-cool-model`` targets
the HSSM registry and registers the model ``my_cool_model``. Both the target
registry and the model name are derived from the distribution name; the plugin
itself only has to point at its JSON config:

.. code-block:: python

    # hssm_my_cool_model/__init__.py
    import pluggy
    from pathlib import Path

    hookimpl = pluggy.HookimplMarker("ssm_configs")

    @hookimpl
    def ssm_configs_config_path() -> Path:
        return Path(__file__).parent / "my_cool_model.json"

.. code-block:: toml

    # pyproject.toml
    [project.entry-points."ssm_configs"]
    my_cool_model = "hssm_my_cool_model"

Discovery is lazy: it runs on the first read of any registry's
``external_models``, never at import time. Anything that does not line up -- an
unknown prefix, a name already taken, a hook that raises, a missing file -- is
warned about and skipped, so one broken plugin cannot take the others down with
it.
"""

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pluggy

if TYPE_CHECKING:
    from .registry import BaseModelRegistry

PROJECT_NAME = "ssm_configs"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class SSMConfigsHookSpec:
    """The hooks a ``ssm_configs`` plugin may implement."""

    @hookspec
    def ssm_configs_config_path() -> str | Path:  # type: ignore[empty-body,misc]
        """Return the path to this plugin's JSON model configuration.

        Returns
        -------
        str | Path
            Path to a single JSON config file. The model name and the registry
            it lands in are taken from the plugin's distribution name, so the
            path is all a plugin has to supply.
        """


plugin_manager = pluggy.PluginManager(PROJECT_NAME)
plugin_manager.add_hookspecs(SSMConfigsHookSpec)

_loaded = False


def _normalize(dist_name: str) -> str:
    """Normalize a distribution name the way PEP 503 does."""
    return re.sub(r"[-_.]+", "-", dist_name).lower()


def _resolve_registry(
    dist_name: str,
) -> "tuple[type[BaseModelRegistry[Any]], str] | None":
    """Map a distribution name onto a registry class and a model name.

    Parameters
    ----------
    dist_name : str
        The name of the distribution providing the plugin.

    Returns
    -------
    tuple[type[BaseModelRegistry], str] | None
        The registry the model belongs in and the name to register it under, or
        None if no registry prefix matches the distribution name.
    """
    from .registry import BaseModelRegistry

    normalized = _normalize(dist_name)
    # Longest match first, so a registry whose prefix extends another's still
    # gets its own plugins.
    for prefix in sorted(BaseModelRegistry.registries_by_prefix, key=len, reverse=True):
        if normalized.startswith(f"{prefix}-"):
            model_name = normalized.removeprefix(f"{prefix}-").replace("-", "_")
            return BaseModelRegistry.registries_by_prefix[prefix], model_name
    return None


def _skip(dist_name: str, reason: str) -> None:
    warnings.warn(
        f"ssm_configs: skipping plugin from distribution '{dist_name}': {reason}",
        UserWarning,
        stacklevel=3,
    )


def load_plugins(force: bool = False) -> None:
    """Discover installed plugins and add their configs to the registries.

    Idempotent: the entry-point scan runs once and later calls are no-ops unless
    ``force`` is set. Registering a plugin by hand with
    ``plugin_manager.register(...)`` and then calling this with ``force=True``
    is the supported way to use a plugin that is not installed as a
    distribution, which is also what makes plugins testable.

    Parameters
    ----------
    force : bool
        Re-run discovery even if it has already run. Entries already registered
        under the same path are left alone.
    """
    global _loaded
    if _loaded and not force:
        return
    # Set before the work so that a registry read triggered from inside
    # discovery cannot recurse back into it.
    _loaded = True

    plugin_manager.load_setuptools_entrypoints(PROJECT_NAME)

    distributions = {
        id(plugin): dist for plugin, dist in plugin_manager.list_plugin_distinfo()
    }

    for impl in plugin_manager.hook.ssm_configs_config_path.get_hookimpls():
        dist = distributions.get(id(impl.plugin))
        # A manually registered plugin has no distribution to read a name from,
        # so fall back to the name it was registered under.
        dist_name = (
            dist.project_name
            if dist is not None
            else (plugin_manager.get_name(impl.plugin) or repr(impl.plugin))
        )

        resolved = _resolve_registry(dist_name)
        if resolved is None:
            _skip(dist_name, "its name matches no registry prefix")
            continue
        registry_cls, model_name = resolved

        if model_name in registry_cls.internal_models:
            _skip(
                dist_name,
                f"'{model_name}' is already a built-in model of the "
                f"'{registry_cls.prefix}' registry",
            )
            continue

        try:
            # Hookimpls are called one at a time rather than through the fanned
            # -out hook call, so each returned path stays attributable to the
            # distribution it came from.
            path = Path(cast("str | Path", impl.function()))
        except Exception as exc:  # noqa: BLE001 - a plugin may raise anything
            _skip(dist_name, f"its hook raised {type(exc).__name__}: {exc}")
            continue

        registered = registry_cls._external_models.get(model_name)
        if registered is not None:
            if Path(registered) != path:
                _skip(
                    dist_name,
                    f"'{model_name}' is already registered from {registered}",
                )
            continue

        if not path.exists():
            _skip(dist_name, f"its config file does not exist at {path}")
            continue

        registry_cls._external_models[model_name] = path
