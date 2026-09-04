# ssm-configs

Pydantic-based configuration schemas and model registries for sequential
sampling models.

This package is developed inside the `ssm-simulators` uv workspace. While it is
under active development it is intentionally **not** a runtime dependency of
`ssm-simulators`; it is installed only through the workspace `dev` dependency
group.

## Plugins

A third-party package can add a model to a registry just by being installed —
nobody has to import it. Name the distribution `<registry prefix>-<model name>`
and implement one hook:

```python
# hssm_my_cool_model/__init__.py
from pathlib import Path

import pluggy

hookimpl = pluggy.HookimplMarker("ssm_configs")


@hookimpl
def ssm_configs_config_path() -> Path:
    """Path to this plugin's model config JSON."""
    return Path(__file__).parent / "my_cool_model.json"
```

```toml
# pyproject.toml
[project]
name = "hssm-my-cool-model"
dependencies = ["ssm-configs"]

[project.entry-points."ssm_configs"]
my_cool_model = "hssm_my_cool_model"
```

Installing that distribution registers `my_cool_model` in the HSSM registry:
the `hssm-` prefix picks the registry, the rest of the name becomes the model
name, and one distribution ships one model.

```python
import ssm_configs

ssm_configs.hssm_registry.is_external("my_cool_model")  # True
ssm_configs.hssm_registry.load_config("my_cool_model")  # HSSMConfigSchema
```

Discovery is lazy — it runs on the first read of a registry's
`external_models`, not at import — and forgiving: a distribution whose prefix
matches no registry, a name already taken by a built-in or another plugin, a
hook that raises, or a missing config file all produce a `UserWarning` naming
the distribution and are skipped, so one broken plugin cannot break the rest.

For a plugin that is not installed as a distribution (a local experiment, a
test), register it by hand:

```python
ssm_configs.plugin_manager.register(my_plugin, name="hssm-my-cool-model")
ssm_configs.load_plugins(force=True)
```

`ssm_configs.plugin_manager.set_blocked("hssm-my-cool-model")` disables one.
