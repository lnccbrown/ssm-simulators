# Configuration exports

The `ssms.config` namespace exposes model, boundary, drift, and generator
configuration surfaces. Use the
[configuration guide](../core_tutorials/tutorial_configs.ipynb) for the task
workflow; this page defines the callable and compatibility exports.

## Export inventory

| Export | Role |
| --- | --- |
| `ModelConfigBuilder` | Copy and customize registered model configurations |
| `model_config` | Copy-on-access compatibility mapping of registered configs |
| `boundary_config_to_function_params` | Prefix boundary arguments for simulator calls |
| `register_boundary`, `get_boundary_registry` | Boundary registration and discovery |
| `register_drift`, `get_drift_registry` | Drift registration and discovery |
| `register_model_config`, `register_model_config_factory`, `get_model_registry` | Model registration and discovery |
| `get_default_generator_config` | Default generator configuration |
| `get_kde_simulation_filters` | Default simulation-filter settings |
| `get_lan_kde_config` | KDE-based LAN generator configuration |
| `get_lan_config` | Deprecated alias of `get_lan_kde_config` |
| `get_defective_detector_config` | Defective-detector configuration |
| `get_ratio_estimator_config` | Ratio-estimator configuration |

## Model and component registries

::: ssms.config.ModelConfigBuilder

::: ssms.config.register_boundary

::: ssms.config.get_boundary_registry

::: ssms.config.register_drift

::: ssms.config.get_drift_registry

::: ssms.config.register_model_config

::: ssms.config.register_model_config_factory

::: ssms.config.get_model_registry

::: ssms.config.boundary_config_to_function_params

## Generator configuration functions

```python
get_default_generator_config(
    approach: str | None = None,
    model: str = "ddm",
) -> dict
get_kde_simulation_filters() -> dict
get_lan_kde_config(model: str = "ddm") -> dict
get_lan_config(model: str = "ddm") -> dict
get_defective_detector_config(model: str = "ddm") -> dict
get_ratio_estimator_config(model: str = "ddm") -> dict
```

All return new configuration dictionaries. `get_default_generator_config`
selects the approach-specific factory; `get_lan_config` is retained only as a
deprecated alias of `get_lan_kde_config`.
