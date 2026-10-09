# Configuration exports

<span id="ssms.config" aria-hidden="true"></span>

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

<span id="ssms.config.model_config" aria-hidden="true"></span>

### Copy-on-access model mapping

::: ssms.config.CopyOnAccessDict

`model_config` is the compatibility mapping instantiated from this class. Each
lookup returns a deep copy, so local changes do not mutate the registered
configuration.

### Model configuration builder

<span id="ssms.config.model_config_builder" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.add_boundary" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.add_drift" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.from_model" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.from_scratch" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.get_sampling_transforms" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.get_simulation_transforms" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.get_transforms" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.minimal_config" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.validate_config" aria-hidden="true"></span>
<span id="ssms.config.model_config_builder.ModelConfigBuilder.with_deadline" aria-hidden="true"></span>

::: ssms.config.ModelConfigBuilder

### Boundary registry

<span id="ssms.config.boundary_registry" aria-hidden="true"></span>
<span id="ssms.config.boundary_registry.register_boundary" aria-hidden="true"></span>
<span id="ssms.config.boundary_registry.get_boundary_registry" aria-hidden="true"></span>

::: ssms.config.register_boundary

::: ssms.config.get_boundary_registry

::: ssms.config.boundary_registry.BoundaryRegistry

### Drift registry

<span id="ssms.config.drift_registry" aria-hidden="true"></span>
<span id="ssms.config.drift_registry.register_drift" aria-hidden="true"></span>
<span id="ssms.config.drift_registry.get_drift_registry" aria-hidden="true"></span>

::: ssms.config.register_drift

::: ssms.config.get_drift_registry

::: ssms.config.drift_registry.DriftRegistry

### Model registry

<span id="ssms.config.model_registry" aria-hidden="true"></span>
<span id="ssms.config.model_registry.register_model_config" aria-hidden="true"></span>
<span id="ssms.config.model_registry.register_model_config_factory" aria-hidden="true"></span>
<span id="ssms.config.model_registry.get_model_registry" aria-hidden="true"></span>

::: ssms.config.register_model_config

::: ssms.config.register_model_config_factory

::: ssms.config.get_model_registry

::: ssms.config.model_registry.ModelConfigRegistry

### Configuration utilities

::: ssms.config.boundary_config_to_function_params

::: ssms.config.config_utils

## Generator configuration functions

<span id="ssms.get_default_generator_config" aria-hidden="true"></span>
<span id="ssms.config.get_default_generator_config" aria-hidden="true"></span>
<span id="ssms.config.get_kde_simulation_filters" aria-hidden="true"></span>
<span id="ssms.config.get_lan_kde_config" aria-hidden="true"></span>
<span id="ssms.config.get_lan_config" aria-hidden="true"></span>
<span id="ssms.config.get_defective_detector_config" aria-hidden="true"></span>
<span id="ssms.config.get_ratio_estimator_config" aria-hidden="true"></span>

::: data_generator_config.get_default_generator_config

::: data_generator_config.get_kde_simulation_filters

::: data_generator_config.get_lan_kde_config

::: data_generator_config.get_lan_config

::: data_generator_config.get_defective_detector_config

::: data_generator_config.get_ratio_estimator_config

All return new configuration dictionaries. `get_default_generator_config`
selects the approach-specific factory; `get_lan_config` is retained only as a
deprecated alias of `get_lan_kde_config`.
