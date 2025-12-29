"""Global registry for model configurations.

This module provides a centralized registry for complete model configurations.
It follows the same pattern as boundary and drift registries for consistency,
with additional support for factory functions to enable lazy loading.

Examples
--------
Register a custom model configuration:

>>> from ssms.config import register_model_config
>>>
>>> my_config = {
...     "name": "my_ddm",
...     "params": ["v", "a", "z", "t"],
...     "nchoices": 2,
...     "simulator": my_sim_fn,
... }
>>>
>>> register_model_config("my_ddm", my_config)
>>>
>>> # Now use it with ConfigBuilder
>>> from ssms.config import ConfigBuilder
>>> config = ConfigBuilder.from_model("my_ddm")

Register using a factory function:

>>> def get_my_model_config():
...     return {...}
>>>
>>> register_model_config_factory("my_model", get_my_model_config)

List available models:

>>> from ssms.config import get_model_registry
>>> print(get_model_registry().list_models())
"""

import copy
from typing import Callable


class ModelConfigRegistry:
    """Global registry for complete model configurations.

    This registry maintains a mapping of model names to their configurations.
    Supports both direct config registration and factory functions for lazy loading.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._configs: dict[str, dict] = {}
        self._factories: dict[str, Callable[[], dict]] = {}

    def register_config(self, name: str, config: dict) -> None:
        """Register a model configuration directly.

        Parameters
        ----------
        name : str
            Unique name for the model (e.g., "ddm", "my_custom_model")
        config : dict
            Complete model configuration dictionary containing at minimum:
            - 'name': Model name
            - 'params': List of parameter names
            - 'nchoices': Number of choices
            - 'simulator': Simulator function

        Raises
        ------
        ValueError
            If name already registered (either as config or factory)

        Examples
        --------
        >>> config = {
        ...     "name": "my_model",
        ...     "params": ["v", "a", "z", "t"],
        ...     "param_bounds": [[-3, 0.3, 0.1, 0], [3, 3.0, 0.9, 2]],
        ...     "nchoices": 2,
        ...     "simulator": my_sim_fn,
        ... }
        >>>
        >>> registry = ModelConfigRegistry()
        >>> registry.register_config("my_model", config)
        """
        if name in self._configs or name in self._factories:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Use a different name or unregister the existing model first."
            )

        self._configs[name] = config

    def register_factory(self, name: str, factory: Callable[[], dict]) -> None:
        """Register a model config factory function.

        Factory functions enable lazy loading - the config is only created when
        first accessed. This is useful for models with expensive initialization
        or to reduce memory footprint.

        Parameters
        ----------
        name : str
            Unique name for the model
        factory : Callable[[], dict]
            Function that returns a complete model configuration dict

        Raises
        ------
        ValueError
            If name already registered (either as config or factory)

        Examples
        --------
        >>> def get_my_model_config():
        ...     # Expensive computation here
        ...     return {...}
        >>>
        >>> registry = ModelConfigRegistry()
        >>> registry.register_factory("my_model", get_my_model_config)
        """
        if name in self._configs or name in self._factories:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Use a different name or unregister the existing model first."
            )

        self._factories[name] = factory

    def get(self, name: str) -> dict:
        """Get model configuration by name.

        Returns a deep copy of the configuration to prevent accidental mutation
        of the registered config.

        Parameters
        ----------
        name : str
            Name of the registered model

        Returns
        -------
        dict
            Complete model configuration dictionary (deep copy)

        Raises
        ------
        KeyError
            If model name not registered

        Examples
        --------
        >>> registry = ModelConfigRegistry()
        >>> config = registry.get("ddm")
        >>> print(config["params"])
        ['v', 'a', 'z', 't']
        """
        if name in self._configs:
            return copy.deepcopy(self._configs[name])

        if name in self._factories:
            # Call factory to get config
            return self._factories[name]()

        available = self.list_models()
        raise KeyError(
            f"Model '{name}' is not registered. "
            f"Available models: {available[:10]}... "
            f"({len(available)} total)"
        )

    def has_model(self, name: str) -> bool:
        """Check if model name is registered.

        Parameters
        ----------
        name : str
            Model name to check

        Returns
        -------
        bool
            True if model is registered, False otherwise

        Examples
        --------
        >>> registry = ModelConfigRegistry()
        >>> registry.has_model("ddm")
        True
        >>> registry.has_model("my_custom_model")
        False
        """
        return name in self._configs or name in self._factories

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns
        -------
        list[str]
            Sorted list of all registered model names

        Examples
        --------
        >>> registry = ModelConfigRegistry()
        >>> models = registry.list_models()
        >>> print(models[:5])
        ['angle', 'ddm', 'ddm_par2', 'ddm_sdv', 'ddm_st']
        """
        return sorted(list(self._configs.keys()) + list(self._factories.keys()))

    def __repr__(self) -> str:
        """String representation of registry."""
        n_configs = len(self._configs)
        n_factories = len(self._factories)
        total = n_configs + n_factories
        return f"ModelConfigRegistry({total} models: {n_configs} direct, {n_factories} factories)"


# Global singleton instance
_GLOBAL_MODEL_REGISTRY = ModelConfigRegistry()


def register_model_config(name: str, config: dict) -> None:
    """Register a model configuration globally.

    This is the main entry point for registering custom model configurations.
    Once registered, models can be used with ConfigBuilder.from_model()
    just like built-in models.

    Parameters
    ----------
    name : str
        Unique name for the model
    config : dict
        Complete model configuration dictionary

    Raises
    ------
    ValueError
        If name already registered

    Examples
    --------
    Register a complete custom model:

    >>> from ssms.config import register_model_config
    >>>
    >>> my_model = {
    ...     "name": "my_custom_ddm",
    ...     "params": ["v", "a", "z", "t"],
    ...     "param_bounds": [[-3, 0.3, 0.1, 0], [3, 3.0, 0.9, 2]],
    ...     "nchoices": 2,
    ...     "n_params": 4,
    ...     "default_params": [1.0, 1.5, 0.5, 0.3],
    ...     "simulator": my_simulator_function,
    ... }
    >>>
    >>> register_model_config("my_custom_ddm", my_model)
    >>>
    >>> # Now use it like any built-in model
    >>> from ssms.config import ConfigBuilder
    >>> config = ConfigBuilder.from_model("my_custom_ddm")
    >>>
    >>> # Or with Simulator
    >>> from ssms.basic_simulators import Simulator
    >>> sim = Simulator(model="my_custom_ddm")

    Register with custom boundary and drift:

    >>> advanced_model = {
    ...     "name": "advanced_ddm",
    ...     "params": ["v", "a", "z", "t", "theta"],
    ...     "nchoices": 2,
    ...     "boundary": my_boundary_fn,
    ...     "boundary_name": "custom",
    ...     "boundary_params": ["theta"],
    ...     "drift": my_drift_fn,
    ...     "drift_name": "custom",
    ...     "drift_params": [],
    ...     "simulator": my_simulator,
    ... }
    >>>
    >>> register_model_config("advanced_ddm", advanced_model)
    """
    _GLOBAL_MODEL_REGISTRY.register_config(name, config)


def register_model_config_factory(name: str, factory: Callable[[], dict]) -> None:
    """Register a model config factory function globally.

    Use this when you want lazy loading of model configurations, or when
    the config requires computation/processing at access time.

    Parameters
    ----------
    name : str
        Unique name for the model
    factory : Callable[[], dict]
        Function that returns a complete model configuration dict

    Raises
    ------
    ValueError
        If name already registered

    Examples
    --------
    Register with lazy loading:

    >>> from ssms.config import register_model_config_factory
    >>>
    >>> def get_my_model():
    ...     # This only runs when the model is first accessed
    ...     return {
    ...         "name": "my_model",
    ...         "params": ["v", "a", "z", "t"],
    ...         "nchoices": 2,
    ...         "simulator": create_simulator(),  # Expensive operation
    ...     }
    >>>
    >>> register_model_config_factory("my_model", get_my_model)
    >>>
    >>> # Factory is called only when accessing the model
    >>> config = ConfigBuilder.from_model("my_model")
    """
    _GLOBAL_MODEL_REGISTRY.register_factory(name, factory)


def get_model_registry() -> ModelConfigRegistry:
    """Get the global model registry.

    Use this to access registry methods like list_models() or has_model().

    Returns
    -------
    ModelConfigRegistry
        The global model registry instance

    Examples
    --------
    >>> from ssms.config import get_model_registry
    >>>
    >>> # List all available models
    >>> registry = get_model_registry()
    >>> print(registry.list_models())
    ['ddm', 'angle', 'weibull_cdf', ...]
    >>>
    >>> # Check if a model exists
    >>> if registry.has_model("my_model"):
    ...     config = registry.get("my_model")
    """
    return _GLOBAL_MODEL_REGISTRY


# Initialize with all built-in models automatically
# This uses the existing get_model_config() function which already collects all models
# This way, when someone adds a new model to _modelconfig/__init__.py, it's automatically
# registered here without needing to update this file
from ssms.config._modelconfig import get_model_config

# Get all built-in model configs and register them
_builtin_configs = get_model_config()
for model_name, model_config in _builtin_configs.items():
    # Register as direct config (already loaded by get_model_config)
    register_model_config(model_name, model_config)
