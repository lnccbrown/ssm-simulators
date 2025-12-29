"""Global registry for drift functions.

This module provides a centralized registry for drift functions used in
sequential sampling models. It follows the same pattern as the parameter
sampling constraint registry and boundary registry for consistency.

Examples
--------
Register a custom drift:

>>> from ssms.config import register_drift
>>>
>>> def sinusoidal_drift(t, frequency=1.0, amplitude=0.5, baseline=1.0):
...     return baseline + amplitude * np.sin(2 * np.pi * frequency * t)
>>>
>>> register_drift("sinusoidal", sinusoidal_drift, ["frequency", "amplitude", "baseline"])
>>>
>>> # Use with ConfigBuilder
>>> from ssms.config import ConfigBuilder
>>> config = ConfigBuilder.from_model("ddm")
>>> config = ConfigBuilder.add_drift(config, "sinusoidal")

List available drifts:

>>> from ssms.config import get_drift_registry
>>> print(get_drift_registry().list_drifts())
"""

from typing import Any, Callable


class DriftRegistry:
    """Global registry for drift functions.

    This registry maintains a mapping of drift names to their configuration,
    including the function and its parameters.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._drifts: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable,
        params: list[str],
    ) -> None:
        """Register a drift function.

        Parameters
        ----------
        name : str
            Unique name for the drift (e.g., "gamma_drift", "constant")
        function : Callable
            Drift function with signature (t, **params) -> float or array
            where t is time and params are drift-specific parameters
        params : list[str]
            List of parameter names the function expects (e.g., ["shape", "scale", "c"])

        Raises
        ------
        ValueError
            If name already registered

        Examples
        --------
        >>> def linear_drift(t, slope=0.5, intercept=1.0):
        ...     return intercept + slope * t
        >>>
        >>> registry = DriftRegistry()
        >>> registry.register("linear", linear_drift, ["slope", "intercept"])
        """
        if name in self._drifts:
            raise ValueError(
                f"Drift '{name}' is already registered. "
                f"Use a different name or unregister the existing drift first."
            )

        self._drifts[name] = {
            "fun": function,
            "params": params,
        }

    def get(self, name: str) -> dict[str, Any]:
        """Get drift configuration by name.

        Parameters
        ----------
        name : str
            Name of the registered drift

        Returns
        -------
        dict
            Dictionary containing:
            - 'fun': The drift function
            - 'params': List of parameter names

        Raises
        ------
        KeyError
            If drift name not registered

        Examples
        --------
        >>> registry = DriftRegistry()
        >>> config = registry.get("gamma_drift")
        >>> print(config["params"])
        ['shape', 'scale', 'c']
        """
        if name not in self._drifts:
            available = self.list_drifts()
            raise KeyError(
                f"Drift '{name}' is not registered. Available drifts: {available}"
            )
        return self._drifts[name]

    def is_registered(self, name: str) -> bool:
        """Check if drift name is registered.

        Parameters
        ----------
        name : str
            Drift name to check

        Returns
        -------
        bool
            True if drift is registered, False otherwise

        Examples
        --------
        >>> registry = DriftRegistry()
        >>> registry.is_registered("gamma_drift")
        True
        >>> registry.is_registered("my_custom_drift")
        False
        """
        return name in self._drifts

    def list_drifts(self) -> list[str]:
        """List all registered drift names.

        Returns
        -------
        list[str]
            Sorted list of all registered drift names

        Examples
        --------
        >>> registry = DriftRegistry()
        >>> drifts = registry.list_drifts()
        >>> print(drifts)
        ['constant', 'gamma_drift', ...]
        """
        return sorted(self._drifts.keys())

    def __repr__(self) -> str:
        """String representation of registry."""
        n_drifts = len(self._drifts)
        return f"DriftRegistry({n_drifts} drifts registered)"


# Global singleton instance
_GLOBAL_DRIFT_REGISTRY = DriftRegistry()


def register_drift(
    name: str,
    function: Callable,
    params: list[str],
) -> None:
    """Register a drift function globally.

    This is the main entry point for registering custom drift functions.
    Once registered, drifts can be used with ConfigBuilder.add_drift()
    just like built-in drifts.

    Parameters
    ----------
    name : str
        Unique name for the drift
    function : Callable
        Drift function with signature (t, **params) -> float or array
    params : list[str]
        List of parameter names the function expects

    Raises
    ------
    ValueError
        If name already registered

    Examples
    --------
    Register a custom sinusoidal drift:

    >>> import numpy as np
    >>> from ssms.config import register_drift
    >>>
    >>> def sinusoidal_drift(t, frequency=1.0, amplitude=0.5, baseline=1.0):
    ...     return baseline + amplitude * np.sin(2 * np.pi * frequency * t)
    >>>
    >>> register_drift(
    ...     name="sinusoidal",
    ...     function=sinusoidal_drift,
    ...     params=["frequency", "amplitude", "baseline"]
    ... )
    >>>
    >>> # Now use it with ConfigBuilder
    >>> from ssms.config import ConfigBuilder
    >>> config = ConfigBuilder.from_model("ddm")
    >>> config = ConfigBuilder.add_drift(config, "sinusoidal")

    Register a time-varying drift:

    >>> def exponential_drift(t, rate=0.1, asymptote=2.0):
    ...     return asymptote * (1 - np.exp(-rate * t))
    >>>
    >>> register_drift("exponential", exponential_drift, ["rate", "asymptote"])
    """
    _GLOBAL_DRIFT_REGISTRY.register(name, function, params)


def get_drift_registry() -> DriftRegistry:
    """Get the global drift registry.

    Use this to access registry methods like list_drifts() or is_registered().

    Returns
    -------
    DriftRegistry
        The global drift registry instance

    Examples
    --------
    >>> from ssms.config import get_drift_registry
    >>>
    >>> # List all available drifts
    >>> registry = get_drift_registry()
    >>> print(registry.list_drifts())
    ['constant', 'gamma_drift', ...]
    >>>
    >>> # Check if a drift exists
    >>> if registry.is_registered("my_drift"):
    ...     config = registry.get("my_drift")
    """
    return _GLOBAL_DRIFT_REGISTRY


# Initialize with all built-in drifts automatically
# This uses the existing drift_config dict which already defines all built-in drifts
# This way, when someone adds a new drift function to base.py, it's automatically
# registered here without needing to update this file
from ssms.config._modelconfig.base import drift_config

for drift_name, drift_spec in drift_config.items():
    register_drift(
        name=drift_name,
        function=drift_spec["fun"],
        params=drift_spec["params"],
    )
