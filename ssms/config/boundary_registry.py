"""Global registry for boundary functions.

This module provides a centralized registry for boundary functions used in
sequential sampling models. It follows the same pattern as the parameter
sampling constraint registry for consistency.

Examples
--------
Register a custom boundary:

>>> from ssms.config import register_boundary
>>>
>>> def my_boundary(t, a=1.0, decay=0.1):
...     return a * np.exp(-decay * t)
>>>
>>> register_boundary("exponential", my_boundary, ["a", "decay"])
>>>
>>> # Use with ModelConfigBuilder
>>> from ssms.config import ModelConfigBuilder
>>> config = ModelConfigBuilder.from_model("ddm")
>>> config = ModelConfigBuilder.add_boundary(config, "exponential")

List available boundaries:

>>> from ssms.config import get_boundary_registry
>>> print(get_boundary_registry().list_boundaries())
"""

from typing import Any, Callable


class BoundaryRegistry:
    """Global registry for boundary functions.

    This registry maintains a mapping of boundary names to their configuration,
    including the function and parameters.

    All boundary functions accept 'a' as an explicit parameter and return the final
    boundary value directly.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._boundaries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable,
        params: list[str],
    ) -> None:
        """Register a boundary function.

        Parameters
        ----------
        name : str
            Unique name for the boundary (e.g., "angle", "weibull_cdf")
        function : Callable
            Boundary function with signature (t, **params) -> float or array
            where t is time and params are boundary-specific parameters (including 'a')
        params : list[str]
            List of parameter names the function expects (e.g., ["a", "theta"] for angle)

        Raises
        ------
        ValueError
            If name already registered

        Examples
        --------
        >>> def exponential_decay(t, a=1.0, rate=0.1):
        ...     return a * np.exp(-rate * t)
        >>>
        >>> registry = BoundaryRegistry()
        >>> registry.register("exp_decay", exponential_decay, ["a", "rate"])
        """
        if name in self._boundaries:
            raise ValueError(
                f"Boundary '{name}' is already registered. "
                f"Use a different name or unregister the existing boundary first."
            )

        self._boundaries[name] = {
            "fun": function,
            "params": params,
        }

    def get(self, name: str) -> dict[str, Any]:
        """Get boundary configuration by name.

        Parameters
        ----------
        name : str
            Name of the registered boundary

        Returns
        -------
        dict
            Dictionary containing:
            - 'fun': The boundary function
            - 'params': List of parameter names (including 'a')

        Raises
        ------
        KeyError
            If boundary name not registered

        Examples
        --------
        >>> registry = BoundaryRegistry()
        >>> config = registry.get("angle")
        >>> print(config["params"])
        ['theta']
        """
        if name not in self._boundaries:
            available = self.list_boundaries()
            raise KeyError(
                f"Boundary '{name}' is not registered. "
                f"Available boundaries: {available}"
            )
        return self._boundaries[name]

    def is_registered(self, name: str) -> bool:
        """Check if boundary name is registered.

        Parameters
        ----------
        name : str
            Boundary name to check

        Returns
        -------
        bool
            True if boundary is registered, False otherwise

        Examples
        --------
        >>> registry = BoundaryRegistry()
        >>> registry.is_registered("angle")
        True
        >>> registry.is_registered("my_custom_boundary")
        False
        """
        return name in self._boundaries

    def list_boundaries(self) -> list[str]:
        """List all registered boundary names.

        Returns
        -------
        list[str]
            Sorted list of all registered boundary names

        Examples
        --------
        >>> registry = BoundaryRegistry()
        >>> boundaries = registry.list_boundaries()
        >>> print(boundaries)
        ['angle', 'constant', 'weibull_cdf', ...]
        """
        return sorted(self._boundaries.keys())

    def __repr__(self) -> str:
        """String representation of registry."""
        n_boundaries = len(self._boundaries)
        return f"BoundaryRegistry({n_boundaries} boundaries registered)"


# Global singleton instance
_GLOBAL_BOUNDARY_REGISTRY = BoundaryRegistry()


def register_boundary(
    name: str,
    function: Callable,
    params: list[str],
) -> None:
    """Register a boundary function globally.

    This is the main entry point for registering custom boundary functions.
    Once registered, boundaries can be used with ModelConfigBuilder.add_boundary()
    just like built-in boundaries.

    Parameters
    ----------
    name : str
        Unique name for the boundary
    function : Callable
        Boundary function with signature (t, **params) -> float or array
    params : list[str]
        List of parameter names the function expects (must include 'a')

    Raises
    ------
    ValueError
        If name already registered

    Examples
    --------
    Register a custom exponential decay boundary:

    >>> import numpy as np
    >>> from ssms.config import register_boundary
    >>>
    >>> def exponential_decay(t, a=1.0, rate=0.1):
    ...     return a * np.exp(-rate * t)
    >>>
    >>> register_boundary(
    ...     name="exponential_decay",
    ...     function=exponential_decay,
    ...     params=["a", "rate"]
    ... )
    >>>
    >>> # Now use it with ModelConfigBuilder
    >>> from ssms.config import ModelConfigBuilder
    >>> config = ModelConfigBuilder.from_model("ddm")
    >>> config = ModelConfigBuilder.add_boundary(config, "exponential_decay")

    Register a collapsing boundary:

    >>> def linear_collapse(t, a=1.0, slope=-0.5):
    ...     return a + slope * t
    >>>
    >>> register_boundary("linear_collapse", linear_collapse, ["a", "slope"])
    """
    _GLOBAL_BOUNDARY_REGISTRY.register(name, function, params)


def get_boundary_registry() -> BoundaryRegistry:
    """Get the global boundary registry.

    Use this to access registry methods like list_boundaries() or is_registered().

    Returns
    -------
    BoundaryRegistry
        The global boundary registry instance

    Examples
    --------
    >>> from ssms.config import get_boundary_registry
    >>>
    >>> # List all available boundaries
    >>> registry = get_boundary_registry()
    >>> print(registry.list_boundaries())
    ['angle', 'constant', 'weibull_cdf', ...]
    >>>
    >>> # Check if a boundary exists
    >>> if registry.is_registered("my_boundary"):
    ...     config = registry.get("my_boundary")
    """
    return _GLOBAL_BOUNDARY_REGISTRY


# Initialize with all built-in boundaries automatically
# This uses the existing boundary_config dict which already defines all built-in boundaries
# This way, when someone adds a new boundary function to base.py, it's automatically
# registered here without needing to update this file
from ssms.config._modelconfig.base import boundary_config

for boundary_name, boundary_spec in boundary_config.items():
    register_boundary(
        name=boundary_name,
        function=boundary_spec["fun"],
        params=boundary_spec["params"],
    )
