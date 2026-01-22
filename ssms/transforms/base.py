"""Base class for all parameter transforms.

This module defines the abstract base class that all parameter transforms
(both sampling-time and simulation-time) must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any


class ParameterTransform(ABC):
    """Abstract base class for all parameter transforms.

    A parameter transform is a single, focused operation that modifies
    theta parameters. Transforms can be composed into pipelines and are
    used at two phases:

    1. **Sampling phase**: Applied after parameter sampling to enforce
       constraints (e.g., ensure a > z by swapping values)

    2. **Simulation phase**: Applied before calling the simulator to
       prepare parameters for the expected format (e.g., stack v0, v1 → v)

    Subclasses must implement the `apply` method. The method signature
    includes optional `model_config` and `n_trials` arguments to support
    both simple constraints (which only need theta) and complex adapters
    (which may need additional context).

    Examples
    --------
    Create a custom transform:

    >>> class ScaleParameter(ParameterTransform):
    ...     def __init__(self, param_name: str, scale: float):
    ...         self.param_name = param_name
    ...         self.scale = scale
    ...
    ...     def apply(self, theta, model_config=None, n_trials=None):
    ...         if self.param_name in theta:
    ...             theta[self.param_name] = theta[self.param_name] * self.scale
    ...         return theta

    Use in model config:

    >>> model_config = {
    ...     "name": "my_model",
    ...     "parameter_transforms": {
    ...         "sampling": [SwapIfLessConstraint("a", "z")],
    ...         "simulation": [ColumnStackParameters(["v0", "v1"], "v")],
    ...     }
    ... }
    """

    @abstractmethod
    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Apply transform to theta parameters.

        This method should modify the theta dictionary in place and return it.
        It may add new parameters, modify existing parameters, or remove
        parameters as needed.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of model parameters. Values are typically numpy arrays.
        model_config : dict[str, Any] or None, optional
            Model configuration dictionary. Available for transforms that need
            additional context (e.g., looking up simulator_fixed_params).
            Sampling-time transforms typically ignore this.
        n_trials : int or None, optional
            Number of trials. Available for transforms that need to create
            arrays of a specific size. Sampling-time transforms typically
            ignore this.

        Returns
        -------
        dict[str, Any]
            The modified theta dictionary (usually the same object passed in).

        Notes
        -----
        - Transforms should be pure when possible (no side effects)
        - If creating new arrays, use dtype=np.float32 for consistency
        - Document any parameters that are added, modified, or removed
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns
        -------
        str
            A string describing this transform.
        """
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                attrs.append(f"{key}={value!r}")

        attrs_str = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
