"""
Base classes for theta transformations.

This module defines the abstract base class that all theta transformations
must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any


class ThetaTransformation(ABC):
    """Abstract base class for theta parameter transformations.

    A theta transformation is a single, focused operation that modifies
    the theta parameter dictionary in some way. Transformations can be
    composed into pipelines to handle complex parameter processing logic.

    Subclasses must implement the `apply` method which performs the actual
    transformation.

    Examples
    --------
    Create a custom transformation:

    >>> class ScaleParameter(ThetaTransformation):
    ...     def __init__(self, param_name: str, scale: float):
    ...         self.param_name = param_name
    ...         self.scale = scale
    ...
    ...     def apply(self, theta, model_config, n_trials):
    ...         if self.param_name in theta:
    ...             theta[self.param_name] = theta[self.param_name] * self.scale
    ...         return theta
    """

    @abstractmethod
    def apply(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """Apply transformation to theta parameters.

        This method should modify the theta dictionary in place and return it.
        It may add new parameters, modify existing parameters, or remove
        parameters as needed.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of model parameters. Values are typically numpy arrays.
        model_config : dict[str, Any]
            Model configuration dictionary containing model metadata.
        n_trials : int
            Number of trials to be simulated.

        Returns
        -------
        dict[str, Any]
            The modified theta dictionary (usually the same object that was passed in).

        Notes
        -----
        - Transformations should be pure when possible (no side effects)
        - If creating new arrays, use dtype=np.float32 for consistency
        - Document any parameters that are added, modified, or removed
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns
        -------
        str
            A string describing this transformation.
        """
        # Default implementation shows class name and attributes
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                # Truncate long values
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                attrs.append(f"{key}={value!r}")

        attrs_str = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
