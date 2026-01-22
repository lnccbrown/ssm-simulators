"""Protocols for parameter sampling and constraints."""

from typing import Protocol
import numpy as np


class ParameterSamplerProtocol(Protocol):
    """Protocol for parameter sampling strategies.

    This protocol defines the interface that all parameter samplers must implement.
    Samplers are responsible for generating parameter sets from a parameter space,
    potentially with dependencies between parameters and transformations applied.
    """

    def sample(self, n_samples: int = 1) -> dict[str, np.ndarray]:
        """Sample n_samples parameter sets.

        Args:
            n_samples: Number of parameter sets to sample

        Returns:
            Dictionary mapping parameter names to sampled values (arrays of length n_samples)
        """
        ...

    def get_param_space(self) -> dict[str, tuple]:
        """Get the parameter space bounds.

        Returns:
            Dictionary mapping parameter names to (lower, upper) bound tuples
        """
        ...


class ParameterSamplingConstraintProtocol(Protocol):
    """Protocol for parameter sampling constraints.

    Constraints are applied after sampling to enforce relationships or modify
    parameter values (e.g., ensuring a > z, normalizing drift rates to sum to 1).
    Constraints can trigger resampling if validation fails.
    """

    def apply(self, theta: dict) -> dict:
        """Apply constraint to sampled parameters.

        Args:
            theta: Dictionary of parameter values (can be scalars or arrays)

        Returns:
            Constrained parameter dictionary (modified in place or new dict)
        """
        ...
