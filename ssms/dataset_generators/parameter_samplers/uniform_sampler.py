"""Uniform random parameter sampler (default behavior)."""

import numpy as np
from ssms.dataset_generators.parameter_samplers.base_sampler import (
    AbstractParameterSampler,
)


class UniformParameterSampler(AbstractParameterSampler):
    """Uniform random sampling of parameters.

    This sampler generates parameter values uniformly distributed between
    lower and upper bounds. It matches the current behavior of the
    sample_parameters_from_constraints() function.

    Example:
        >>> param_space = {"v": (-1.0, 1.0), "a": (0.5, 2.0)}
        >>> sampler = UniformParameterSampler(param_space)
        >>> samples = sampler.sample(n_samples=100)
        >>> samples["v"].shape
        (100,)
    """

    def _sample_parameter(
        self,
        param: str,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample uniformly between lower and upper bounds.

        Args:
            param: Name of the parameter (unused, kept for interface compatibility)
            lower: Lower bound (scalar or array)
            upper: Upper bound (scalar or array)
            n_samples: Number of samples to generate
            rng: Random number generator to use for sampling

        Returns:
            Array of uniformly sampled values (length n_samples)
        """
        # Ensure lower and upper are arrays of the correct size
        lower_array = np.full(n_samples, lower) if np.isscalar(lower) else lower
        upper_array = np.full(n_samples, upper) if np.isscalar(upper) else upper

        # Sample uniformly within bounds using provided RNG
        return rng.uniform(low=lower_array, high=upper_array, size=n_samples).astype(
            np.float32
        )
