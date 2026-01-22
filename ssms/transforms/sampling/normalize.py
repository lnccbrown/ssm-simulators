"""Normalize transform for drift rate constraints."""

from typing import Any

from ssms.transforms.base import ParameterTransform


class NormalizeToSumConstraint(ParameterTransform):
    """Normalize a set of parameters to sum to 1.

    This constraint is commonly used for drift rates in multi-accumulator models
    (e.g., RLWM models) where the drift rates must sum to 1.

    The normalization includes a small epsilon (1e-20) to avoid division by zero
    and ensure numerical stability.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names to normalize together

    Examples
    --------
    >>> constraint = NormalizeToSumConstraint(["v1", "v2", "v3"])
    >>> theta = {"v1": 0.2, "v2": 0.3, "v3": 0.5}
    >>> result = constraint.apply(theta)
    >>> sum([result["v1"], result["v2"], result["v3"]])
    1.0

    Use in model config:

    >>> model_config = {
    ...     "name": "dev_rlwm_lba_race_v1",
    ...     "parameter_transforms": {
    ...         "sampling": [
    ...             NormalizeToSumConstraint(["vRL0", "vRL1", "vRL2"]),
    ...             NormalizeToSumConstraint(["vWM0", "vWM1", "vWM2"]),
    ...         ],
    ...     }
    ... }
    """

    def __init__(self, param_names: list[str]):
        """Initialize the normalize constraint.

        Parameters
        ----------
        param_names : list[str]
            List of parameter names to normalize together
        """
        self.param_names = param_names

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
        epsilon: float = 1e-20,
    ) -> dict[str, Any]:
        """Normalize specified parameters to sum to 1.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of parameter values (can be scalars or arrays)
        model_config : dict[str, Any] or None, optional
            Not used by this transform (included for interface compatibility)
        n_trials : int or None, optional
            Not used by this transform (included for interface compatibility)
        epsilon : float, optional
            Small value to add to the total to avoid division by zero.
            Default is 1e-20.

        Returns
        -------
        dict[str, Any]
            Modified theta dictionary with normalized values
        """
        # Check all parameters exist before normalizing
        if all(p in theta for p in self.param_names):
            # Calculate total with small epsilon for numerical stability
            total = sum(theta[p] for p in self.param_names) + epsilon

            # Normalize each parameter
            for param in self.param_names:
                theta[param] = (theta[param] + epsilon / len(self.param_names)) / total

        return theta
