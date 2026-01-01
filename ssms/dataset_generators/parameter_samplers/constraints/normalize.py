"""Normalize constraint for drift rate constraints."""


class NormalizeToSumConstraint:
    """Normalize a set of parameters to sum to 1.

    This constraint is commonly used for drift rates in multi-accumulator models
    (e.g., RLWM models) where the drift rates must sum to 1.

    The normalization includes a small epsilon (1e-20) to avoid division by zero
    and ensure numerical stability.

    Example:
        >>> constraint = NormalizeToSumConstraint(["v1", "v2", "v3"])
        >>> theta = {"v1": 0.2, "v2": 0.3, "v3": 0.5}
        >>> result = constraint.apply(theta)
        >>> sum([result["v1"], result["v2"], result["v3"]])
        1.0
    """

    def __init__(self, param_names: list[str]):
        """Initialize the normalize constraint.

        Args:
            param_names: List of parameter names to normalize together
        """
        self.param_names = param_names

    def apply(self, theta: dict, epsilon: float = 1e-20) -> dict:
        """Normalize specified parameters to sum to 1.

        Args:
            theta: Dictionary of parameter values (can be scalars or arrays)
            epsilon: Small value to add to the total to avoid division by zero
        Returns:
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
