"""Swap constraint for ensuring parameter ordering constraints."""

import numpy as np


class SwapIfLessConstraint:
    """Swap two parameters if first is less than or equal to second.

    This constraint ensures ordering relationships like a > z by swapping
    the values when the constraint is violated. Works with both scalar
    and array inputs.

    Example:
        >>> constraint = SwapIfLessConstraint("a", "z")
        >>> theta = {"a": np.array([0.5, 1.5]), "z": np.array([1.0, 0.5])}
        >>> result = constraint.apply(theta)
        >>> # First element: a=0.5 <= z=1.0 → swap → a=1.0, z=0.5
        >>> # Second element: a=1.5 > z=0.5 → no swap → a=1.5, z=0.5
    """

    def __init__(self, param_a: str, param_b: str):
        """Initialize the swap constraint.

        Args:
            param_a: First parameter name (should be > param_b after constraint)
            param_b: Second parameter name (should be < param_a after constraint)
        """
        self.param_a = param_a
        self.param_b = param_b

    def apply(self, theta: dict) -> dict:
        """Swap param_a and param_b where param_a <= param_b.

        Args:
            theta: Dictionary of parameter values (can be scalars or arrays)

        Returns:
            Modified theta dictionary with swapped values where needed
        """
        if self.param_a in theta and self.param_b in theta:
            a_vals = theta[self.param_a]
            b_vals = theta[self.param_b]

            # Handle both scalar and array cases
            if np.isscalar(a_vals):
                # Scalar case
                if a_vals <= b_vals:
                    theta[self.param_a], theta[self.param_b] = b_vals, a_vals
            else:
                # Array case: swap elements where a <= b
                swap_mask = a_vals <= b_vals
                theta[self.param_a] = np.where(swap_mask, b_vals, a_vals)
                theta[self.param_b] = np.where(swap_mask, a_vals, b_vals)

        return theta
