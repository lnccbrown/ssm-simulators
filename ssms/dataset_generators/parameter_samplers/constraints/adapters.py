"""Adapters for wrapping functions as constraints."""

from typing import Callable


class FunctionConstraintAdapter:
    """Adapter to wrap a simple function as a ParameterSamplingConstraintProtocol.

    This allows users to register simple functions without needing to
    create full constraint classes.

    Example:
        def clip_v(theta):
            if 'v' in theta:
                theta['v'] = np.clip(theta['v'], -5, 5)
            return theta

        adapter = FunctionConstraintAdapter(clip_v, "clip_v")
        result = adapter.apply(theta)
    """

    def __init__(self, func: Callable[[dict], dict], name: str = "custom"):
        """Initialize the adapter.

        Args:
            func: Function that takes theta dict and returns modified theta
            name: Name for debugging/logging
        """
        self.func = func
        self.name = name

    def apply(self, theta: dict) -> dict:
        """Apply the wrapped function.

        Args:
            theta: Dictionary of parameter values

        Returns:
            Modified theta dictionary
        """
        return self.func(theta)

    def __repr__(self):
        return f"FunctionConstraintAdapter('{self.name}')"
