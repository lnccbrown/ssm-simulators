"""Global registry for custom parameter sampling constraints."""

from typing import Callable, Dict, Type, Union

from ssms.dataset_generators.parameter_samplers.protocols import (
    ParameterSamplingConstraintProtocol,
)


class ConstraintRegistry:
    """Global registry for custom parameter sampling constraints.

    Supports registration of:
    - Constraint classes (must implement ParameterSamplingConstraintProtocol)
    - Simple functions (will be wrapped automatically)

    Examples:
        # Register a class
        registry.register_class("my_constraint", MyConstraintClass)

        # Register a function
        registry.register_function("clip_v", lambda theta: {...})
    """

    def __init__(self):
        self._constraints: Dict[str, Type[ParameterSamplingConstraintProtocol]] = {}
        self._function_constraints: Dict[str, Callable] = {}

    def register_class(
        self,
        name: str,
        constraint_class: Type[ParameterSamplingConstraintProtocol],
    ) -> None:
        """Register a constraint class.

        Args:
            name: Unique name for the constraint
            constraint_class: Class implementing ParameterSamplingConstraintProtocol

        Raises:
            ValueError: If name already registered
        """
        if name in self._constraints or name in self._function_constraints:
            raise ValueError(f"Constraint '{name}' already registered")
        self._constraints[name] = constraint_class

    def register_function(
        self,
        name: str,
        func: Callable[[dict], dict],
        description: str = "",
    ) -> None:
        """Register a simple function as a constraint.

        Args:
            name: Unique name for the constraint
            func: Function that takes and returns theta dict
            description: Optional description for documentation

        Raises:
            ValueError: If name already registered
        """
        if name in self._constraints or name in self._function_constraints:
            raise ValueError(f"Constraint '{name}' already registered")
        self._function_constraints[name] = func

    def get(
        self, name: str
    ) -> Union[Type[ParameterSamplingConstraintProtocol], Callable]:
        """Get a registered constraint by name.

        Args:
            name: Name of the registered constraint

        Returns:
            Constraint class or function

        Raises:
            KeyError: If constraint name not registered
        """
        if name in self._constraints:
            return self._constraints[name]
        if name in self._function_constraints:
            return self._function_constraints[name]
        raise KeyError(f"Constraint '{name}' not registered")

    def is_registered(self, name: str) -> bool:
        """Check if a constraint name is registered.

        Args:
            name: Name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._constraints or name in self._function_constraints

    def list_constraints(self) -> list[str]:
        """List all registered constraint names.

        Returns:
            List of all registered constraint names
        """
        return list(self._constraints.keys()) + list(self._function_constraints.keys())


# Global registry instance
_GLOBAL_REGISTRY = ConstraintRegistry()


def register_constraint_class(
    name: str,
    constraint_class: Type[ParameterSamplingConstraintProtocol],
) -> None:
    """Register a custom constraint class globally.

    Args:
        name: Unique name for the constraint
        constraint_class: Class implementing ParameterSamplingConstraintProtocol

    Raises:
        ValueError: If name already registered

    Example:
        class MyConstraint:
            def __init__(self, param_name: str):
                self.param_name = param_name

            def apply(self, theta: dict) -> dict:
                # Custom logic here
                return theta

        register_constraint_class("my_constraint", MyConstraint)
    """
    _GLOBAL_REGISTRY.register_class(name, constraint_class)


def register_constraint_function(
    name: str,
    func: Callable[[dict], dict],
    description: str = "",
) -> None:
    """Register a custom constraint function globally.

    Args:
        name: Unique name for the constraint
        func: Function that takes and returns theta dict
        description: Optional description for documentation

    Raises:
        ValueError: If name already registered

    Example:
        def clip_v(theta: dict) -> dict:
            if 'v' in theta:
                theta['v'] = np.clip(theta['v'], -5, 5)
            return theta

        register_constraint_function("clip_v", clip_v)
    """
    _GLOBAL_REGISTRY.register_function(name, func, description)


def get_registry() -> ConstraintRegistry:
    """Get the global constraint registry.

    Returns:
        The global ConstraintRegistry instance
    """
    return _GLOBAL_REGISTRY
