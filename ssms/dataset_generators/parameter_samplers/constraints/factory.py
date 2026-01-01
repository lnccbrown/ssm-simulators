"""Factory for creating constraint instances from config dictionaries."""

from ssms.dataset_generators.parameter_samplers.constraints.swap import (
    SwapIfLessConstraint,
)
from ssms.dataset_generators.parameter_samplers.constraints.normalize import (
    NormalizeToSumConstraint,
)
from ssms.dataset_generators.parameter_samplers.constraints.registry import (
    get_registry,
)
from ssms.dataset_generators.parameter_samplers.constraints.adapters import (
    FunctionConstraintAdapter,
)


def create_constraint_from_config(config: dict):
    """Create a constraint instance from a config dictionary.

    First checks built-in types (swap, normalize), then checks the
    global registry for custom constraints.

    Args:
        config: Dictionary with 'type' key and type-specific parameters.
                For 'swap': requires 'param_a' and 'param_b'
                For 'normalize': requires 'param_names' (list of strings)
                For custom constraints: parameters depend on the constraint

    Returns:
        Constraint instance (built-in or custom)

    Raises:
        ValueError: If constraint type is unknown or required parameters are missing

    Example:
        >>> # Built-in constraint
        >>> config = {"type": "swap", "param_a": "a", "param_b": "z"}
        >>> constraint = create_constraint_from_config(config)
        >>> isinstance(constraint, SwapIfLessConstraint)
        True

        >>> # Custom registered constraint
        >>> from ssms.dataset_generators.parameter_samplers import register_constraint_function
        >>> register_constraint_function("my_constraint", lambda theta: theta)
        >>> config = {"type": "my_constraint"}
        >>> constraint = create_constraint_from_config(config)
    """
    constraint_type = config.get("type")

    # Check built-in types first
    if constraint_type == "swap":
        if "param_a" not in config or "param_b" not in config:
            raise ValueError(
                "Swap constraint requires 'param_a' and 'param_b' in config"
            )
        return SwapIfLessConstraint(
            param_a=config["param_a"],
            param_b=config["param_b"],
        )
    elif constraint_type == "normalize":
        if "param_names" not in config:
            raise ValueError("Normalize constraint requires 'param_names' in config")
        return NormalizeToSumConstraint(
            param_names=config["param_names"],
        )

    # Check registry for custom constraints
    registry = get_registry()

    if registry.is_registered(constraint_type):
        constraint_or_func = registry.get(constraint_type)

        # If it's a function, wrap it in an adapter
        if callable(constraint_or_func) and not isinstance(constraint_or_func, type):
            return FunctionConstraintAdapter(constraint_or_func, constraint_type)

        # If it's a class, instantiate with config params (excluding 'type')
        params = {k: v for k, v in config.items() if k != "type"}
        return constraint_or_func(**params)

    # Unknown constraint type
    raise ValueError(
        f"Unknown constraint type: '{constraint_type}'. "
        f"Supported built-in types: 'swap', 'normalize'. "
        f"Registered custom types: {registry.list_constraints()}"
    )


def get_constraints_from_model_config(model_config: dict) -> list:
    """Extract and instantiate constraints from a model config.

    Looks for a 'parameter_sampling_constraints' field in the model config and
    instantiates each constraint using create_constraint_from_config().

    Args:
        model_config: Model configuration dictionary (may contain 'parameter_sampling_constraints')

    Returns:
        List of constraint instances (empty list if no constraints defined)

    Example:
        >>> model_config = {
        ...     "name": "lba_angle_3",
        ...     "parameter_sampling_constraints": [
        ...         {"type": "swap", "param_a": "a", "param_b": "z"}
        ...     ]
        ... }
        >>> constraints = get_constraints_from_model_config(model_config)
        >>> len(constraints)
        1
    """
    constraint_configs = model_config.get("parameter_sampling_constraints", [])
    return [create_constraint_from_config(cfg) for cfg in constraint_configs]
