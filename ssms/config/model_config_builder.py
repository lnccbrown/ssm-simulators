"""
Utilities for building custom model configurations.

This module provides helper classes and functions for creating valid
model configurations for use with the Simulator class.
"""

from collections.abc import Callable

from ssms.config.model_registry import get_model_registry
from ssms.config.boundary_registry import get_boundary_registry
from ssms.config.drift_registry import get_drift_registry


class ModelConfigBuilder:
    """Helper class for building custom model configurations.

    This class provides static methods for creating model configurations
    in various ways:
    - Starting from an existing model and overriding specific values
    - Building a configuration from scratch for fully custom simulators
    - Creating minimal valid configurations
    - Validating configurations

    Examples
    --------
    Start from existing model and override:

    >>> config = ModelConfigBuilder.from_model("ddm",
    ...                                     param_bounds=[[-4, 0.3, 0.1, 0],
    ...                                                   [4, 3.0, 0.9, 2.0]])

    Build from scratch:

    >>> config = ModelConfigBuilder.from_scratch(
    ...     name="my_model",
    ...     params=["v", "a", "z", "t"],
    ...     simulator_function=my_sim_fn,
    ...     nchoices=2
    ... )

    Create minimal configuration:

    >>> config = ModelConfigBuilder.minimal_config(
    ...     params=["v", "a"],
    ...     simulator_function=my_sim_fn,
    ...     nchoices=2
    ... )
    """

    @staticmethod
    def from_model(model_name: str, **overrides) -> dict:
        """Create configuration starting from an existing model.

        Parameters
        ----------
        model_name : str
            Name of the base model (e.g., "ddm", "angle", "lca_3")
        **overrides
            Configuration fields to override. Common options:
            - params : list[str] - Parameter names
            - param_bounds : list - [[lower bounds], [upper bounds]]
            - boundary : Callable - Custom boundary function
            - boundary_name : str - Boundary name
            - boundary_params : list[str] - Boundary parameter names
            - drift : Callable - Custom drift function
            - drift_name : str - Drift name
            - drift_params : list[str] - Drift parameter names
            - simulator : Callable - Custom simulator function
            - nchoices : int - Number of choices
            - choices : list - Possible choice values

        Returns
        -------
        dict
            Configuration dictionary

        Raises
        ------
        ValueError
            If model_name is not recognized

        Examples
        --------
        >>> config = ModelConfigBuilder.from_model("ddm",
        ...                                     param_bounds=[[-4, 0.3, 0.1, 0],
        ...                                                   [4, 3.0, 0.9, 2.0]])
        """
        registry = get_model_registry()
        if not registry.has_model(model_name):
            available = registry.list_models()
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available[:10]}... ({len(available)} total)"
            )

        config = registry.get(model_name)  # Already returns a deep copy
        config.update(overrides)

        return config

    @staticmethod
    def from_scratch(
        name: str,
        params: list[str],
        simulator_function: Callable,
        nchoices: int,
        **config,
    ) -> dict:
        """Build a complete configuration from scratch.

        Use this method when creating a fully custom simulator that doesn't
        build on any existing model.

        Parameters
        ----------
        name : str
            Model name
        params : list[str]
            List of parameter names
        simulator_function : Callable
            Simulator function
        nchoices : int
            Number of choices
        **config
            Additional configuration fields:
            - param_bounds : list - [[lower bounds], [upper bounds]]
            - default_params : list - Default parameter values
            - choices : list - Possible choice values
            - n_particles : int - Number of particles (default 1)
            - boundary : Callable - Boundary function
            - boundary_name : str - Boundary name
            - boundary_params : list[str] - Boundary parameter names (including 'a')
            - drift : Callable - Drift function
            - drift_name : str - Drift name
            - drift_params : list[str] - Drift parameter names

        Returns
        -------
        dict
            Complete configuration dictionary

        Examples
        --------
        >>> def my_sim(v, a, **kwargs):
        ...     # Custom simulation logic
        ...     return {'rts': ..., 'choices': ..., 'metadata': ...}
        >>>
        >>> config = ModelConfigBuilder.from_scratch(
        ...     name="my_custom_model",
        ...     params=["v", "a"],
        ...     simulator_function=my_sim,
        ...     nchoices=2,
        ...     param_bounds=[[-2, 0.5], [2, 2.0]],
        ...     default_params=[0.0, 1.0]
        ... )
        """
        base_config = {
            "name": name,
            "params": params,
            "n_params": len(params),
            "nchoices": nchoices,
            "choices": config.get("choices", list(range(nchoices))),
            "n_particles": config.get("n_particles", 1),
            "simulator": simulator_function,
        }

        # Add optional fields if provided
        optional_fields = [
            "param_bounds",
            "default_params",
            "boundary",
            "boundary_name",
            "boundary_params",
            "drift",
            "drift_name",
            "drift_params",
        ]

        for field in optional_fields:
            if field in config:
                base_config[field] = config[field]

        return base_config

    @staticmethod
    def minimal_config(
        params: list[str],
        simulator_function: Callable,
        nchoices: int = 2,
        name: str = "custom",
    ) -> dict:
        """Create a minimal valid configuration.

        This is the simplest way to create a configuration for a custom simulator.
        It includes only the required fields.

        Parameters
        ----------
        params : list[str]
            List of parameter names
        simulator_function : Callable
            Simulator function
        nchoices : int, default=2
            Number of choices
        name : str, default="custom"
            Model name

        Returns
        -------
        dict
            Minimal configuration dictionary

        Examples
        --------
        >>> config = ModelConfigBuilder.minimal_config(
        ...     params=["v", "a", "z", "t"],
        ...     simulator_function=my_sim_fn,
        ...     nchoices=2
        ... )
        """
        return {
            "name": name,
            "params": params,
            "n_params": len(params),
            "nchoices": nchoices,
            "choices": list(range(nchoices)),
            "n_particles": 1,
            "simulator": simulator_function,
        }

    @staticmethod
    def validate_config(config: dict, strict: bool = False) -> tuple[bool, list[str]]:
        """Validate a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration to validate
        strict : bool, default=False
            If True, also check for recommended optional fields

        Returns
        -------
        is_valid : bool
            Whether the configuration is valid
        errors : list[str]
            List of error messages (empty if valid)

        Examples
        --------
        >>> config = {"params": ["v", "a"], "nchoices": 2}
        >>> is_valid, errors = ModelConfigBuilder.validate_config(config)
        >>> if not is_valid:
        ...     print("Errors:", errors)
        """
        errors = []

        # Check required fields
        required_fields = {
            "params": list,
            "nchoices": int,
            "simulator": Callable,
        }

        for field, expected_type in required_fields.items():
            if field not in config:
                errors.append(f"Missing required field: '{field}'")
            elif not isinstance(config[field], expected_type):
                if field == "simulator" and not callable(config[field]):
                    errors.append(
                        f"Field '{field}' must be callable, got {type(config[field])}"
                    )
                else:
                    errors.append(
                        f"Field '{field}' has wrong type: expected {expected_type}, "
                        f"got {type(config[field])}"
                    )

        # Check consistency
        if "params" in config and "n_params" in config:
            if config["n_params"] != len(config["params"]):
                errors.append(
                    f"Inconsistent n_params: expected {len(config['params'])}, "
                    f"got {config['n_params']}"
                )

        if "param_bounds" in config and "params" in config:
            bounds = config["param_bounds"]
            if len(bounds) != 2:
                errors.append(
                    "param_bounds must have exactly 2 elements: [lower_bounds, upper_bounds]"
                )
            elif len(bounds[0]) != len(config["params"]) or len(bounds[1]) != len(
                config["params"]
            ):
                errors.append(
                    f"param_bounds must match number of params ({len(config['params'])})"
                )

        if "default_params" in config and "params" in config:
            if len(config["default_params"]) != len(config["params"]):
                errors.append(
                    f"default_params must match number of params ({len(config['params'])})"
                )

        # Strict mode: check recommended fields
        if strict:
            recommended = ["name", "n_particles", "choices", "param_bounds"]
            for field in recommended:
                if field not in config:
                    errors.append(f"Recommended field missing: '{field}'")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def add_boundary(
        config: dict,
        boundary: str | Callable,
        boundary_params: list[str] | None = None,
    ) -> dict:
        """Add or replace boundary function in configuration.

        Parameters
        ----------
        config : dict
            Configuration to modify
        boundary : str or Callable
            Boundary function name or callable
        boundary_params : list[str] or None
            Parameter names for boundary function (required if boundary is callable)
        multiplicative : bool, default=True
            Whether boundary is multiplicative (True) or additive (False)

        Returns
        -------
        dict
            Modified configuration (note: modifies in place and returns)

        Raises
        ------
        ValueError
            If boundary specification is invalid

        Examples
        --------
        >>> config = ModelConfigBuilder.from_model("ddm")
        >>> config = ModelConfigBuilder.add_boundary(config, "angle")
        """
        if isinstance(boundary, str):
            registry = get_boundary_registry()
            if not registry.is_registered(boundary):
                available = registry.list_boundaries()
                raise ValueError(
                    f"Unknown boundary '{boundary}'. Available: {available}"
                )
            boundary_spec = registry.get(boundary)
            config["boundary_name"] = boundary
            config["boundary"] = boundary_spec["fun"]
            config["boundary_params"] = boundary_spec["params"]
        elif callable(boundary):
            if boundary_params is None:
                raise ValueError(
                    "Must provide boundary_params when using custom boundary function"
                )
            config["boundary"] = boundary
            config["boundary_name"] = getattr(boundary, "__name__", "custom")
            config["boundary_params"] = boundary_params
        else:
            raise ValueError("boundary must be string name or callable")

        return config

    @staticmethod
    def add_drift(
        config: dict,
        drift: str | Callable,
        drift_params: list[str] | None = None,
    ) -> dict:
        """Add or replace drift function in configuration.

        Parameters
        ----------
        config : dict
            Configuration to modify
        drift : str or Callable
            Drift function name or callable
        drift_params : list[str] or None
            Parameter names for drift function (required if drift is callable)

        Returns
        -------
        dict
            Modified configuration (note: modifies in place and returns)

        Raises
        ------
        ValueError
            If drift specification is invalid

        Examples
        --------
        >>> config = ModelConfigBuilder.from_model("ddm")
        >>> config = ModelConfigBuilder.add_drift(config, "gamma_drift")
        """
        if isinstance(drift, str):
            registry = get_drift_registry()
            if not registry.is_registered(drift):
                available = registry.list_drifts()
                raise ValueError(f"Unknown drift '{drift}'. Available: {available}")
            drift_spec = registry.get(drift)
            config["drift_name"] = drift
            config["drift"] = drift_spec["fun"]
            config["drift_params"] = drift_spec["params"]
        elif callable(drift):
            if drift_params is None:
                raise ValueError(
                    "Must provide drift_params when using custom drift function"
                )
            config["drift"] = drift
            config["drift_name"] = getattr(drift, "__name__", "custom")
            config["drift_params"] = drift_params
        else:
            raise ValueError("drift must be string name or callable")

        return config
