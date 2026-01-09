"""
Utilities for building custom model configurations.

This module provides helper classes and functions for creating valid
model configurations for use with the Simulator class.
"""

from collections.abc import Callable
from copy import deepcopy

from ssms.config.model_registry import get_model_registry
from ssms.config.boundary_registry import get_boundary_registry
from ssms.config.drift_registry import get_drift_registry


# Centralized configuration for the deadline parameter variant
# These values are used when adding deadline support to any model
DEADLINE_PARAM_CONFIG = {
    "name": "deadline",
    "default_bounds": (0.001, 10.0),
    "default_value": 10.0,
}


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

        This method automatically parses variant suffixes like "_deadline"
        from the model name and applies the appropriate transformations.

        Parameters
        ----------
        model_name : str
            Name of the model, optionally with variant suffixes.
            Examples: "ddm", "angle", "ddm_deadline", "angle_deadline"
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
            Configuration dictionary with any variants applied

        Raises
        ------
        ValueError
            If the base model name is not recognized

        Examples
        --------
        >>> config = ModelConfigBuilder.from_model("ddm",
        ...                                     param_bounds=[[-4, 0.3, 0.1, 0],
        ...                                                   [4, 3.0, 0.9, 2.0]])

        >>> # With deadline variant
        >>> config = ModelConfigBuilder.from_model("ddm_deadline")
        >>> "deadline" in config["params"]
        True
        """
        # Parse variant suffixes from model name
        base_model = model_name
        has_deadline = False

        if "_deadline" in model_name:
            has_deadline = True
            base_model = model_name.replace("_deadline", "")

        # Look up base model in registry
        registry = get_model_registry()
        if not registry.has_model(base_model):
            available = registry.list_models()
            raise ValueError(
                f"Unknown model '{base_model}'. "
                f"Available models: {available[:10]}... ({len(available)} total)"
            )

        config = registry.get(base_model)  # Already returns a deep copy
        config.update(overrides)

        # If param_bounds was overridden, update param_bounds_dict accordingly
        if "param_bounds" in overrides:
            from ssms.config._modelconfig import _normalize_param_bounds

            # Remove old param_bounds_dict to force re-normalization
            config.pop("param_bounds_dict", None)
            config = _normalize_param_bounds(config)

        # Apply deadline variant if requested
        if has_deadline:
            config = ModelConfigBuilder.with_deadline(config)

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

    @staticmethod
    def with_deadline(config: dict) -> dict:
        """Add deadline parameter to a model configuration.

        Creates a NEW configuration with the deadline parameter added.
        This is an immutable operation - the original config is not modified.

        The deadline parameter allows models to incorporate response deadlines,
        where trials are terminated if no response is made within the deadline.

        This method is idempotent - calling it on a config that already has
        the deadline parameter will return an equivalent config.

        Parameters
        ----------
        config : dict
            Model configuration to extend with deadline support

        Returns
        -------
        dict
            New configuration with deadline parameter added. Includes:
            - "deadline" appended to params list
            - Updated param_bounds (both list and dict formats)
            - Updated default_params
            - "_deadline" suffix added to name
            - Incremented n_params
            - "deadline" metadata flag set to True

        Examples
        --------
        >>> base_config = ModelConfigBuilder.from_model("ddm")
        >>> deadline_config = ModelConfigBuilder.with_deadline(base_config)
        >>> "deadline" in deadline_config["params"]
        True
        >>> deadline_config["name"]
        'ddm_deadline'
        """
        # Check if deadline is already present (idempotent)
        if "deadline" in config.get("params", []):
            return deepcopy(config)

        # Create a deep copy to avoid mutating the original
        new_config = deepcopy(config)

        deadline_name = DEADLINE_PARAM_CONFIG["name"]
        deadline_bounds = DEADLINE_PARAM_CONFIG["default_bounds"]
        deadline_default = DEADLINE_PARAM_CONFIG["default_value"]

        # Add deadline to params list
        new_config["params"] = new_config.get("params", []) + [deadline_name]

        # Update n_params
        new_config["n_params"] = new_config.get("n_params", 0) + 1

        # Update name with _deadline suffix (if not already present)
        current_name = new_config.get("name", "")
        if not current_name.endswith("_deadline"):
            new_config["name"] = current_name + "_deadline"

        # Update param_bounds based on format (list or dict)
        if "param_bounds" in new_config:
            bounds = new_config["param_bounds"]
            if isinstance(bounds, list) and len(bounds) == 2:
                # List format: [[lower bounds], [upper bounds]]
                new_config["param_bounds"] = [
                    bounds[0] + [deadline_bounds[0]],
                    bounds[1] + [deadline_bounds[1]],
                ]
            elif isinstance(bounds, dict):
                # Dict format: {param: (lower, upper)}
                new_config["param_bounds"] = {**bounds, deadline_name: deadline_bounds}

        # Update param_bounds_dict
        if "param_bounds_dict" in new_config:
            new_config["param_bounds_dict"] = {
                **new_config["param_bounds_dict"],
                deadline_name: deadline_bounds,
            }
        elif "param_bounds" in new_config:
            # Regenerate param_bounds_dict from updated param_bounds
            from ssms.config._modelconfig import _normalize_param_bounds

            # Remove existing to force regeneration
            new_config.pop("param_bounds_dict", None)
            new_config = _normalize_param_bounds(new_config)

        # Update default_params
        if "default_params" in new_config:
            new_config["default_params"] = new_config["default_params"] + [
                deadline_default
            ]

        # Add metadata flag indicating this is a deadline variant
        new_config["deadline"] = True

        return new_config

    @staticmethod
    def get_transforms(config: dict, phase: str) -> list:
        """Get transforms for a specific phase from model config.

        This method extracts parameter transforms from the unified
        `parameter_transforms` field in the model configuration.

        Parameters
        ----------
        config : dict
            Model configuration dictionary
        phase : str
            Either 'sampling' or 'simulation'

        Returns
        -------
        list
            List of transform instances (empty if none defined)

        Examples
        --------
        >>> config = {
        ...     "name": "lba_angle_3",
        ...     "parameter_transforms": {
        ...         "sampling": [SwapIfLessConstraint("a", "z")],
        ...         "simulation": [ColumnStackParameters(["v0", "v1"], "v")],
        ...     }
        ... }
        >>> sampling_transforms = ModelConfigBuilder.get_transforms(config, "sampling")
        >>> len(sampling_transforms)
        1
        """
        if phase not in ("sampling", "simulation"):
            raise ValueError(f"phase must be 'sampling' or 'simulation', got '{phase}'")

        transforms = config.get("parameter_transforms", {})
        return transforms.get(phase, [])

    @staticmethod
    def get_sampling_transforms(config: dict) -> list:
        """Get parameter sampling transforms from model config.

        These transforms are applied during the parameter sampling stage of the
        training data generation workflow. They enforce parameter relationships
        (e.g., a > z) when generating synthetic training data for likelihood
        approximation networks.

        Note: These are NOT directly relevant for basic Simulator usage, which
        uses simulation transforms via ParameterSimulatorAdapters instead.

        Parameters
        ----------
        config : dict
            Model configuration dictionary

        Returns
        -------
        list
            List of transform/constraint instances (empty if none defined)

        Examples
        --------
        >>> config = {
        ...     "parameter_transforms": {
        ...         "sampling": [SwapIfLessConstraint("a", "z")],
        ...     }
        ... }
        >>> transforms = ModelConfigBuilder.get_sampling_transforms(config)
        """
        return ModelConfigBuilder.get_transforms(config, "sampling")

    @staticmethod
    def get_simulation_transforms(config: dict) -> list:
        """Get simulation transforms from model config.

        These transforms are applied via ParameterSimulatorAdapters when running
        the basic Simulator. They prepare user-provided parameters for the
        low-level C/Cython simulators (e.g., stacking v0, v1, v2 into a single
        v array, expanding dimensions).

        Parameters
        ----------
        config : dict
            Model configuration dictionary

        Returns
        -------
        list
            List of transform instances (empty if none defined)

        Examples
        --------
        >>> config = {
        ...     "parameter_transforms": {
        ...         "simulation": [
        ...             ColumnStackParameters(["v0", "v1", "v2"], "v"),
        ...             ExpandDimension(["a", "z"]),
        ...         ],
        ...     }
        ... }
        >>> transforms = ModelConfigBuilder.get_simulation_transforms(config)
        >>> len(transforms)
        2
        """
        return ModelConfigBuilder.get_transforms(config, "simulation")
