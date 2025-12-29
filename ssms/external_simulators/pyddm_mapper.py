"""Mapper for converting ssms model configurations to PyDDM models.

This module provides utilities for mapping ssms model configurations
to PyDDM models, enabling the use of PyDDM's analytical Fokker-Planck
solver for compatible models.
"""

import numpy as np
from typing import Any, Callable


class SSMSToPyDDMMapper:
    """Maps ssms model configurations to PyDDM models.

    PyDDM can solve the Fokker-Planck equation for:
    - Single-particle, two-choice models
    - Gaussian noise only
    - Arbitrary drift functions (time-dependent, position-dependent, parameter-dependent)
    - Arbitrary boundary functions (time-dependent, parameter-dependent)

    Compatible models include:
    - ddm and variants (without sv/sz/st)
    - angle, weibull (collapsing boundaries)
    - ornstein (leaky/unstable integration)
    - gamma_drift (time-dependent drift)
    - conflict_*, attend_*, shrink_spot_* (custom drift models)

    Incompatible models:
    - race_*, lca_*, lba* (multi-particle)
    - levy (non-Gaussian noise)
    - full_ddm, ddm_sdv, ddm_st (inter-trial variability)
    """

    # Models that use multi-particle dynamics
    MULTI_PARTICLE_MODELS = {
        "race",
        "lca",
        "lba",
    }

    # Models with non-Gaussian noise
    NON_GAUSSIAN_NOISE_MODELS = {
        "levy",
    }

    # Parameters indicating inter-trial variability
    VARIABILITY_PARAMS = {"sv", "sz", "st"}

    @classmethod
    def is_compatible(cls, model_config: dict[str, Any]) -> tuple[bool, str]:
        """Check if model can use PyDDM analytical solver.

        Args:
            model_config: Model configuration dictionary from ssms.config

        Returns:
            (is_compatible, reason_if_not)
        """
        model_name = model_config.get("name", "")

        # Check for multi-particle models
        for family in cls.MULTI_PARTICLE_MODELS:
            if model_name.startswith(family):
                n_choices = model_config.get("n_choices", 2)
                return (
                    False,
                    f"Multi-particle model ({family}, {n_choices} accumulators)",
                )

        # Check for non-Gaussian noise
        for non_gaussian in cls.NON_GAUSSIAN_NOISE_MODELS:
            if non_gaussian in model_name:
                return (
                    False,
                    f"Non-Gaussian noise ({non_gaussian}) not supported by PyDDM",
                )

        # Check for inter-trial variability parameters
        params = set(model_config.get("params", []))
        variability_found = params & cls.VARIABILITY_PARAMS
        if variability_found:
            return (
                False,
                f"Inter-trial variability parameters {variability_found} require numerical integration",
            )

        # Check n_choices
        n_choices = model_config.get("n_choices", 2)
        if n_choices != 2:
            return (
                False,
                f"PyDDM requires exactly 2 choices (model has {n_choices})",
            )

        # If we passed all checks, it's compatible!
        return True, "Compatible"

    @classmethod
    def create_drift_function(cls, model_config: dict[str, Any]) -> Callable:
        """Create PyDDM drift function from ssms model config.

        Returns callable with signature: drift(t, x, **theta)

        Handles:
        - Constant drift (v)
        - Position-dependent drift (Ornstein: v - g*x)
        - Custom time-dependent drift (gamma_drift, conflict models, etc.)

        Architecture:
        - Standard model configs specify drift_name and drift_fun
        - Metadata (params) is looked up from the drift registry
        - Direct drift_config dict is supported for custom runtime configurations

        This design avoids duplicating metadata across 100+ model configs.

        Args:
            model_config: Model configuration dictionary from ssms.config

        Returns:
            Callable that accepts (t, x, **theta) and returns drift rate
        """
        model_name = model_config["name"]
        drift_cfg = model_config.get("drift_config", None)

        # Standard case: look up drift metadata from registry
        if drift_cfg is None:
            drift_name = model_config.get("drift_name", "constant")
            drift_fn = model_config.get("drift_fun", None)

            # Look up parameters from drift registry
            if drift_name != "constant" and drift_fn is not None:
                from ssms.config import get_drift_registry

                drift_registry = get_drift_registry()
                if drift_registry.is_registered(drift_name):
                    # Registry lookup (standard path for all predefined models)
                    drift_info = drift_registry.get(drift_name)
                    drift_params = drift_info["params"]
                else:
                    # Fallback for custom drifts defined at runtime
                    drift_params = model_config.get("drift_params", [])

                # Assemble drift configuration
                drift_cfg = {
                    "fun": drift_fn,
                    "params": drift_params,
                }

        # If there's a custom drift function specified
        if drift_cfg is not None and drift_cfg != "constant":
            drift_fn = drift_cfg["fun"]
            drift_params = drift_cfg["params"]

            # Adapt ssms drift function (f(t, **params)) to PyDDM signature (f(t, x, **params))
            def pyddm_drift(
                t: float, x: float | np.ndarray, **theta: Any
            ) -> float | np.ndarray:
                # Extract just the drift params from theta
                drift_kwargs = {p: theta[p] for p in drift_params if p in theta}

                # ssms drift functions expect array input, return array
                # They don't depend on x (position), only on t (time)
                result = drift_fn(np.array([t]), **drift_kwargs)

                # Add base drift 'v' if it exists
                base_drift = theta.get("v", 0.0)
                total_drift = result[0] + base_drift

                # Return scalar or array depending on input x
                if np.ndim(x) == 0:
                    return float(total_drift)
                else:
                    return np.full_like(x, total_drift, dtype=np.float64)

            return pyddm_drift

        # Standard drifts (position-dependent or constant)
        if "ornstein" in model_name:
            # Position-dependent: v - g*x
            # x can be scalar or array, so this naturally works
            return lambda t, x, v, g, **kw: v - g * x
        else:
            # Constant drift: v
            # When x is an array, return array of v; when scalar, return scalar
            def constant_drift(t, x, v, **kw):
                if np.ndim(x) == 0:
                    return v
                else:
                    return np.full_like(x, v, dtype=np.float64)

            return constant_drift

    @classmethod
    def create_boundary_function(cls, model_config: dict[str, Any]) -> Callable:
        """Create PyDDM boundary function from ssms model config.

        Returns callable with signature: boundary(t, **theta)

        Architecture:
        - Standard model configs specify boundary_name and boundary function
        - Metadata (params, multiplicative) is looked up from the boundary registry
        - Direct boundary_config dict is supported for custom runtime configurations

        This design avoids duplicating metadata across 100+ model configs and ensures
        consistent behavior for all instances of a given boundary type (e.g., all
        "angle" boundaries work the same way).

        Args:
            model_config: Model configuration dictionary from ssms.config

        Returns:
            Callable that accepts (t, **theta) and returns boundary value
        """
        # Check if boundary_config dict is directly embedded (custom runtime config)
        boundary_cfg = model_config.get("boundary_config", None)

        # Standard case: look up boundary metadata from registry
        if boundary_cfg is None:
            boundary_name = model_config.get("boundary_name", "constant")
            boundary_fn = model_config.get("boundary", None)

            # Look up parameters and multiplicative flag from boundary registry
            if boundary_name != "constant" and boundary_fn is not None:
                from ssms.config import get_boundary_registry

                boundary_registry = get_boundary_registry()
                if boundary_registry.is_registered(boundary_name):
                    # Registry lookup (standard path for all predefined models)
                    boundary_info = boundary_registry.get(boundary_name)
                    boundary_params = boundary_info["params"]
                    is_multiplicative = boundary_info["multiplicative"]
                else:
                    # Fallback for custom boundaries defined at runtime
                    boundary_params = model_config.get("boundary_params", [])
                    is_multiplicative = model_config.get(
                        "boundary_multiplicative", True
                    )

                # Assemble boundary configuration
                boundary_cfg = {
                    "fun": boundary_fn,
                    "params": boundary_params,
                    "multiplicative": is_multiplicative,
                }

        # If custom boundary specified
        if boundary_cfg is not None and boundary_cfg != "constant":
            boundary_fn = boundary_cfg["fun"]
            boundary_params = boundary_cfg["params"]
            is_multiplicative = boundary_cfg.get("multiplicative", True)

            def pyddm_boundary(t: float, **theta: Any) -> float:
                # Extract boundary params
                boundary_kwargs = {p: theta[p] for p in boundary_params if p in theta}
                a = theta["a"]

                # ssms boundary functions expect array input
                result = boundary_fn(np.array([t]), **boundary_kwargs)

                if is_multiplicative:
                    # Multiplicative: boundary(t) = a * f(t)
                    return float(a * result[0])
                else:
                    # Additive: boundary(t) = a + f(t)
                    # (f(t) is typically negative for collapsing bounds)
                    return float(a + result[0])

            return pyddm_boundary

        # Constant boundary
        return lambda t, a, **kw: a

    @classmethod
    def transform_z_to_x0(
        cls, z: float, bound_at_t0: float, safety_margin: float = 0.99
    ) -> float:
        """Transform z from [0,1] to PyDDM starting position within valid bounds.

        Args:
            z: Starting position in ssms format (0=lower, 0.5=center, 1=upper)
            bound_at_t0: Boundary value at t=0 (defines valid range)
            safety_margin: Scale factor to keep x0 away from exact boundaries (default 0.99)

        Returns:
            Starting position in PyDDM format, within [-bound_at_t0, bound_at_t0]

        Note:
            PyDDM requires starting position to be strictly within the bounds at t=0.
            The safety margin ensures x0 is not at the exact boundary, avoiding
            discretization issues with PyDDM's spatial grid.
            Formula: x0 = (2z - 1) * bound(0) * safety_margin
        """
        return (2 * z - 1) * bound_at_t0 * safety_margin

    @classmethod
    def build_pyddm_model(
        cls,
        model_config: dict[str, Any],
        theta: dict[str, Any],
        generator_config: dict[str, Any] | None = None,
    ):
        """Build PyDDM model for specific parameter values.

        Args:
            model_config: Model structure (name, params, drift_config, boundary_config)
            theta: Specific parameter values for this instance
            generator_config: Simulation settings (delta_t, max_t). If None, uses
                defaults: {'delta_t': 0.001, 'max_t': 20.0}

        Returns:
            pyddm.Model instance ready to solve()

        Raises:
            ValueError: If model is not compatible with PyDDM
            ImportError: If pyddm package is not installed
        """
        try:
            import pyddm
        except ImportError:
            raise ImportError(
                "PyDDM package required for analytical PDF estimation. "
                "Install with: pip install pyddm"
            )

        # Use default generator config if none provided
        if generator_config is None:
            generator_config = {
                "delta_t": 0.001,  # 1ms time step (good balance of accuracy/speed)
                "max_t": 20.0,  # 20s maximum decision time (captures long tail)
            }

        is_compat, reason = cls.is_compatible(model_config)
        if not is_compat:
            raise ValueError(
                f"Model '{model_config['name']}' not compatible with PyDDM: {reason}"
            )

        # Create drift and boundary functions
        drift_fn = cls.create_drift_function(model_config)
        boundary_fn = cls.create_boundary_function(model_config)

        # Convert numpy arrays to scalars (PyDDM expects scalar values)
        theta_scalars = {}
        for k, v in theta.items():
            if isinstance(v, np.ndarray):
                # Extract scalar from array (assumes shape (1,) or scalar)
                theta_scalars[k] = float(v.ravel()[0])
            else:
                theta_scalars[k] = float(v)

        # Filter out 't' from theta to avoid conflict with time argument
        # ('t' in ssms = non-decision time, 't' in drift/bound = time variable)
        theta_for_functions = {k: v for k, v in theta_scalars.items() if k != "t"}

        # Transform starting position from ssms [0, 1] to PyDDM [-1, 1]
        # In ssms: z=0 (lower), z=0.5 (center), z=1 (upper)
        # In PyDDM: starting_position=-1 (lower), 0 (center), 1 (upper)
        z = theta_scalars.get("z", 0.5)
        pyddm_starting_position = 2 * z - 1

        # Extract non-decision time
        t_ndt = theta_scalars.get("t", 0.0)

        # Build PyDDM model
        # The lambda wrappers close over theta, making parameters available
        # when PyDDM calls drift(t, x) and bound(t) during solve()

        model = pyddm.gddm(
            drift=lambda t, x: drift_fn(t, x, **theta_for_functions),
            bound=lambda t: boundary_fn(t, **theta_for_functions),
            starting_position=pyddm_starting_position,
            nondecision=t_ndt,
            mixture_coef=0.0,  # For training data computations we don't use mixture model for now
            dt=generator_config.get("delta_t", 0.001),
            T_dur=generator_config.get("max_t", 20.0),
        )

        return model
