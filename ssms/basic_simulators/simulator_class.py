"""
Class-based interface for Sequential Sampling Model simulations.

This module provides a modern, object-oriented interface for SSM simulations
that supports custom boundary functions, drift functions, and even fully custom
simulator implementations.
"""

import inspect
import warnings
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pandas as pd

from ssms.basic_simulators.simulator import (
    _get_unique_seed,
    _preprocess_theta_deadline,
    _preprocess_theta_generic,
    make_boundary_dict,
    make_drift_dict,
    make_noise_vec,
    validate_ssm_parameters,
)
from ssms.basic_simulators.modular_parameter_simulator_adapter import (
    ModularParameterSimulatorAdapter,
    ParameterSimulatorAdapterProtocol,
)
from ssms.basic_simulators.parameter_adapters import ParameterAdaptation
from ssms.config import get_boundary_registry, get_drift_registry


class Simulator:
    """Class-based interface for Sequential Sampling Model simulations.

    This class provides a flexible, extensible way to configure and run SSM simulations.
    It supports:
    - Pre-defined models (via string names like "ddm", "angle", etc.)
    - Custom boundary and drift functions with existing simulators
    - Fully custom simulator functions
    - Configuration overrides for any model parameter

    Examples
    --------
    Basic usage with pre-defined model:

    >>> sim = Simulator("ddm")
    >>> results = sim.simulate(theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3})

    Custom boundary function:

    >>> def my_boundary(t, theta, scale):
    ...     return scale * np.sin(theta * t)
    >>> sim = Simulator("ddm", boundary=my_boundary, boundary_params=["theta", "scale"])
    >>> results = sim.simulate(theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3,
    ...                               'theta': 0.5, 'scale': 1.0})

    Fully custom simulator:

    >>> def my_sim(v, a, z, t, max_t=20, n_samples=1000, **kwargs):
    ...     rts = np.random.exponential(1/abs(v), n_samples) + t
    ...     choices = np.where(np.random.random(n_samples) < z, 1, -1)
    ...     return {'rts': rts, 'choices': choices,
    ...             'metadata': {'model': 'custom', 'n_samples': n_samples}}
    >>> sim = Simulator(simulator_function=my_sim, params=["v", "a", "z", "t"],
    ...                 nchoices=2)
    >>> results = sim.simulate(theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3})

    Attributes
    ----------
    config : dict
        The full model configuration dictionary
    """

    def __init__(
        self,
        model: str | dict | None = None,
        boundary: str | Callable | None = None,
        drift: str | Callable | None = None,
        simulator_function: Callable | None = None,
        parameter_adapter: ParameterSimulatorAdapterProtocol | None = None,
        parameter_adaptations: list[ParameterAdaptation] | None = None,
        **config_overrides,
    ):
        """Initialize a Simulator instance.

        Parameters
        ----------
        model : str, dict, or None
            Either a model name (e.g., "ddm", "angle"), a full configuration dictionary,
            or None if providing a custom simulator_function.
        boundary : str, Callable, or None
            Boundary function. Can be:
            - A string name from the boundary registry (e.g., "angle", "weibull_cdf")
            - A callable function with signature: func(t, **params) -> np.ndarray
            - None to use the model's default boundary
        drift : str, Callable, or None
            Drift function. Can be:
            - A string name from the drift registry (e.g., "constant", "gamma_drift")
            - A callable function with signature: func(t, **params) -> np.ndarray
            - None to use the model's default drift (if applicable)
        simulator_function : Callable or None
            Custom simulator function. If provided, this overrides the model's
            default simulator. Must follow the simulator interface contract.
        parameter_adapter : ParameterSimulatorAdapterProtocol or None
            Custom parameter adapter for preparing parameters for simulators. If None, uses
            ModularParameterSimulatorAdapter with default adaptations for the model.
        parameter_adaptations : list[ParameterAdaptation] or None
            Additional parameter adaptations to apply AFTER the model's default
            adaptations. Useful for adding custom parameter preparation without
            replacing the entire adapter. Only used if parameter_adapter is None.
        **config_overrides
            Additional configuration parameters to override. Common options:
            - params : list[str] - Parameter names (required if simulator_function provided)
            - param_bounds : list - [[lower bounds], [upper bounds]]
            - nchoices : int - Number of choices (required if simulator_function provided)
            - choices : list - Possible choice values
            - boundary_params : list[str] - Parameters for custom boundary function
            - drift_params : list[str] - Parameters for custom drift function
            - name : str - Model name (for custom simulators)

        Raises
        ------
        ValueError
            If configuration is invalid or missing required parameters

        Examples
        --------
        Use default modular adapter:

        >>> sim = Simulator("lba2")
        >>> results = sim.simulate(theta={'v0': 0.5, 'v1': 0.6, 'A': 0.5, 'b': 1.0})

        Add custom adaptations:

        >>> from ssms.basic_simulators.parameter_adapters import SetDefaultValue
        >>> sim = Simulator("ddm", parameter_adaptations=[SetDefaultValue("custom_param", 42)])

        Use a custom parameter adapter:

        >>> custom_adapter = ModularParameterSimulatorAdapter()
        >>> sim = Simulator("ddm", parameter_adapter=custom_adapter)
        """
        # Set parameter adapter (default to modular)
        self._parameter_adapter = (
            parameter_adapter or ModularParameterSimulatorAdapter()
        )
        self._custom_adaptations = parameter_adaptations or []

        self._config = self._build_config(
            model, boundary, drift, simulator_function, config_overrides
        )

    def _build_config(
        self,
        model: str | dict | None,
        boundary: str | Callable | None,
        drift: str | Callable | None,
        simulator_function: Callable | None,
        config_overrides: dict,
    ) -> dict:
        """Build the configuration dictionary from inputs.

        Parameters
        ----------
        model : str, dict, or None
            Model specification
        boundary : str, Callable, or None
            Boundary function specification
        drift : str, Callable, or None
            Drift function specification
        simulator_function : Callable or None
            Custom simulator function
        config_overrides : dict
            Configuration overrides

        Returns
        -------
        dict
            Complete configuration dictionary

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Case 1: Full config dict provided
        if isinstance(model, dict):
            config = deepcopy(model)
        # Case 2: Model name provided (including variants like "_deadline")
        elif isinstance(model, str):
            # Use ModelConfigBuilder.from_model() which handles variant suffixes
            from ssms.config import ModelConfigBuilder

            config = ModelConfigBuilder.from_model(model)
        # Case 3: Custom simulator without base model
        elif model is None and simulator_function is not None:
            config = self._create_minimal_config(simulator_function, config_overrides)
        else:
            raise ValueError(
                "Must provide either a model name, config dict, or simulator_function"
            )

        # Apply config overrides
        config.update(config_overrides)

        # Handle custom simulator function
        if simulator_function is not None:
            self._validate_simulator_function(simulator_function, config)
            config["simulator"] = simulator_function

        # Handle custom boundary
        if boundary is not None:
            self._apply_custom_boundary(config, boundary)

        # Handle custom drift
        if drift is not None:
            self._apply_custom_drift(config, drift)

        # Validate final configuration
        self._validate_config(config)

        return config

    def _create_minimal_config(
        self, simulator_function: Callable, overrides: dict
    ) -> dict:
        """Create minimal config for custom simulator.

        Parameters
        ----------
        simulator_function : Callable
            The custom simulator function
        overrides : dict
            User-provided configuration parameters

        Returns
        -------
        dict
            Minimal valid configuration

        Raises
        ------
        ValueError
            If required parameters are missing
        """
        # Check for required parameters
        if "params" not in overrides:
            raise ValueError(
                "When using a custom simulator_function without a base model, "
                "you must provide 'params' (list of parameter names)"
            )
        if "nchoices" not in overrides:
            raise ValueError(
                "When using a custom simulator_function without a base model, "
                "you must provide 'nchoices' (number of choices)"
            )

        params = overrides["params"]
        nchoices = overrides["nchoices"]

        # Build minimal config
        config = {
            "name": overrides.get("name", "custom"),
            "params": params,
            "n_params": len(params),
            "nchoices": nchoices,
            "choices": overrides.get("choices", list(range(nchoices))),
            "n_particles": overrides.get("n_particles", 1),
            "simulator": simulator_function,
        }

        # Add optional fields if provided
        if "param_bounds" in overrides:
            config["param_bounds"] = overrides["param_bounds"]
        if "default_params" in overrides:
            config["default_params"] = overrides["default_params"]

        return config

    def _validate_simulator_function(self, func: Callable, config: dict) -> None:
        """Validate that simulator function has correct signature.

        Parameters
        ----------
        func : Callable
            The simulator function to validate
        config : dict
            Configuration dictionary containing parameter names

        Raises
        ------
        ValueError
            If simulator function signature is invalid
        """
        sig = inspect.signature(func)
        params = sig.parameters

        # Check for required simulation parameters
        required_sim_params = ["max_t", "n_samples", "delta_t", "random_state"]

        for param in required_sim_params:
            if param not in params:
                warnings.warn(
                    f"Simulator function '{func.__name__}' is missing recommended "
                    f"parameter '{param}'. This may cause issues during simulation.",
                    UserWarning,
                )

        # Check that model parameters are in signature
        model_params = config.get("params", [])
        for param in model_params:
            if param not in params:
                raise ValueError(
                    f"Simulator function '{func.__name__}' is missing required "
                    f"model parameter '{param}'"
                )

    def _infer_simulator_requirements(self, func: Callable) -> dict:
        """Inspect simulator function to determine its requirements.

        Parameters
        ----------
        func : Callable
            The simulator function to inspect

        Returns
        -------
        dict
            Dictionary with inferred requirements:
            - required_params : list of required parameter names
            - optional_params : list of optional parameter names
            - supports_boundary : whether it accepts boundary_fun
            - supports_drift : whether it accepts drift_fun
        """
        sig = inspect.signature(func)
        params = sig.parameters

        required = []
        optional = []

        for name, param in params.items():
            if param.default == inspect.Parameter.empty and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                required.append(name)
            else:
                optional.append(name)

        return {
            "required_params": required,
            "optional_params": optional,
            "supports_boundary": "boundary_fun" in params,
            "supports_drift": "drift_fun" in params,
        }

    def _apply_custom_boundary(self, config: dict, boundary: str | Callable) -> None:
        """Apply custom boundary to configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to modify
        boundary : str or Callable
            Boundary specification

        Raises
        ------
        ValueError
            If boundary specification is invalid
        """
        if isinstance(boundary, str):
            # Look up boundary by name
            boundary_registry = get_boundary_registry()
            if not boundary_registry.is_registered(boundary):
                raise ValueError(
                    f"Unknown boundary '{boundary}'. Available boundaries: "
                    f"{boundary_registry.list_boundaries()}"
                )
            boundary_info = boundary_registry.get(boundary)
            config["boundary_name"] = boundary
            config["boundary"] = boundary_info["fun"]
            if "boundary_params" not in config:
                config["boundary_params"] = boundary_info["params"]
        elif callable(boundary):
            # Custom boundary function
            self._validate_boundary_function(boundary)
            config["boundary"] = boundary
            config["boundary_name"] = getattr(boundary, "__name__", "custom")
            # boundary_params should be provided in config_overrides
            if "boundary_params" not in config:
                warnings.warn(
                    "Custom boundary function provided without 'boundary_params'. "
                    "You may need to specify boundary parameter names.",
                    UserWarning,
                )
                config["boundary_params"] = []
        else:
            raise ValueError("boundary must be a string name or callable function")

    def _apply_custom_drift(self, config: dict, drift: str | Callable) -> None:
        """Apply custom drift to configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to modify
        drift : str or Callable
            Drift specification

        Raises
        ------
        ValueError
            If drift specification is invalid
        """
        if isinstance(drift, str):
            # Look up drift by name
            drift_registry = get_drift_registry()
            if not drift_registry.is_registered(drift):
                raise ValueError(
                    f"Unknown drift '{drift}'. Available drifts: "
                    f"{drift_registry.list_drifts()}"
                )
            drift_info = drift_registry.get(drift)
            config["drift_name"] = drift
            config["drift"] = drift_info["fun"]
            if "drift_params" not in config:
                config["drift_params"] = drift_info["params"]
        elif callable(drift):
            # Custom drift function
            self._validate_drift_function(drift)
            config["drift"] = drift
            config["drift_name"] = getattr(drift, "__name__", "custom")
            # drift_params should be provided in config_overrides
            if "drift_params" not in config:
                warnings.warn(
                    "Custom drift function provided without 'drift_params'. "
                    "You may need to specify drift parameter names.",
                    UserWarning,
                )
                config["drift_params"] = []
        else:
            raise ValueError("drift must be a string name or callable function")

    def _validate_boundary_function(self, func: Callable) -> None:
        """Validate boundary function signature.

        Parameters
        ----------
        func : Callable
            Boundary function to validate

        Raises
        ------
        ValueError
            If function signature is invalid
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if not params or params[0] != "t":
            raise ValueError(
                f"Boundary function '{func.__name__}' must accept 't' as "
                "first positional argument"
            )

    def _validate_drift_function(self, func: Callable) -> None:
        """Validate drift function signature.

        Parameters
        ----------
        func : Callable
            Drift function to validate

        Raises
        ------
        ValueError
            If function signature is invalid
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if not params or params[0] != "t":
            raise ValueError(
                f"Drift function '{func.__name__}' must accept 't' as "
                "first positional argument"
            )

    def _validate_config(self, config: dict) -> None:
        """Validate configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration to validate

        Raises
        ------
        ValueError
            If configuration is missing required fields
        """
        required_fields = ["params", "nchoices", "simulator"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Configuration missing required field: '{field}'")

    def simulate(
        self,
        theta: list | np.ndarray | dict | pd.DataFrame,
        n_samples: int = 1000,
        delta_t: float = 0.001,
        max_t: float = 20,
        no_noise: bool = False,
        sigma_noise: float | None = None,
        smooth_unif: bool = True,
        random_state: int | None = None,
        return_option: str = "full",
        n_threads: int = 1,
    ) -> dict:
        """Run simulation with given parameters.

        Parameters
        ----------
        theta : list, np.ndarray, dict, or pd.DataFrame
            Model parameters. If dict or DataFrame, keys/columns should match
            parameter names in config. If array, order should match config['params'].
        n_samples : int, default=1000
            Number of simulation samples per parameter set
        delta_t : float, default=0.001
            Time step size for simulation
        max_t : float, default=20
            Maximum simulation time
        no_noise : bool, default=False
            If True, disable noise in simulation
        sigma_noise : float or None
            Standard deviation of noise. If None, uses model defaults.
        smooth_unif : bool, default=True
            Whether to add uniform smoothing to RTs
        random_state : int or None
            Random seed for reproducibility
        return_option : str, default="full"
            Output format: "full" or "minimal"
        n_threads : int, default=1
            Number of threads for parallel execution. If > 1 and OpenMP is available,
            uses multi-threaded simulation. Note: trajectory recording is only
            available with n_threads=1.

        Returns
        -------
        dict
            Simulation results with keys:
            - 'rts' : np.ndarray of reaction times
            - 'choices' : np.ndarray of choices
            - 'metadata' : dict with simulation metadata
            - Additional keys depending on model and return_option

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # Get model config - deadline params are already included if present
        # (handled by ModelConfigBuilder.from_model() during initialization)
        model_config_local = deepcopy(self._config)
        model_name = model_config_local.get("name", "").replace("_deadline", "")

        # Check if this is a deadline model by looking at params
        is_deadline_model = "deadline" in model_config_local.get("params", [])

        if random_state is None:
            random_state = _get_unique_seed()

        # Preprocess theta
        theta = _preprocess_theta_generic(theta)
        n_trials, theta = _preprocess_theta_deadline(
            theta, is_deadline_model, model_config_local
        )

        # Build simulation parameters dict
        sim_param_dict = {
            "max_t": max_t,
            "n_samples": n_samples,
            "n_trials": n_trials,
            "delta_t": delta_t,
            "random_state": random_state,
            "return_option": return_option,
            "smooth_unif": smooth_unif,
            "n_threads": n_threads,
        }

        # Handle noise
        if "sd" in theta or "s" in theta:
            if sigma_noise is not None:
                raise ValueError(
                    "sigma_noise parameter should be None if 'sd' or 's' is "
                    "passed via theta dictionary"
                )
            elif no_noise:
                sigma_noise = 0.0
            elif "sd" in theta:
                sigma_noise = theta["sd"]
            elif "s" in theta:
                sigma_noise = theta["s"]
        else:
            if no_noise:
                sigma_noise = 0.0
            elif "lba" in model_name and sigma_noise is None:
                sigma_noise = 0.1
            elif sigma_noise is None:
                sigma_noise = 1.0

        noise_vec = make_noise_vec(
            sigma_noise, n_trials, model_config_local["n_particles"]
        )

        if "lba" in model_name:
            theta["sd"] = noise_vec
        else:
            theta["s"] = noise_vec

        # Adapt parameters with configured adapter
        theta = self._parameter_adapter.adapt_parameters(
            theta, model_config_local, n_trials
        )

        # Apply custom adaptations (if using ModularParameterSimulatorAdapter and custom adaptations provided)
        if self._custom_adaptations and isinstance(
            self._parameter_adapter, ModularParameterSimulatorAdapter
        ):
            for adaptation in self._custom_adaptations:
                theta = adaptation.apply(theta, model_config_local, n_trials)

        # Make boundary and drift dictionaries (if applicable)
        boundary_dict = {}
        drift_dict = {}
        if "boundary_name" in model_config_local or "boundary" in model_config_local:
            boundary_dict = make_boundary_dict(model_config_local, theta)
        if "drift_name" in model_config_local or "drift" in model_config_local:
            drift_dict = make_drift_dict(model_config_local, theta)

        # Validate parameters
        validate_ssm_parameters(model_name, theta)

        # Call simulator
        x = model_config_local["simulator"](
            **theta,
            **boundary_dict,
            **drift_dict,
            **sim_param_dict,
        )

        if not isinstance(x, dict):
            raise TypeError(f"Simulator returned {type(x).__name__}, expected dict")

        # Postprocess results - squeeze dimensions for compatibility
        # (This matches the behavior of the legacy simulator() function)
        if n_trials == 1:
            x["rts"] = np.squeeze(x["rts"], axis=1)
            x["choices"] = np.squeeze(x["choices"], axis=1)
        if n_trials > 1 and n_samples == 1:
            x["rts"] = np.squeeze(x["rts"], axis=0)
            x["choices"] = np.squeeze(x["choices"], axis=0)

        x["metadata"]["model"] = model_name

        return x

    def validate_params(self, theta: dict) -> None:
        """Validate parameter values.

        Parameters
        ----------
        theta : dict
            Parameter dictionary to validate

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        model_name = self._config.get("name", "")
        validate_ssm_parameters(model_name, theta)

    @property
    def config(self) -> dict:
        """Get the full model configuration.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return deepcopy(self._config)

    @property
    def parameter_adapter(self) -> ParameterSimulatorAdapterProtocol:
        """Get the configured parameter adapter.

        Returns
        -------
        ParameterSimulatorAdapterProtocol
            The parameter adapter instance used for parameter preparation
        """
        return self._parameter_adapter
