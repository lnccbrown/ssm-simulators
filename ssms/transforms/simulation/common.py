"""Common simulation-time parameter transforms.

This module provides a library of commonly-used parameter transforms that
prepare theta parameters for simulator consumption.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ssms.transforms.base import ParameterTransform


class SetDefaultValue(ParameterTransform):
    """Set a parameter to a default value if not present.

    This transform adds a parameter with a default value if it doesn't
    already exist in theta. The value is automatically tiled to match n_trials.

    Parameters
    ----------
    param_name : str
        Name of the parameter to set
    default_value : float or int or np.ndarray
        Default value to use. If scalar, will be tiled to (n_trials,).
    dtype : np.dtype, optional
        Data type for the parameter. Defaults to np.float32.

    Examples
    --------
    >>> transform = SetDefaultValue("nact", 3)
    >>> theta = {}
    >>> theta = transform.apply(theta, {}, n_trials=5)
    >>> theta["nact"]  # array([3., 3., 3., 3., 3.], dtype=float32)
    """

    def __init__(
        self,
        param_name: str,
        default_value: float | int | np.ndarray,
        dtype: np.dtype = np.float32,
    ):
        self.param_name = param_name
        self.default_value = default_value
        self.dtype = dtype

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Add parameter with default value if not present."""
        if self.param_name not in theta:
            if n_trials is None:
                n_trials = 1
            if isinstance(self.default_value, (int, float)):
                theta[self.param_name] = np.tile(
                    np.array([self.default_value], dtype=self.dtype), n_trials
                )
            else:
                theta[self.param_name] = np.asarray(
                    self.default_value, dtype=self.dtype
                )
        return theta


class ExpandDimension(ParameterTransform):
    """Expand dimensions of specified parameters.

    Converts 1D arrays of shape (n_trials,) to shape (n_trials, 1) by adding
    a dimension. This is commonly needed for multi-particle models.

    Parameters
    ----------
    param_names : list[str]
        Names of parameters to expand

    Examples
    --------
    >>> transform = ExpandDimension(["a", "t"])
    >>> theta = {"a": np.array([1.0, 2.0]), "t": np.array([0.3, 0.3])}
    >>> theta = transform.apply(theta)
    >>> theta["a"].shape  # (2, 1)
    """

    def __init__(self, param_names: list[str]):
        self.param_names = param_names

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Expand dimensions of specified parameters."""
        for param in self.param_names:
            if param in theta:
                theta[param] = np.expand_dims(theta[param], axis=1)
        return theta


class ColumnStackParameters(ParameterTransform):
    """Stack multiple parameters into a single multi-column array.

    Takes individual parameters (e.g., v0, v1, v2) and stacks them column-wise
    into a single array (e.g., v with shape (n_trials, 3)).

    Parameters
    ----------
    source_params : list[str]
        Names of parameters to stack
    target_param : str
        Name of the resulting stacked parameter
    delete_sources : bool, default=True
        Whether to delete the source parameters after stacking

    Examples
    --------
    >>> transform = ColumnStackParameters(["v0", "v1", "v2"], "v")
    >>> theta = {"v0": [0.5], "v1": [0.6], "v2": [0.7]}
    >>> theta = transform.apply(theta)
    >>> theta["v"]  # array([[0.5, 0.6, 0.7]])
    """

    def __init__(
        self, source_params: list[str], target_param: str, delete_sources: bool = True
    ):
        self.source_params = source_params
        self.target_param = target_param
        self.delete_sources = delete_sources

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Stack parameters column-wise."""
        if all(param in theta for param in self.source_params):
            theta[self.target_param] = np.column_stack(
                [theta[param] for param in self.source_params]
            )
            if self.delete_sources:
                for param in self.source_params:
                    del theta[param]
        return theta


class RenameParameter(ParameterTransform):
    """Rename a parameter and optionally transform its value.

    Parameters
    ----------
    old_name : str
        Current parameter name
    new_name : str
        New parameter name
    transform_fn : Callable or None, optional
        Optional function to apply to the value during renaming

    Examples
    --------
    >>> transform = RenameParameter("A", "z", lambda x: np.expand_dims(x, axis=1))
    >>> theta = {"A": np.array([0.5])}
    >>> theta = transform.apply(theta)
    >>> "z" in theta  # True
    """

    def __init__(
        self,
        old_name: str,
        new_name: str,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.old_name = old_name
        self.new_name = new_name
        self.transform_fn = transform_fn

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Rename parameter and optionally transform its value."""
        if self.old_name in theta:
            value = theta.pop(self.old_name)
            if self.transform_fn is not None:
                value = self.transform_fn(value)
            theta[self.new_name] = value
        return theta


class DeleteParameters(ParameterTransform):
    """Delete specified parameters from theta.

    Parameters
    ----------
    param_names : list[str]
        Names of parameters to delete

    Examples
    --------
    >>> transform = DeleteParameters(["A", "b"])
    >>> theta = {"v": [0.5], "A": [0.5], "b": [1.0]}
    >>> theta = transform.apply(theta)
    >>> "A" in theta  # False
    """

    def __init__(self, param_names: list[str]):
        self.param_names = param_names

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Delete specified parameters."""
        for param in self.param_names:
            theta.pop(param, None)
        return theta


class SetZeroArray(ParameterTransform):
    """Set a parameter to an array of zeros.

    Parameters
    ----------
    param_name : str
        Name of the parameter to set
    shape : tuple or None, optional
        Shape of the array. Use None for n_trials dimension.
    dtype : np.dtype, optional
        Data type for the array. Defaults to np.float32.

    Examples
    --------
    >>> transform = SetZeroArray("v")
    >>> theta = transform.apply({}, {}, n_trials=5)
    >>> theta["v"].shape  # (5,)
    """

    def __init__(
        self, param_name: str, shape: tuple | None = None, dtype: np.dtype = np.float32
    ):
        self.param_name = param_name
        self.shape = shape
        self.dtype = dtype

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Set parameter to zero array."""
        if n_trials is None:
            n_trials = 1
        if self.shape is None:
            theta[self.param_name] = np.zeros(n_trials, dtype=self.dtype)
        else:
            actual_shape = tuple(n_trials if dim is None else dim for dim in self.shape)
            theta[self.param_name] = np.zeros(actual_shape, dtype=self.dtype)
        return theta


class TileArray(ParameterTransform):
    """Tile a constant value across trials.

    Parameters
    ----------
    param_name : str
        Name of the parameter to create
    value : float or array-like
        Value(s) to tile
    dtype : np.dtype, optional
        Data type for the array. Defaults to np.float32.

    Examples
    --------
    >>> transform = TileArray("z", 0.5)
    >>> theta = transform.apply({}, {}, n_trials=3)
    >>> theta["z"]  # array([0.5, 0.5, 0.5], dtype=float32)
    """

    def __init__(
        self,
        param_name: str,
        value: float | list | np.ndarray,
        dtype: np.dtype = np.float32,
    ):
        self.param_name = param_name
        self.value = value
        self.dtype = dtype

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Tile value across trials."""
        if n_trials is None:
            n_trials = 1
        if isinstance(self.value, (int, float)):
            theta[self.param_name] = np.tile(
                np.array([self.value], dtype=self.dtype), n_trials
            )
        else:
            value_array = np.array(self.value, dtype=self.dtype)
            if value_array.ndim == 0:
                theta[self.param_name] = np.tile(value_array, n_trials)
            else:
                theta[self.param_name] = np.tile(value_array, (n_trials, 1))
        return theta


class ApplyMapping(ParameterTransform):
    """Apply a mapping function to transform a parameter.

    Uses a mapping from model_config to transform a parameter value.
    Commonly used for random variable distributions.

    Parameters
    ----------
    source_param : str
        Parameter to read from
    target_param : str
        Parameter to write to
    mapping_key : str
        Key in model_config["simulator_param_mappings"]
    additional_sources : list[str], optional
        Additional parameters to pass to the mapping function
    delete_source : bool, default=False
        Whether to delete the source parameter after mapping
    """

    def __init__(
        self,
        source_param: str,
        target_param: str,
        mapping_key: str,
        additional_sources: list[str] | None = None,
        delete_source: bool = False,
    ):
        self.source_param = source_param
        self.target_param = target_param
        self.mapping_key = mapping_key
        self.additional_sources = additional_sources or []
        self.delete_source = delete_source

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Apply mapping function to transform parameter."""
        if self.source_param in theta and model_config is not None:
            mappings = model_config.get("simulator_param_mappings", {})
            if self.mapping_key in mappings:
                mapping_fn = mappings[self.mapping_key]
                args = [theta[self.source_param]]
                for param in self.additional_sources:
                    if param in theta:
                        args.append(theta[param])
                theta[self.target_param] = mapping_fn(*args)
                if self.delete_source:
                    del theta[self.source_param]
        return theta


class ConditionalAdaptation(ParameterTransform):
    """Apply a transform only if a condition is met.

    Parameters
    ----------
    condition : Callable
        Function that takes (theta, model_config, n_trials) and returns bool
    transformation : ParameterTransform
        Transform to apply if condition is True

    Examples
    --------
    >>> transform = ConditionalAdaptation(
    ...     condition=lambda theta, cfg, n: "a" in theta,
    ...     transformation=ExpandDimension(["a"])
    ... )
    """

    def __init__(
        self,
        condition: Callable[[dict, dict | None, int | None], bool],
        transformation: ParameterTransform,
    ):
        self.condition = condition
        self.transformation = transformation

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Apply transform if condition is met."""
        if self.condition(theta, model_config, n_trials):
            theta = self.transformation.apply(theta, model_config, n_trials)
        return theta

    def __repr__(self) -> str:
        return f"ConditionalAdaptation({self.transformation})"


class LambdaAdaptation(ParameterTransform):
    """Wrap a lambda or callable function as a transform.

    Parameters
    ----------
    func : Callable
        Function with signature: func(theta, model_config, n_trials) → theta
    name : str, optional
        Optional name for debugging

    Examples
    --------
    >>> transform = LambdaAdaptation(
    ...     lambda theta, cfg, n: theta.update({"nact": 3}) or theta,
    ...     name="set_nact_to_3"
    ... )
    """

    def __init__(
        self,
        func: Callable[[dict, dict | None, int | None], dict],
        name: str | None = None,
    ):
        self.func = func
        self.name = name or "lambda"

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Apply the wrapped function."""
        return self.func(theta, model_config, n_trials)

    def __repr__(self) -> str:
        return f"LambdaAdaptation(name='{self.name}')"
