from functools import partial
from typing import Any, Callable, Protocol, cast
import logging

import numpy as np

from .basic_simulators.simulator import simulator
from .config import model_config as ssms_model_config

_logger = logging.getLogger(__name__)


class _HasListParams(Protocol):  # for mypy
    _list_params: list[str]


def _create_arg_arrays(cls: _HasListParams, args: tuple) -> list[np.ndarray]:
    """
    Create argument arrays from input arguments.

    Parameters
    ----------
    cls : type
        The class containing `_list_params`.
    args : tuple
        Input arguments.

    Returns
    -------
    list of np.ndarray
        List of argument arrays.
    """
    num_params = len(cls._list_params)
    n_args = min(num_params, len(args))
    arg_arrays = [np.asarray(arg) for arg in args[:n_args]]
    return arg_arrays


def _extract_size_val(size: tuple | int) -> int:
    """Extract integer value from size, handling tuple or scalar."""
    if isinstance(size, tuple):
        return size[0]
    return size


def _calculate_n_replicas(is_all_args_scalar, size, new_data_size):
    """
    Calculate the number of replicas (samples) to draw from each trial based on input arguments.

    Parameters
    ----------
    is_all_args_scalar : bool
        Indicates whether all input arguments are scalars.
    size : int or None
        The total number of samples to be drawn. If None or 1, only one replica is
        drawn.
    new_data_size : int
        The size of the new data to be used for sampling.

    Returns
    -------
    int
        The number of replicas to draw for each trial.

    Raises
    ------
    ValueError
        If `size` is not compatible with `new_data_size` as determined by
        `_validate_size`.
    """  # noqa: E501
    # The multiple then becomes how many samples we draw from each trial.
    if any([is_all_args_scalar, size is None, size == 1]):
        return 1
    size_val = _extract_size_val(size)
    _validate_size(size_val, new_data_size)
    return size_val // new_data_size


def _extract_size(args, kwargs):
    """Extract size from args and kwargs.

    Returns
    -------
    size : int
        The size of the random sample to generate.
    args : tuple
        The original arguments, with size removed if it was present.
    kwargs : dict
        The original keyword arguments, with size removed if it was present.
    """
    if "size" in kwargs:
        size = kwargs.pop("size")
    else:
        size = args[-1]
        args = args[:-1]

    if size is None:
        size = 1

    return size, args, kwargs


def _get_seed(rng):
    """Get a seed for the random number generator."""
    iinfo32 = np.iinfo(np.uint32)
    return rng.integers(0, iinfo32.max, dtype=np.uint32)


def _prepare_theta_and_shape(arg_arrays, size):
    """
    Prepare the parameter matrix `theta` for simulation.

    If all parameters passed are scalar, assemble all parameters into a 1D array
    and pass it to the `theta` argument. In this case, size is number of observations.
    If any parameter is a vector, preprocess all parameters, reshape them into a matrix
    of dimension (size, n_params) where size is the number of elements in the largest
    of all parameters passed to *arg.
    """
    is_all_args_scalar = all(arg.size == 1 for arg in arg_arrays)
    if is_all_args_scalar:
        # If all parameters passed are scalar, assemble all parameters into a 1D array
        # and pass it to the `theta` argument. In this case, size is the number of
        # observations.
        theta = np.stack(arg_arrays)
        if theta.ndim > 1:
            theta = theta.squeeze(axis=-1)

        if isinstance(size, tuple) and len(size) == 1:
            size_ = size[0]
        elif isinstance(size, int):
            size_ = size
        else:
            raise ValueError(
                f"Size must be a tuple of length 1 or an integer, but got {type(size)}"
            )
        theta = np.tile(theta, (size_, 1))
        return True, theta, None, None

    # Preprocess all parameters, reshape them into a matrix of dimension
    # (size, n_params) where size is the number of elements in the
    # largest of all parameters passed to *arg
    largest_param_idx = np.argmax([arg.size for arg in arg_arrays])
    max_shape = arg_arrays[largest_param_idx].shape
    new_data_size = max_shape[-1]
    theta = np.column_stack(
        [np.broadcast_to(arg, max_shape).reshape(-1) for arg in arg_arrays]
    )
    return False, theta, max_shape, new_data_size


def _reshape_sims_out(max_shape, n_replicas, obs_dim_int):
    """Calculate the output shape for simulation results.

    Parameters
    ----------
    max_shape : tuple or list
        The maximum shape of the input parameters.
    n_replicas : int
        Number of replicas (samples) to draw for each trial.
    obs_dim_int : int
        The number of observation dimensions.

    Returns
    -------
    tuple
        The shape of the simulation output.
    """
    shape = [*max_shape[:-1], max_shape[-1]]
    if n_replicas != 1:
        shape.append(n_replicas)
    shape.append(obs_dim_int)
    return tuple(shape)


def _validate_size(size_val: int, new_data_size: int) -> None:
    """Validate that `size` is a multiple of `new_data_size`.

    Parameters
    ----------
    size_val : int
        The total number of samples to be drawn.
    new_data_size : int
        The size of the new data to be used for sampling.

    Raises
    ------
    ValueError
        If `size_val` is not a multiple of `new_data_size`.
    """
    # If size is not None, we check if size is a multiple of the largest size.
    # If not, an error is thrown.
    if size_val % new_data_size != 0:
        raise ValueError("`size` needs to be a multiple of the size of data")


def _get_p_outlier(cls: _HasListParams, arg_arrays):
    """Get p_outlier from arg_arrays and update arg_arrays."""
    list_params = cls._list_params
    p_outlier = None
    if list_params and list_params[-1] == "p_outlier":
        p_outlier = arg_arrays.pop(-1)
    return p_outlier, arg_arrays


def _validate_simulator_fun_arg(simulator_fun: str | Callable) -> None:
    """
    Validate the simulator function argument.

    Parameters
    ----------
    simulator_fun : Callable or str
        The simulator function or the name of the model as a string. If a string, we assume
        it is a valid model in the ssm-simulators package.

    Raises
    ------
    ValueError
        If the simulator argument is not a string or a callable.
    """
    if not (isinstance(simulator_fun, str) or callable(simulator_fun)):
        raise ValueError(
            "`simulator_fun` must be a string or a callable, "
            f"but you passed {type(simulator_fun)}."
        )


def decorate_atomic_simulator(
    model_name: str,
    choices: list | np.ndarray | None = None,
    obs_dim: int = 2,  # At least for now ssms models all fall under 2 obs dims
):
    """
    Decorator to add metadata attributes to simulator functions.

    This decorator attaches the following attributes to the decorated function as expected of simulators in HSSM:
    - model_name: Name of the model.
    - choices: List or array of possible choices/responses.
    - obs_dim: Number of observation dimensions.

    Parameters
    ----------
    model_name : str
        Name of the model.
    choices : list or np.ndarray, optional
        List or array of possible choices/responses (default: [-1, 1]).
    obs_dim : int, optional
        Number of observation dimensions (default: 2).

    Returns
    -------
    Callable
        Decorator that adds attributes to the simulator function.
    """

    choices = [-1, 1] if choices is None else choices

    def decorator(func):
        func.model_name = model_name
        func.choices = choices
        func.obs_dim = obs_dim
        return func

    return decorator


def hssm_sim_wrapper(simulator_fun, theta, model, n_replicas, random_state, **kwargs):
    """Wrap a ssms simulator function to match HSSM's expected interface.

    Parameters
    ----------
    simulator_fun : callable
        The simulator function to wrap, which should have the following interface:
        - theta: array-like, shape (n_trials, n_parameters)
        - model: str, name of the model to simulate
        - n_samples: int, number of replica datasets to generate
        - random_state: int, to be used as the random seed internally
        - **kwargs: additional keyword arguments
    theta : array-like
        Model parameters, shape (n_trials, n_parameters)
    model : str
        Name of the model to simulate
    n_replicas : int
        Number of replica datasets to generate
    random_state : int or numpy.random.Generator
        Random seed or random number generator
    **kwargs
        Additional keyword arguments passed to simulator_fun

    Returns
    -------
    array-like
        Array of shape (n_trials, 2) containing reaction times and choices
        stacked column-wise
    """
    out = simulator_fun(
        theta=theta,
        model=model,
        n_samples=n_replicas,
        random_state=random_state,
        **kwargs,
    )
    return np.stack([out["rts"], out["choices"]], axis=-1).squeeze()


def _build_decorated_simulator(
    model_name: str, choices: list, obs_dim: int = 2
) -> Callable:
    """
    Build a decorated simulator function for a given model and choices.

    Parameters
    ----------
    model_name : str
        The name of the model to use for simulation.
    choices : list
        A list of possible choices for the simulator.

    Returns
    -------
    Callable
        A decorated simulator function.
    """
    decorated_simulator = decorate_atomic_simulator(
        model_name=model_name,
        choices=choices,
        obs_dim=obs_dim,
    )
    sim_wrapper = partial(
        hssm_sim_wrapper,
        simulator_fun=simulator,
        model=model_name,
    )
    return decorated_simulator(sim_wrapper)


def _get_simulator_fun_internal(simulator_fun: Callable | str):
    """
    Get the internal simulator function for a given model.

    Parameters
    ----------
    simulator_fun : Callable or str
        The simulator function or the name of the model as a string.

    Returns
    -------
    Callable
        The decorated simulator function.

    Raises
    ------
    ValueError
        If the simulator argument is not a string or a callable.
    """
    _validate_simulator_fun_arg(simulator_fun)

    if callable(simulator_fun):
        return cast("Callable[..., Any]", simulator_fun)

    simulator_fun_str = simulator_fun
    if simulator_fun_str not in ssms_model_config:
        _logger.warning(
            "You supplied a model '%s', which is currently not supported in "
            "the ssm_simulators package. An error will be thrown when sampling "
            "from the random variable or when using any "
            "posterior or prior predictive sampling methods.",
            simulator_fun_str,
        )
    choices = ssms_model_config.get(simulator_fun_str, {}).get("choices", [0, 1, 2])
    simulator_fun_internal = _build_decorated_simulator(
        model_name=simulator_fun_str,
        choices=choices,
    )
    return simulator_fun_internal


def _validate_simulator_fun(simulator_fun: Any) -> tuple[str, list, int]:
    """
    Validate that the simulator function has required attributes.

    Parameters
    ----------
    simulator_fun : Any
        The simulator function or object to validate.

    Returns
    -------
    tuple
        A tuple containing model_name, choices, and obs_dim_int.

    Raises
    ------
    ValueError
        If any required attribute is missing or invalid.
    """
    if not hasattr(simulator_fun, "model_name"):
        raise ValueError("The simulator function must have a `model_name` attribute.")
    model_name = simulator_fun.model_name

    if not hasattr(simulator_fun, "choices"):
        raise ValueError("The simulator function must have a `choices` attribute.")
    choices = simulator_fun.choices

    if not hasattr(simulator_fun, "obs_dim"):
        raise ValueError("The simulator function must have a `obs_dim` attribute.")
    obs_dim = simulator_fun.obs_dim

    if not isinstance(obs_dim, int):
        raise ValueError("The obs_dim attribute must be an integer")
    obs_dim_int = obs_dim

    return model_name, choices, obs_dim_int
