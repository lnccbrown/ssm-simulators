"""
This module defines the basic simulator function which is the main
workshorse of the package.
In addition some utility functions are provided that help
with preprocessing the output of the simulator function.
"""

from copy import deepcopy
from threading import Lock

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
from ssms.config import model_config
from ssms.config._modelconfig.base import boundary_config, drift_config

_global_rng = default_rng()
_rng_lock = Lock()


def _get_unique_seed() -> int:
    """
    Generate a unique seed for the random number generator.
    """
    with _rng_lock:
        return _global_rng.integers(0, 2**32 - 1)


def _make_valid_dict(dict_in: dict) -> dict:
    """Turn all values in dictionary into numpy arrays and make sure,
    that all thetas are either scalars or vectors of the same length

    Arguments:
    ----------
        dict_in: dict
            Dictionary of parameters, potentially with different length / types per
            parameter (key)

    Returns:
    --------
        dict_in: dict
            Aligned to same size np.float32 np.arrays for every parameter
    """

    collect_lengths: list[int] = []
    for key, value in dict_in.items():
        # Turn all values into numpy arrays
        if isinstance(value, list):
            dict_in[key] = np.array(value).astype(np.float32)
        elif isinstance(value, int | float):
            dict_in[key] = np.array([value]).astype(np.float32)

        # Squeeze all values to make sure they are 1d arrays
        dict_in[key] = np.squeeze(dict_in[key]).astype(np.float32)

        # Check if all thetas are either scalars or vectors of the same length
        if dict_in[key].ndim > 1:
            raise ValueError(f"Dimension of {key} is greater than 1")
        elif dict_in[key].ndim > 0:
            collect_lengths.append(
                dict_in[key].shape[0]
            )  # add vector parameters to list

    if len(set(collect_lengths)) > 1:
        raise ValueError(
            "thetas have to be either scalars or same length for "
            "all thetas which are not scalars"
        )

    # If there were any thetas provided as vectors (and they had the same length),
    # tile all scalar thetas to that length
    if len(set(collect_lengths)) > 0:
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.tile(value, collect_lengths[0])
    else:  # Expand scalars to 1d arrays
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.expand_dims(value, axis=0)
    return dict_in


def _theta_dict_to_array(
    theta: dict = {},  # TODO: #80 use dict instead of dict | None  # noqa: B006, FIX002
    model_param_list: list[str] | None = None,
) -> np.ndarray:
    """Converts theta dictionary to numpy array for use with simulator function.

    This function takes a dictionary of model parameters and a list of parameter names,
    and converts them into a 2D numpy array where each row represents a set of parameters
    and each column represents a specific parameter.

    Args:
        theta (dict): A dictionary containing model parameters. Default is an empty dict.
        model_param_list (list[str] | None): A list of parameter names in the desired order.
            If None, a ValueError will be raised.

    Returns:
        np.ndarray: A 2D numpy array of model parameters, with shape (n_sets, n_params),
            where n_sets is the number of parameter sets and n_params is the number of parameters.

    Raises:
        ValueError: If model_param_list is None.

    Example:
        >>> theta = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
        >>> model_param_list = ['a', 'b', 'c']
        >>> _theta_dict_to_array(theta, model_param_list)
        array([[1., 3., 5.],
               [2., 4., 6.]], dtype=float32)
    """
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")

    return np.stack([theta[param] for param in model_param_list], axis=1).astype(
        np.float32
    )


def _theta_array_to_dict(
    theta: np.ndarray | None = None, model_param_list: list[str] | None = None
) -> dict:
    """
    Converts theta array to dictionary for use with simulator function.

    This function takes a numpy array of parameter values and a list of parameter names,
    and converts them into a dictionary where keys are parameter names and values are
    the corresponding parameter values.

    Args:
        theta (np.ndarray | None): A 1D or 2D numpy array of parameter values.
            If None, a ValueError will be raised.
        model_param_list (list[str] | None): A list of parameter names.
            If None, a ValueError will be raised.

    Returns:
        dict: A dictionary where keys are parameter names and values are the corresponding
            parameter values from the input theta array.

    Raises:
        ValueError: If model_param_list is None, theta is None, or if the dimensions
            of theta do not match the length of model_param_list.

    Example:
        >>> theta = np.array([[1, 2, 3], [4, 5, 6]])
        >>> model_param_list = ['a', 'b', 'c']
        >>> _theta_array_to_dict(theta, model_param_list)
        {'a': array([1, 4]), 'b': array([2, 5]), 'c': array([3, 6])}
    """
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")
    elif theta is None:
        raise ValueError("theta array is not supplied")
    elif theta.ndim == 1 and len(model_param_list) != theta.shape[0]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    elif theta.ndim == 2 and len(model_param_list) != theta.shape[1]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    else:
        if theta.ndim == 1:
            theta = np.expand_dims(theta, axis=0).astype(np.float32)
        return {param: theta[:, i] for i, param in enumerate(model_param_list)}


def _preprocess_theta_generic(
    theta: list | np.ndarray | dict | pd.DataFrame,
) -> np.ndarray:
    """
    Preprocess the input theta to a consistent format.

    This function takes theta in various input formats and converts it to a
    standardized numpy array or dictionary format for further processing.

    Args:
        theta (list | np.ndarray | dict | pd.DataFrame): The input theta in
            various possible formats.

    Returns:
        np.ndarray | dict: The preprocessed theta in a consistent format.

    Raises:
        ValueError: If theta is not supplied as a list, numpy array, dictionary,
            or pandas DataFrame.
    """
    if isinstance(theta, list):
        theta = np.asarray(theta).astype(np.float32)
    elif isinstance(theta, np.ndarray):
        theta = theta.astype(np.float32)
    elif isinstance(theta, dict):
        theta = _make_valid_dict(deepcopy(theta))
    elif isinstance(theta, pd.DataFrame):
        theta = theta.to_dict(orient="list")
        theta = {k: np.array(v).astype(np.float32) for k, v in theta.items()}
    else:
        raise ValueError(
            "theta is not supplied as list, numpy array, dictionary, or pandas DataFrame!"
        )
    return theta


def _preprocess_theta_deadline(
    theta: dict | np.ndarray, deadline: bool, config: dict
) -> tuple[int, dict]:
    """
    Preprocess the input theta to a consistent format.

    This function takes theta in various input formats and converts it to a
    standardized numpy array or dictionary format for further processing.

    Args:
        theta (dict | np.ndarray): The input theta in dictionary or numpy array format
        deadline (bool): Whether the model is a deadline model
        config (dict): The model configuration

    Returns:
        tuple[int, dict]: The number of trials and the preprocessed theta in dictionary format
    """
    if not isinstance(theta, dict):
        theta = _theta_array_to_dict(theta, config["params"])

    n_trials = theta[config["params"][0]].shape[0]

    if not deadline:
        theta["deadline"] = np.tile(np.array([999], dtype=np.float32), n_trials)

    return n_trials, theta


def make_boundary_dict(config: dict, theta: dict) -> dict:
    """
    Create a dictionary containing boundary-related parameters and functions.

    This function extracts boundary-related parameters from the input theta dictionary,
    based on the boundary configuration specified in the config. It also retrieves
    the appropriate boundary function and multiplicative flag from the boundary_config.

    Args:
        config (dict): A dictionary containing model configuration, including the boundary name.
        theta (dict): A dictionary of parameter values, potentially including boundary-related parameters.

    Returns:
        dict: A dictionary containing:
            - boundary_params (dict): Extracted boundary-related parameters.
            - boundary_fun (callable): The boundary function corresponding to the specified boundary name.
            - boundary_multiplicative (bool): Flag indicating if the boundary is multiplicative.

    """
    boundary_name = config["boundary_name"]
    boundary_params = {
        param_name: value
        for param_name, value in theta.items()
        if param_name in boundary_config[boundary_name]["params"]
    }

    boundary_fun = boundary_config[boundary_name]["fun"]
    boundary_multiplicative = boundary_config[boundary_name]["multiplicative"]
    boundary_dict = {
        "boundary_params": boundary_params,
        "boundary_fun": boundary_fun,
        "boundary_multiplicative": boundary_multiplicative,
    }
    return boundary_dict


def make_drift_dict(config: dict, theta: dict) -> dict:
    """
    Create a dictionary containing drift-related parameters and functions.

    This function extracts drift-related parameters from the input theta dictionary,
    based on the drift configuration specified in the config. It also retrieves
    the appropriate drift function from the drift_config.

    Args:
        config (dict): A dictionary containing model configuration, including the drift name.
        theta (dict): A dictionary of parameter values, potentially including drift-related parameters.

    Returns:
        dict: A dictionary containing:
            - drift_fun (callable): The drift function corresponding to the specified drift name.
            - drift_params (dict): Extracted drift-related parameters.
            If no drift name is specified in config, returns an empty dictionary.
    """
    if "drift_name" in config:
        drift_name = config["drift_name"]
        drift_params = {
            param_name: value
            for param_name, value in theta.items()
            if param_name in drift_config[drift_name]["params"]
        }
        drift_fun = drift_config[drift_name]["fun"]
        drift_dict = {"drift_fun": drift_fun, "drift_params": drift_params}
    else:
        drift_dict = {}
    return drift_dict


# TODO: Make useful as independent utility,
# this is dropped from basic simulator call now
def bin_simulator_output_pointwise(
    out: tuple[np.ndarray, np.ndarray] = (np.array([0]), np.array([0])),
    bin_dt: float = 0.04,
    nbins: int = 0,
) -> np.ndarray:  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out: tuple
            Output of the 'simulator' function
        bin_dt: float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins: int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the
            number of bins automatically.

    Returns
    -------
        2d array. The first columns collects bin-identifiers
        by trial, the second column lists the corresponding choices.
    """
    out_copy = deepcopy(out)

    # Generate bins
    if nbins == 0:
        nbins = int(out["metadata"]["max_t"] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf

    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp

    out_copy[1][out_copy[1] == -1] = 0

    return np.concatenate([out_copy[0], out_copy[1]], axis=-1).astype(np.int32)


def bin_simulator_output(
    out: dict | None = None,
    bin_dt: float = 0.04,
    nbins: int = 0,
    max_t: float = -1,
    freq_cnt: bool = False,
) -> np.ndarray:  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out : dict
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        max_t : float <default=-1>
            Override the 'max_t' metadata as part of the simulator output.
            Sometimes useful, but usually default will do the job.
        freq_cnt : bool <default=False>
            Decide whether to return proportions (default) or counts in bins.

    Returns
    -------
        A histogram of counts or proportions.

    """

    if out is None:
        raise ValueError("out is not supplied")

    if max_t == -1:
        max_t = out["metadata"]["max_t"]

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros((nbins, len(out["metadata"]["possible_choices"])))

    for choice in out["metadata"]["possible_choices"]:
        counts[:, cnt] = np.histogram(out["rts"][out["choices"] == choice], bins=bins)[
            0
        ]
        cnt += 1

    if freq_cnt is False:
        counts = counts / out["metadata"]["n_samples"]

    return counts


def validate_ssm_parameters(model: str, theta: dict) -> None:
    """
    Validate the parameters for Sequential Sampling Models (SSM).

    This function checks the validity of parameters for different SSM models.
    It performs specific checks based on the model type.

    Args:
        model (str): The name of the SSM model.
        theta (dict): A dictionary containing the model parameters.

    Raises:
        ValueError: If any of the parameter validations fail.
    """

    def check_num_drifts_and_actions(drifts: np.ndarray, num_actions: int) -> None:
        """
        Check if the number of drift rates matches the number of actions.

        Args:
            drifts (np.ndarray): Array of drift rates.
            num_actions (int): Number of actions.

        Raises:
            ValueError: If the number of drift rates doesn't match the number of actions.
        """
        drifts = np.array(drifts)
        if drifts.shape[1] != num_actions:
            raise ValueError("Number of drift rates does not match number of actions")

    def check_lba_drifts_sum(drifts: np.ndarray) -> None:
        """
        Check if the drift rates for LBA models sum to 1 for each trial.

        Args:
            drifts (np.ndarray): Array of drift rates.

        Raises:
            ValueError: If the drift rates don't sum to 1 for any trial.
        """
        v_sum = np.sum(drifts, axis=1)
        if np.any(v_sum <= 0.99) or np.any(v_sum >= 1.01):
            raise ValueError("Drift rates do not sum to 1 for each trial")

    def check_if_z_gt_a(z: np.ndarray, a: np.ndarray) -> None:
        """
        Check if the starting point (z) is greater than or equal to the threshold (a).

        Args:
            z (np.ndarray): Array of starting points.
            a (np.ndarray): Array of thresholds.

        Raises:
            ValueError: If z >= a for any trial.
        """
        if np.any(z >= a):
            raise ValueError("Starting point z >= a for at least one trial")

    if model in [
        "lba_3_vs_constraint",
        "lba_angle_3_vs_constraint",
        "lba_angle_3",
        "dev_rlwm_lba_race_v1",
        "dev_rlwm_lba_race_v2",
        "dev_rlwm_lba_pw_v1",
    ]:
        if model in ["lba_3_vs_constraint", "lba_angle_3_vs_constraint"]:
            check_lba_drifts_sum(theta["v"])
            check_if_z_gt_a(theta["z"], theta["a"])
        elif model in ["dev_rlwm_lba_race_v1"]:
            check_lba_drifts_sum(theta["vRL"])
            check_lba_drifts_sum(theta["vWM"])
            check_if_z_gt_a(theta["z"], theta["a"])
        elif model in [
            "lba3",
            "lba2",
            "lba_angle_3",
            "dev_rlwm_lba_pw_v1",
            "dev_rlwm_lba_race_v2",
        ]:
            check_if_z_gt_a(theta["z"], theta["a"])




def make_noise_vec(
    sigma_noise: float | np.ndarray, n_trials: int, n_particles: int
) -> np.ndarray:
    """
    Create a noise vector for simulation.

    This function generates a noise vector based on the provided sigma_noise parameter,
    replicating it across trials and particles as needed.

    Arguments
    ---------
        sigma_noise (float | np.ndarray): Standard deviation of the noise. Can be a single float value
            or a numpy array.
        n_trials (int): Number of trials to simulate.
        n_particles (int): Number of particles per trial.

    Returns
    -------
        np.ndarray: A noise vector with appropriate shape for the simulation,
            containing the replicated sigma_noise values.
    """
    if n_particles == 1 or n_particles is None:
        shape_tuple = n_trials
    else:
        shape_tuple = (n_trials, 1)

    noise_vec = np.tile(
        np.array(
            (
                [sigma_noise[0]] * n_particles
                if isinstance(sigma_noise, np.ndarray)
                else [sigma_noise] * n_particles
            ),
            dtype=np.float32,
        ),
        shape_tuple,
    )
    return noise_vec


def simulator(
    theta: list | np.ndarray | dict | pd.DataFrame,
    model: str = "angle",
    n_samples: int = 1000,
    delta_t: float = 0.001,
    max_t: float = 20,
    no_noise: bool = False,
    sigma_noise: float | None = None,
    smooth_unif: bool = True,
    random_state: int | None = None,
    return_option: str = "full",
) -> dict:
    """Basic data simulator for the models included in HDDM.

    Arguments
    ---------
        theta : list, numpy.array, dict or pd.DataFrame
            Parameters of the simulator. If 2d array, each row is treated as a 'trial'
            and the function runs n_sample * n_trials simulations.
        deadline : numpy.array <default=None>
            If supplied, the simulator will run a deadline model. RTs will be returned
        model: str <default='angle'>
            Determines the model that will be simulated.
        n_samples: int <default=1000>
            Number of simulation runs for each row in the theta argument.
        delta_t: float
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float
            Maximum reaction the simulator can reach
        no_noise: bool <default=False>
            Turn noise of (useful for plotting purposes mostly)
        sigma_noise: float | None <default=None>
            Standard deviation of noise in the diffusion process. If None, defaults to 1.0 for most models
            and 0.1 for LBA models. If no_noise is True, sigma_noise will be set to 0.0.
            If 'sd' or 's' is passed via theta dictionary, sigma_noise must be None.
        smooth_unif: bool <default=True>
            Whether to add uniform random noise to RTs to smooth the distributions.
        random_state: int | None <default=None>
            Integer passed to random_seed function in the simulator.
            Can be used for reproducibility.
        return_option: str <default='full'>
            Determines what the function returns. Can be either
            'full' or 'minimal'. If 'full' the function returns
            a dictionary with keys 'rts', 'responses' and 'metadata', and
            metadata contains the model parameters and some additional
            information. 'metadata' is a simpler dictionary with less information
            if 'minimal' is chosen.

    Return
    ------
    dictionary where keys
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """
    # Preprocess theta to be a 2d numpy array with correct column ordering
    # (if supplied as 2d array or list in the first place,
    # user has to supply the correct ordering to begin with)

    if "_deadline" in model:
        deadline = True
        model = model.replace("_deadline", "")
    else:
        deadline = False

    model_config_local = deepcopy(model_config[model])
    
    if deadline:
        model_config_local["params"] += ["deadline"]

    if random_state is None:
        random_state = _get_unique_seed()

    theta = _preprocess_theta_generic(theta)
    n_trials, theta = _preprocess_theta_deadline(theta, deadline, model_config_local)

    # Initialize dictionary that collects
    # simulator inputs that are common across simulator functions
    sim_param_dict = {
        "max_t": max_t,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "delta_t": delta_t,
        "random_state": random_state,
        "return_option": return_option,
        "smooth_unif": smooth_unif,
    }

    # Update all values of sim_param_dict that are defined in locals()
    # locals_dict = locals()
    # sim_param_dict = {
    #     key_: locals_dict[key_] for key_ in locals_dict if key_ in sim_param_dict
    # }

    # Fix up noise level
    if "sd" in theta or "s" in theta:
        if sigma_noise is not None:
            raise ValueError(
                "sigma_noise parameter should be None if 'sd' or 's' is passed via theta dictionary"
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
        elif "lba" in model and sigma_noise is None:
            sigma_noise = 0.1
        elif sigma_noise is None:
            sigma_noise = 1.0

    noise_vec = make_noise_vec(sigma_noise, n_trials, model_config_local["n_particles"])

    if "lba" in model:
        theta["sd"] = noise_vec
    else:
        theta["s"] = noise_vec

    # Process theta
    theta = SimpleThetaProcessor().process_theta(theta, model_config_local, n_trials)

    # Make boundary dictionary
    boundary_dict = make_boundary_dict(model_config_local, theta)
    # Make drift dictionary
    drift_dict = make_drift_dict(model_config_local, theta)

    # Check if parameters are valid
    validate_ssm_parameters(model, theta)

    # Call to the simulator
    x = model_config_local["simulator"](
        **theta,
        **boundary_dict,
        **drift_dict,
        **sim_param_dict,
    )
    
    # Ensure x is a dictionary
    if not isinstance(x, dict):
        raise TypeError(
            f"Expected simulator to return a dictionary, got {type(x).__name__}"
        )

    # Postprocess simulator output ----------------------------
    # Additional model outputs, easy to compute:
    # Choice probability
    x["choice_p"] = np.zeros((n_trials, len(x["metadata"]["possible_choices"])))
    x["choice_p_no_omission"] = np.zeros(
        (n_trials, len(x["metadata"]["possible_choices"]))
    )
    x["omission_p"] = np.zeros((n_trials, 1))
    x["nogo_p"] = np.zeros((n_trials, 1))
    x["go_p"] = np.zeros((n_trials, 1))

    # Calculate choice probabilities by trial
    # TODO: #79 vectorize this  # noqa: FIX002
    for k in range(n_trials):
        out_len = x["rts"][:, k, :].shape[0]
        out_len_no_omission = x["rts"][:, k, :][x["rts"][:, k, :] != -999].shape[0]

        for n, choice in enumerate(x["metadata"]["possible_choices"]):
            x["choice_p"][k, n] = (x["choices"][:, k, :] == choice).sum() / out_len
            if out_len_no_omission > 0:
                x["choice_p_no_omission"][k, n] = (
                    x["choices"][:, k, :][x["rts"][:, k, :] != -999] == choice
                ).sum() / out_len_no_omission
            else:
                # AF-TODO: Don't get why -999 is used here
                x["choice_p_no_omission"][k, n] = -999

        # Omission Probability (deadline)
        x["omission_p"][k, 0] = (x["rts"][:, k, :] == -999).sum() / out_len

        # Nogo Probability
        # NOTE: If deadline is set in simulator --> this is the nogo probability
        # + the omission probability
        x["nogo_p"][k, 0] = (
            # AF-TODO: This should rather have a designated no-go choice
            # instead of `max`
            (x["choices"][:, k, :] != max(x["metadata"]["possible_choices"]))
            | (x["rts"][:, k, :] == -999)
        ).sum() / out_len
        x["go_p"][k, 0] = 1 - x["nogo_p"][k, 0]

    # Output compatibility
    if n_trials == 1:
        x["rts"] = np.squeeze(x["rts"], axis=1)
        x["choices"] = np.squeeze(x["choices"], axis=1)
    if n_trials > 1 and n_samples == 1:
        x["rts"] = np.squeeze(x["rts"], axis=0)
        x["choices"] = np.squeeze(x["choices"], axis=0)

    x["metadata"]["model"] = model

    x["binned_128"] = np.expand_dims(
        bin_simulator_output(x, nbins=128, max_t=-1, freq_cnt=True), axis=0
    )
    x["binned_256"] = np.expand_dims(
        bin_simulator_output(x, nbins=256, max_t=-1, freq_cnt=True), axis=0
    )
    return x
