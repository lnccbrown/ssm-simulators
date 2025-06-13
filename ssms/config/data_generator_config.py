"""Data generator configuration.

Convenience functions for getting default configurations for data generation.
"""

import warnings


class DeprecatedDict(dict):
    """
    A pseudo-dictionary that raises a DeprecationWarning when accessed.
    This is used to indicate that the configuration dictionary is deprecated
    and should not be used directly.

    Parameters
    ----------
    lookup_func : callable, optional
        A function that takes a key and returns the corresponding value.
    alternative : str, optional
        A string indicating the alternative method to use instead of this
        configuration dictionary."""

    def __init__(self, lookup_func=None, alternative="get_default_generator_config"):
        self._lookup_func = lookup_func
        self._alternative = alternative

    def __getitem__(self, key):
        message = f"Accessing this configuration dict is deprecated and will be removed in a future version. Use `{self._alternative}` instead."
        warnings.warn(
            message,
            DeprecationWarning,
            stacklevel=2,
        )
        if self._lookup_func is None or not callable(self._lookup_func):
            raise ValueError("A valid callable lookup_func must be provided.")
        return self._lookup_func(key)


def get_kde_simulation_filters() -> dict:
    return {
        "mode": 20,  # != (if mode is max_rt)
        "choice_cnt": 0,  # > (each choice receive at least 10 samples )
        "mean_rt": 17,  # < (mean_rt is smaller than specified value
        "std": 0,  # > (std is positive for each choice)
        "mode_cnt_rel": 0.95,  # < (mode can't be large proportion of all samples)
    }


def get_opn_only_config() -> dict:
    return {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    }


def get_cpn_only_config() -> dict:
    return {
        "output_folder": "data/cpn_only/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "smooth_unif": False,
    }


def get_lan_config() -> dict:
    return {
        "output_folder": "data/lan_mlp/",
        "model": "ddm",  # should be ['ddm'],
        "nbins": 0,
        "n_samples": 100_000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
        "smooth_unif": True,
        "kde_displace_t": False,
    }


def get_ratio_estimator_config() -> dict:
    return {
        "output_folder": "data/ratio/",
        "model": "ddm",
        "nbins": 0,
        "n_samples": {"low": 100000, "high": 100000},
        "n_parameter_sets": 100000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    }


def get_defective_detector_config() -> dict:
    return {
        "output_folder": "data/defective_detector/",
        "model": "ddm",
        "nbins": 0,
        "n_samples": {"low": 100_000, "high": 100_000},
        "n_parameter_sets": 100_000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1_000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": get_kde_simulation_filters(),
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    }


def get_snpe_config() -> dict:
    return {
        "output_folder": "data/snpe_training/",
        "model": "ddm",  # should be ['ddm'],
        "n_samples": 5000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subruns": 10,
        "separate_response_channels": False,
    }


def get_default_generator_config(approach) -> dict:
    """
    Dynamically retrieve the data generator configuration for the given approach.

    Parameters
    ----------
    approach : str
        The approach corresponding to the desired data generator configuration.
        Valid options include:
        - "opn_only"
        - "cpn_only"
        - "lan"
        - "ratio_estimator"
        - "defective_detector"
        - "snpe"

    Returns
    -------
    dict
        The configuration dictionary for the specified approach.

    Raises
    ------
    KeyError
        If the approach is not found in the available configurations.
    """
    config_functions = {
        "opn_only": get_opn_only_config,
        "cpn_only": get_cpn_only_config,
        "lan": get_lan_config,
        "ratio_estimator": get_ratio_estimator_config,
        "defective_detector": get_defective_detector_config,
        "snpe": get_snpe_config,
    }

    if approach not in config_functions:
        raise KeyError(
            f"'{approach}' is not a valid data generator configuration approach."
        )

    return config_functions[approach]()


# TODO: Add for compatibility with lanfactory's test_end_to_end.py test. Delete when
#       lanfactory uses get_default_generator_config.
data_generator_config = DeprecatedDict(
    get_default_generator_config, "get_default_generator_config"
)