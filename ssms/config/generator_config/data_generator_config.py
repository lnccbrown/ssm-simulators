"""Data generator configuration.

Convenience functions for getting default configurations for data generation.
"""


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
