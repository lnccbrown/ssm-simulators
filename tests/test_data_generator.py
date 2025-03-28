from copy import deepcopy

import ssms


def test_data_generator():
    # Check included models
    assert list(ssms.config.model_config.keys())[:10] == [
        "ddm",
        "ddm_legacy",
        "angle",
        "weibull",
        "levy",
        "levy_angle",
        "full_ddm",
        "full_ddm_rv",
        "ddm_st",
        "ddm_truncnormt",
    ]

    # Take an example config for a given model
    model_conf = ssms.config.model_config["ddm"]

    assert type(model_conf["simulator"]).__name__ == "cython_function_or_method"

    assert callable(model_conf["simulator"])
    assert callable(model_conf["boundary"])

    del model_conf["simulator"]
    del model_conf["boundary"]

    assert model_conf == {
        "name": "ddm",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary_name": "constant",
        "boundary_params": [],
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 0.001],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
    }

    gen_config = ssms.config.data_generator_config["lan"]
    assert gen_config == {
        "output_folder": "data/lan_mlp/",
        "model": "ddm",
        "nbins": 0,
        "n_samples": 100000,
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": {
            "mode": 20,
            "choice_cnt": 0,
            "mean_rt": 17,
            "std": 0,
            "mode_cnt_rel": 0.95,
        },
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
        "smooth_unif": True,
        "kde_displace_t": False,
    }

    # Initialize the generator config (for MLP LANs)
    generator_config = deepcopy(gen_config)
    # Specify generative model (one from the list of included models mentioned above)
    generator_config["dgp_list"] = "angle"
    # Specify number of parameter sets to simulate
    generator_config["n_parameter_sets"] = 100
    # Specify how many samples a simulation run should entail
    generator_config["n_samples"] = 1000

    # Now let's define our corresponding `model_config`.
    model_config = ssms.config.model_config["angle"]
    my_dataset_generator = ssms.dataset_generators.lan_mlp.DataGenerator(
        generator_config=generator_config, model_config=model_config
    )
    training_data = my_dataset_generator.generate_data_training_uniform(save=False)
    assert list(training_data.keys()) == [
        "cpn_data",
        "cpn_labels",
        "cpn_no_omission_data",
        "cpn_no_omission_labels",
        "opn_data",
        "opn_labels",
        "gonogo_data",
        "gonogo_labels",
        "theta",
        "lan_data",
        "lan_labels",
        "binned_128",
        "binned_256",
        "generator_config",
        "model_config",
    ]

    expected_model_config = ssms.config.model_config["angle"]
    assert model_config == expected_model_config
