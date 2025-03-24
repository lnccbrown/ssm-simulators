"""
This module defines a data generator class for use with LANs.
The class defined below can be used to generate training data
compatible with the expectations of LANs.
"""

import pickle
import uuid
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import psutil
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import mode

from ssms.basic_simulators.simulator import (
    _theta_dict_to_array,
    simulator,  # , bin_simulator_output
)
from ssms.config import KDE_NO_DISPLACE_T
from ssms.support_utils import kde_class
from ssms.support_utils.utils import sample_parameters_from_constraints


def _generate_random_seeds(*, a=400000000, size):
    """
    Generate an array of random seeds.

    Parameters:
    a (int, optional): The upper limit for the random seed values. Default is 400000000.
    size (int): The number of random seeds to generate.

    Returns:
    numpy.ndarray: An array of random seeds.
    """
    return np.random.choice(a, size=size)


class DataGenerator:
    """The DataGenerator() class is used to generate training data
      for various likelihood approximators.

    Attributes
    ----------
        generator_config: dict
            Configuation dictionary for the data generator.
            (For an example load ssms.config.data_generator_config['lan'])
        model_config: dict
            Configuration dictionary for the model to be simulated.
            (For an example load ssms.config.model_config['ddm'])
    Methods
    -------
        generate_data_training_uniform(save=False, verbose=True, cpn_only=False)
            Generates training data for LANs.
        get_simulations(theta=None, random_seed=None)
            Generates simulations for a given parameter set.
        _filter_simulations(simulations=None)
            Filters simulations according to the criteria
            specified in the generator_config.
        _make_kde_data(simulations=None, theta=None)
            Generates KDE data from simulations.
        _mlp_get_processed_data_for_theta(random_seed_tuple)
            Helper function for generating training data for MLPs.
        _cpn_get_processed_data_for_theta(random_seed_tuple)
            Helper function for generating training data for CPNs.
        _get_rejected_parameter_setups(random_seed_tuple)
            Helper function that collectes parameters sets which were rejected
            by the filter used in the _filter_simulations() method.
        _make_save_file_name(unique_tag=None)
            Helper function for generating save file names.
        _build_simulator()
            Builds simulator function for LANs.
        _get_ncpus()
            Helper function for determining the number of
            cpus to use for parallelization.

    Returns
    -------
        DataGenerator object
    """

    def __init__(
        self, generator_config: dict | None = None, model_config: dict | None = None
    ):
        """Initialize data generator class.

        Arguments
        ---------
        generator_config: dict
            Configuration dictionary for the data generator.
            (For an example load ssms.config.data_generator_config['lan'])
        model_config: dict
            Configuration dictionary for the model to be simulated.
            (For an example load ssms.config.model_config['ddm'])

        Raises
        ------
        ValueError
            If no generator_config or model_config is specified.

        Returns
        -------
        DataGenerator object
        """
        # INIT -----------------------------------------
        if generator_config is None:
            raise ValueError("No generator_config specified")
        elif model_config is None:
            raise ValueError("No model_config specified")
        else:
            self.generator_config = deepcopy(generator_config)
            self.model_config = deepcopy(model_config)

            # Account for deadline if in model name
            if "deadline" in self.generator_config["model"]:
                self.model_config["params"].append("deadline")
                if isinstance(self.model_config["param_bounds"], list):
                    self.model_config["param_bounds"][0].append(0.001)
                    self.model_config["param_bounds"][1].append(10)
                    self.model_config["default_params"].append(10)
                    self.model_config["name"] += "_deadline"
                    self.model_config["n_params"] += 1
                elif isinstance(self.model_config["param_bounds"], dict):
                    self.model_config["param_bounds"]["deadline"] = (0.001, 10)
                    self.model_config["default_params"].append(10)
                    self.model_config["name"] += "_deadline"
                    self.model_config["n_params"] += 1

            if "kde_displace_t" not in self.generator_config:
                self.generator_config["kde_displace_t"] = False

            if (
                self.generator_config["kde_displace_t"] is True
                and self.model_config["name"].split("_deadline")[0] in KDE_NO_DISPLACE_T
            ):
                warnings.warn(
                    f"kde_displace_t is True, but model is in {KDE_NO_DISPLACE_T}. Overriding setting to False",
                    stacklevel=2,
                )
                self.generator_config["kde_displace_t"] = False

            # Define constrained parameter space as dictionary
            # and add to internal model config
            # AF-COMMENT: This will eventually be replaced so that
            # configs always have dictionary format for parameter
            # bounds
            # print(self.model_config)
            # print(type(self.model_config))
            if isinstance(self.model_config["param_bounds"], list):
                bounds_tmp = self.model_config["param_bounds"]
                names_tmp = self.model_config["params"]
                self.model_config["constrained_param_space"] = {
                    names_tmp[i]: (bounds_tmp[0][i], bounds_tmp[1][i])
                    for i in range(len(names_tmp))
                }
            elif isinstance(self.model_config["param_bounds"], dict):
                self.model_config["constrained_param_space"] = self.model_config[
                    "param_bounds"
                ]
            else:
                raise ValueError("param_bounds must be a list or a dictionary")

            # print(self.model_config)

            self._build_simulator()
            self._get_ncpus()

        # Make output folder if not already present
        folder_str_split = self.generator_config["output_folder"].split()

        cnt = 0
        folder_partial = ""
        for folder_str_part in folder_str_split:
            if cnt > 0:
                folder_partial += "/" + folder_str_part
            else:
                folder_partial += folder_str_part

            print("checking: ", folder_partial)

            if not Path(folder_partial).exists():
                Path(folder_partial).mkdir(parents=True)

    def _get_ncpus(self):
        """Get the number cpus to use for parallelization."""
        # Get number of cpus
        if self.generator_config["n_cpus"] == "all":
            n_cpus = psutil.cpu_count(logical=False)
            print("n_cpus used: ", n_cpus)
        else:
            n_cpus = self.generator_config["n_cpus"]

        self.generator_config["n_cpus"] = n_cpus

    def _build_simulator(self):
        """Build simulator function for LANs."""
        self.simulator = partial(
            simulator,
            n_samples=self.generator_config["n_samples"],
            max_t=self.generator_config["max_t"],
            bin_dim=0,
            delta_t=self.generator_config["delta_t"],
            smooth_unif=self.generator_config["smooth_unif"],
        )

    def get_simulations(
        self, theta: dict | None = None, random_seed: int | None = None
    ):
        """Generates simulations for a given parameter set."""
        out = self.simulator(
            theta=theta,
            model=self.model_config["name"],
            random_state=random_seed,
        )
        return out

    def _filter_simulations(
        self,
        simulations: dict | None = None,
    ):
        """Filters simulations according to the criteria
        specified in the generator_config."""
        if simulations is None:
            raise ValueError("No simulations provided")

        keep = 1
        n_sim = simulations["rts"].shape[0]
        for choice_tmp in simulations["metadata"]["possible_choices"]:
            tmp_rts = simulations["rts"][
                (simulations["choices"] == choice_tmp) & (simulations["rts"] != -999)
            ]

            tmp_n_c = len(tmp_rts)
            if tmp_n_c > 0:
                mode_, mode_cnt_ = mode(tmp_rts, keepdims=False)
                std_ = np.std(tmp_rts)
                mean_ = np.mean(tmp_rts)
                if tmp_n_c < 5:
                    mode_cnt_rel_ = 0
                else:
                    mode_cnt_rel_ = mode_cnt_ / tmp_n_c

            else:
                mode_ = -1
                mode_cnt_ = 0
                mean_ = -1
                std_ = 1
                mode_cnt_rel_ = 0

            # AF-TODO: More flexible way with list of
            # filter objects that provides for each filter
            #  1. Function to compute statistic (e.g. mode)
            #  2. Comparison operator (e.g. <=, != , etc.)
            #  3. Comparator (number to test against)

            keep = (
                keep
                & (mode_ < self.generator_config["simulation_filters"]["mode"])
                & (mean_ <= self.generator_config["simulation_filters"]["mean_rt"])
                & (std_ > self.generator_config["simulation_filters"]["std"])
                & (
                    mode_cnt_rel_
                    <= self.generator_config["simulation_filters"]["mode_cnt_rel"]
                )
                & (tmp_n_c > self.generator_config["simulation_filters"]["choice_cnt"])
            )
        return keep, np.array(
            [mode_, mean_, std_, mode_cnt_rel_, tmp_n_c, n_sim], dtype=np.float32
        )

    def _make_kde_data(
        self, simulations: dict | None = None, theta: dict | None = None
    ):
        """Generates KDE data from simulations.

        Arguments
        ---------
        simulations: dict
            Dictionary containing the simulations.
        theta: dict
            Dictionary containing the parameters.

        Returns
        -------
        out: np.array
            Array containing the KDE data.
        """
        if simulations is None:
            raise ValueError("No simulations provided")
        if theta is None:
            raise ValueError("No theta provided")

        n = self.generator_config["n_training_samples_by_parameter_set"]
        p = self.generator_config["kde_data_mixture_probabilities"]
        n_kde = int(n * p[0])
        n_unif_up = int(n * p[1])
        n_unif_down = int(n * p[2])

        if self.generator_config["separate_response_channels"]:
            out = np.zeros(
                (
                    n_kde + n_unif_up + n_unif_down,
                    2 + self.model_config["nchoices"] + len(theta.items()),
                )
            )
        else:
            out = np.zeros((n_kde + n_unif_up + n_unif_down, 3 + len(theta.items())))

        out[:, : len(theta.items())] = np.tile(
            np.stack([theta[key_] for key_ in self.model_config["params"]], axis=1),
            (n_kde + n_unif_up + n_unif_down, 1),
        )

        tmp_kde = kde_class.LogKDE(
            simulations,
            displace_t=self.generator_config["kde_displace_t"],
        )

        # Get kde part
        samples_kde = tmp_kde.kde_sample(n_samples=n_kde)
        likelihoods_kde = tmp_kde.kde_eval(data=samples_kde).ravel()

        if self.generator_config["separate_response_channels"]:
            out[:n_kde, (-2 - self.model_config["nchoices"])] = samples_kde[0].ravel()

            r_cnt = 0
            choices = samples_kde[1].ravel()
            for response in simulations["metadata"]["possible_choices"]:
                out[:n_kde, ((-1 - self.model_config["nchoices"]) + r_cnt)] = (
                    choices == response
                ).astype(int)
                r_cnt += 1
        else:
            out[:n_kde, -3] = samples_kde["rts"].ravel()
            out[:n_kde, -2] = samples_kde["choices"].ravel()

        out[:n_kde, -1] = likelihoods_kde

        # Get positive uniform part:
        choice_tmp = np.random.choice(
            simulations["metadata"]["possible_choices"], size=n_unif_up
        )

        if simulations["metadata"]["max_t"] < 100:
            rt_tmp = np.random.uniform(
                low=0.0001, high=simulations["metadata"]["max_t"], size=n_unif_up
            )
        else:
            rt_tmp = np.random.uniform(low=0.0001, high=100, size=n_unif_up)

        likelihoods_unif = tmp_kde.kde_eval(
            data={"rts": rt_tmp, "choices": choice_tmp}
        ).ravel()

        out[n_kde : (n_kde + n_unif_up), -3] = rt_tmp
        out[n_kde : (n_kde + n_unif_up), -2] = choice_tmp
        out[n_kde : (n_kde + n_unif_up), -1] = likelihoods_unif

        # Get negative uniform part:
        choice_tmp = np.random.choice(
            simulations["metadata"]["possible_choices"], size=n_unif_down
        )

        rt_tmp = np.random.uniform(low=-1.0, high=0.0001, size=n_unif_down)

        out[(n_kde + n_unif_up) :, -3] = rt_tmp
        out[(n_kde + n_unif_up) :, -2] = choice_tmp
        out[(n_kde + n_unif_up) :, -1] = self.generator_config["negative_rt_cutoff"]
        return out.astype(np.float32)

    def parameter_transform_for_data_gen(self, theta: dict):
        """
        Function to impose constraints on the parameters for data generation.

        Arguments
        ---------
            theta: dict
                Dictionary containing the parameters.

        Returns
        -------
            theta: dict
                Dictionary containing the transformed parameters.
        """

        # print(theta)

        if self.model_config["name"] in ["lba_angle_3"]:
            # ensure that a is always greater than z
            if theta["a"] <= theta["z"]:
                tmp = theta["a"]
                theta["a"] = theta["z"]
                theta["z"] = tmp

        if self.model_config["name"] == "dev_rlwm_lba_pw_v1":
            # ensure that a is always greater than z
            if theta["a"] <= theta["z"]:
                tmp = theta["a"]
                theta["a"] = theta["z"]
                theta["z"] = tmp

        # For LBA-based models, we need to ensure that the drift rates sum to 1
        if self.model_config["name"] == "dev_rlwm_lba_race_v1":
            # normalize the RL drift rates
            v_rl_sum = (
                np.sum([theta["v_RL_0"], theta["v_RL_1"], theta["v_RL_2"]]).astype(
                    np.float32
                )
                + 1e-20
            )
            theta["v_RL_0"] = (theta["v_RL_0"] + (1e-20 / 3)) / v_rl_sum
            theta["v_RL_1"] = (theta["v_RL_1"] + (1e-20 / 3)) / v_rl_sum
            theta["v_RL_2"] = (theta["v_RL_2"] + (1e-20 / 3)) / v_rl_sum

            # theta[0:3] = theta[0:3] / np.sum(theta[0:3])

            # normalize the WM drift rates
            v_wm_sum = (
                np.sum([theta["v_WM_0"], theta["v_WM_1"], theta["v_WM_2"]]).astype(
                    np.float32
                )
                + 1e-20
            )
            theta["v_WM_0"] = (theta["v_WM_0"] + (1e-20 / 3)) / v_wm_sum
            theta["v_WM_1"] = (theta["v_WM_1"] + (1e-20 / 3)) / v_wm_sum
            theta["v_WM_2"] = (theta["v_WM_2"] + (1e-20 / 3)) / v_wm_sum

            # theta[3:6] = theta[3:6] / np.sum(theta[3:6])

            # ensure that a is always greater than z
            # if not true switch position between a and z
            # AF-COMMENT: Is this keeping the uniform on hypercube in tact?
            if theta["a"] <= theta["z"]:
                tmp = theta["a"]
                theta["a"] = theta["z"]
                theta["z"] = tmp

        return theta

    def _mlp_get_processed_data_for_theta(self, random_seed_tuple: tuple | list):
        np.random.seed(random_seed_tuple[0])
        keep = 0
        # Keep simulating until we are happy with data
        while not keep:
            theta_dict = sample_parameters_from_constraints(
                self.model_config["constrained_param_space"], 1
            )
            # Run extra checks on parameters
            # (currently used only for very specific RLWM model)
            theta_dict = self.parameter_transform_for_data_gen(theta_dict)

            # Run simulations
            simulations = self.get_simulations(
                theta=deepcopy(theta_dict), random_seed=random_seed_tuple[1]
            )
            # Check if simulations pass filter
            keep, stats = self._filter_simulations(simulations)

        # Now that we are happy with data
        # construct KDEs
        kde_data = self._make_kde_data(simulations=simulations, theta=theta_dict)

        if len(simulations["metadata"]["possible_choices"]) == 2:
            cpn_labels = np.expand_dims(simulations["choice_p"][0, 1], axis=0)
            cpn_no_omission_labels = np.expand_dims(
                simulations["choice_p_no_omission"][0, 1], axis=0
            )
        else:
            cpn_labels = simulations["choice_p"]
            cpn_no_omission_labels = simulations["choice_p_no_omission"]

        # Make theta array
        theta_array = _theta_dict_to_array(theta_dict, self.model_config["params"])
        return {
            "lan_data": kde_data[:, :-1],
            "lan_labels": kde_data[:, -1],
            "cpn_data": theta_array,  # np.expand_dims(theta, axis=0),
            "cpn_labels": cpn_labels,
            "cpn_no_omission_data": theta_array,
            "cpn_no_omission_labels": cpn_no_omission_labels,
            "opn_data": theta_array,
            "opn_labels": simulations["omission_p"],
            "gonogo_data": theta_array,
            "gonogo_labels": simulations["nogo_p"],
            "binned_128": simulations["binned_128"],
            "binned_256": simulations["binned_256"],
            "theta": theta_array,
        }

    def _cpn_get_processed_data_for_theta(self, random_seed_tuple: tuple | list):
        np.random.seed(random_seed_tuple[0])
        theta_dict = sample_parameters_from_constraints(
            self.model_config["constrained_param_space"], 1
        )

        # Run the simulator
        simulations = self.get_simulations(
            theta=deepcopy(theta_dict), random_seed=random_seed_tuple[1]
        )

        if len(simulations["metadata"]["possible_choices"]) == 2:
            cpn_labels = np.expand_dims(simulations["choice_p"][0, 1], axis=0)
            cpn_no_omission_labels = np.expand_dims(
                simulations["choice_p_no_omission"][0, 1], axis=0
            )
        else:
            cpn_labels = simulations["choice_p"]
            cpn_no_omission_labels = simulations["choice_p_no_omission"]

        # Make theta array
        theta_array = _theta_dict_to_array(theta_dict, self.model_config["params"])

        return {
            "cpn_data": theta_array,
            "cpn_labels": cpn_labels,
            "cpn_no_omission_data": theta_array,
            "cpn_no_omission_labels": cpn_no_omission_labels,
            "opn_data": theta_array,
            "opn_labels": simulations["omission_p"],
            "gonogo_data": theta_array,
            "gonogo_labels": simulations["nogo_p"],
            "theta": theta_array,
        }

    def _get_rejected_parameter_setups(self, random_seed_tuple: tuple | list):
        np.random.seed(random_seed_tuple[0])
        rejected_thetas = []
        keep = 1
        rej_cnt = 0
        while rej_cnt < 100:
            theta_dict = sample_parameters_from_constraints(
                self.model_config["constrained_param_space"], 1
            )
            # Run extra checks on parameters
            # (currently used only for very specific RLWM model)
            theta_dict = self.parameter_transform_for_data_gen(theta_dict)

            simulations = self.get_simulations(
                theta=theta_dict, random_seed=random_seed_tuple[1]
            )
            keep, stats = self._filter_simulations(simulations)

            if keep == 0:
                print("simulation rejected")
                print("stats: ", stats)
                print("theta", theta_dict)
                rejected_thetas.append(theta_dict)
            rej_cnt += 1

        return rejected_thetas

    def generate_data_training_uniform(
        self, save: bool = False, verbose: bool = True, cpn_only: bool = False
    ):
        """Generates training data for LANs.

        Arguments
        ---------
            save: bool
                If True, the generated data is saved to disk.
            verbose: bool
                If True, progress is printed to the console.
            cpn_only: bool
                If True, only choice probabilities are computed.
                This is useful for training CPNs.

        Returns
        -------
            data: dict
                Dictionary containing the generated data.
        """
        seeds_1, seeds_2 = [
            _generate_random_seeds(size=self.generator_config["n_parameter_sets"])
            for _ in range(2)
        ]
        seed_args = [(seeds_1[i], seeds_2[i]) for i in range(seeds_1.shape[0])]

        # Inits
        subrun_n = (
            self.generator_config["n_parameter_sets"]
            // self.generator_config["n_subruns"]
        )

        # Get Simulations
        out_list = []
        for i in range(self.generator_config["n_subruns"]):
            if verbose:
                print(
                    "simulation round:",
                    i + 1,
                    " of",
                    self.generator_config["n_subruns"],
                )
            if self.generator_config["n_cpus"] > 1:
                if cpn_only:
                    with Pool(processes=self.generator_config["n_cpus"] - 1) as pool:
                        out_list += pool.map(
                            self._cpn_get_processed_data_for_theta,
                            list(seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]),
                        )
                else:
                    with Pool(processes=self.generator_config["n_cpus"] - 1) as pool:
                        out_list += pool.map(
                            self._mlp_get_processed_data_for_theta,
                            list(seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]),
                        )
            else:
                print("No Multiprocessing, since only one cpu requested!")
                if cpn_only:
                    for k in seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]:
                        out_list.append(self._cpn_get_processed_data_for_theta(k))
                else:
                    for k in seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]:
                        out_list.append(self._mlp_get_processed_data_for_theta(k))
        data = {}

        # Choice probabilities and theta are always needed
        data["cpn_data"] = np.concatenate(
            [out_list[k]["cpn_data"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["cpn_labels"] = np.concatenate(
            [out_list[k]["cpn_labels"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["cpn_no_omission_data"] = np.concatenate(
            [out_list[k]["cpn_no_omission_data"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["cpn_no_omission_labels"] = np.concatenate(
            [out_list[k]["cpn_no_omission_labels"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["opn_data"] = np.concatenate(
            [out_list[k]["opn_data"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["opn_labels"] = np.concatenate(
            [out_list[k]["opn_labels"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["gonogo_data"] = np.concatenate(
            [out_list[k]["gonogo_data"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["gonogo_labels"] = np.concatenate(
            [out_list[k]["gonogo_labels"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["thetas"] = np.concatenate(
            [out_list[k]["theta"] for k in range(len(out_list))]
        ).astype(np.float32)

        # Only if not cpn_only, do we need the rest of the data
        # (which is not computed if cpn_only is selected)
        if not cpn_only:
            data["lan_data"] = np.concatenate(
                [out_list[k]["lan_data"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["lan_labels"] = np.concatenate(
                [out_list[k]["lan_labels"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["binned_128"] = np.concatenate(
                [out_list[k]["binned_128"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["binned_256"] = np.concatenate(
                [out_list[k]["binned_256"] for k in range(len(out_list))]
            ).astype(np.float32)

        # Add metadata to training_data
        data["generator_config"] = self.generator_config
        data["model_config"] = self.model_config

        if save:
            output_folder = Path(self.generator_config["output_folder"])
            output_folder.mkdir(parents=True, exist_ok=True)

            full_file_name = (
                output_folder / f"training_data_{uuid.uuid1().hex}"
            ).with_suffix(".pickle")

            print("Writing to file: ", full_file_name)

            pickle.dump(
                data,
                Path(full_file_name).open("wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            return "Dataset completed"

        else:
            return data

    def _training_defective_simulations_get_preprocessed(self, seed):
        np.random.seed(seed)
        rejected_thetas = []
        accepted_thetas = []
        stats_rej = []
        stats_acc = []
        cnt_max = self.generator_config["n_samples"] // 2
        keep = 1
        rej_cnt = 0
        acc_cnt = 0
        while rej_cnt < cnt_max or acc_cnt < cnt_max:
            theta = np.float32(
                np.random.uniform(
                    low=self.model_config["param_bounds"][0],
                    high=self.model_config["param_bounds"][1],
                )
            )
            simulations = self.get_simulations(theta=theta)
            keep, stats = self._filter_simulations(simulations)

            if keep == 0 and rej_cnt < cnt_max:
                print("simulation rejected")
                print("stats: ", stats)
                print("theta", theta)
                rejected_thetas.append(theta)
                stats_rej.append(stats)
                rej_cnt += 1
            elif acc_cnt < cnt_max and keep == 1:
                accepted_thetas.append(theta)
                stats_acc.append(stats)
                acc_cnt += 1
            else:
                pass
        return rejected_thetas

    def _make_save_file_name(self, unique_tag: str = ""):
        binned = str(0)
        if self.generator_config["nbins"] > 0:
            binned = str(1)

        training_data_folder = (
            self.generator_config["output_folder"]
            + unique_tag
            + binned
            + "_nbins_"
            + str(self.generator_config["nbins"])
            + "_n_"
            + str(self.generator_config["n_samples"])
        )
        if not Path(training_data_folder).exists():
            Path(training_data_folder).mkdir(parents=True, exist_ok=True)

        full_file_name = (
            training_data_folder
            + "/"
            + "training_data_"
            + self.model_config["name"]
            + "_"
            + uuid.uuid1().hex
            + ".pickle"
        )
        return full_file_name

    def generate_rejected_parameterizations(self, save: bool = False):
        """Generates parameterizations that are rejected by the filter.

        Arguments
        ---------
            save: bool
                If True, the generated data is saved to disk.

        Returns
        -------
            rejected_parameterization_list: np.array
                Array containing the rejected parameterizations.
        """
        seeds = _generate_random_seeds(
            size=self.generator_config["n_paramseter_sets_rejected"]
        )

        # Get Simulations
        with Pool(processes=self.generator_config["n_cpus"]) as pool:
            rejected_parameterization_list = pool.map(
                self._get_rejected_parameter_setups, seeds
            )
        rejected_parameterization_list = np.concatenate(
            [l_rej for l_rej in rejected_parameterization_list if len(l_rej) > 0]
        )

        if save:
            training_data_folder = (
                self.generator_config["method_folder"]
                + "training_data_binned_"
                + str(int(self.generator_config["binned"]))
                + "_nbins_"
                + str(self.generator_config["nbins"])
                + "_n_"
                + str(self.generator_config["n_samples"])
            )

            if not Path(training_data_folder).exists():
                Path(training_data_folder).mkdir(parents=True)

            full_file_name = (
                training_data_folder
                + "/"
                + "rejected_parameterizations_"
                + self.generator_config["file_id"]
                + ".pickle"
            )

            print("Writing to file: ", full_file_name)

            pickle.dump(
                np.float32(rejected_parameterization_list),
                Path(full_file_name).open("wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            print("Dataset completed")
            return rejected_parameterization_list
        else:
            print("Dataset completed")
            return rejected_parameterization_list
