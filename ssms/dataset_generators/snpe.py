from ssms.basic_simulators.simulator import simulator
import numpy as np
import dill
import uuid
import os
from multiprocessing import Pool
from ssms.dataset_generators.lan_mlp import data_generator
from functools import partial

"""
    This module defines a data generator class for SNPE.

"""


class data_generator_snpe(data_generator):
    """

    Class for generating data for SNPE.

    Attributes
    ----------
    generator_config: dict
        Configuration for data generation
    model_config: dict
        Configuration for model

    Methods
    -------
    generate_data_training_uniform(save=False)
        Generates data for training SNPE.
    _snpe_get_processed_data_for_theta(random_seed)
        Helper function for generating data for SNPE.
    _build_simulator()
        Builds simulator function for SNPE.

    """

    def __init__(self, generator_config=None, model_config=None):
        super().__init__(generator_config=generator_config, model_config=model_config)

    def generate_data_training_uniform(self, save=False):
        seeds = np.random.choice(
            400000000, size=self.generator_config["n_parameter_sets"]
        )
        seed_args = [
            [seeds[i], i + 1]
            for i in np.arange(0, self.generator_config["n_parameter_sets"], 1)
        ]

        # Inits
        subrun_n = (
            self.generator_config["n_parameter_sets"]
            // self.generator_config["n_subruns"]
        )

        # Get Simulations
        data = {}
        for i in range(self.generator_config["n_subruns"]):
            print("simulation round:", i + 1, " of", self.generator_config["n_subruns"])
            cum_i = int(i * subrun_n)
            with Pool(processes=self.generator_config["n_cpus"] - 1) as pool:
                data_tmp = pool.map(
                    self._snpe_get_processed_data_for_theta,
                    [k for k in seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]],
                )
                data = {
                    **data,
                    **{
                        i + cum_i: {
                            "data": data_tmp[i]["features"],
                            "labels": data_tmp[i]["labels"],
                        }
                        for i in range(len(data_tmp))
                    },
                }

        if save:
            training_data_folder = (
                self.generator_config["output_folder"]
                + "training_data_"
                + "n_"
                + str(self.generator_config["n_samples"])
                + "/"
                + self.model_config["name"]
            )

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = (
                training_data_folder
                + "/"
                + "training_data_"
                + self.model_config["name"]
                + "_"
                + uuid.uuid1().hex
                + ".dill"
            )

            print("Writing to file: ", full_file_name)

            dill.dump(
                data,
                open(full_file_name, "wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            return "Dataset completed"

        else:
            return data

    def _snpe_get_processed_data_for_theta(self, random_seed):
        np.random.seed(random_seed[0])

        theta = np.float32(
            np.random.uniform(
                low=self.model_config["param_bounds"][0],
                high=self.model_config["param_bounds"][1],
            )
        )
        simulations = self.get_simulations(theta=theta)

        return {
            "features": np.hstack([simulations["rts"], simulations["choices"]]),
            "labels": theta,
            "meta": simulations["metadata"],
        }

    def _build_simulator(self):
        self.simulator = partial(
            simulator,
            n_samples=self.generator_config["n_samples"],
            max_t=self.generator_config["max_t"],
            bin_dim=0,
            delta_t=self.generator_config["delta_t"],
        )
