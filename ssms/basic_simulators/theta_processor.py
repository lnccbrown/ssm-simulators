"""Define the `AbstractThetaProcessor` and its concrete implementation.

`SimpleThetaProcessor` for processing theta parameters based on model configurations.

Classes:
    - AbstractThetaProcessor: An abstract base class that defines the interface for
      processing theta parameters.
    - SimpleThetaProcessor: A concrete implementation of `AbstractThetaProcessor` that
      processes theta parameters based on various model configurations.

The `SimpleThetaProcessor` class includes methods to handle different models such as
single particle models, multi-particle models, LBA-based models, and various choice
models. It modifies the theta parameters according to the specified model configuration
and number of trials.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractThetaProcessor(ABC):
    """Abstract base class for theta processors."""

    @abstractmethod
    def process_theta(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """
        Abstract method to process theta parameters.

        Args:
            theta (Dict[str, Any]): Dictionary of theta parameters.
            model_config (Dict[str, Any]): Dictionary of model configuration.
            n_trials (int): Number of trials.

        Returns
        -------
            Dict[str, Any]: Processed theta parameters.
        """


class SimpleThetaProcessor(AbstractThetaProcessor):
    """Simple implementation of the AbstractThetaProcessor.

    This class collects functions (for now very simple) that build the bridge between
    the model_config level specification of the model and the theta parameters that are
    used in the simulator.
    """

    def process_theta(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """
        Process theta parameters based on the model configuration.

        Args:
            theta (Dict[str, Any]): Dictionary of theta parameters.
            model_config (Dict[str, Any]): Dictionary of model configuration.
            n_trials (int): Number of trials.

        Returns
        -------
            Dict[str, Any]: Processed theta parameters.
        """
        model = model_config["name"]

        if model in [
            "glob",
            "ddm",
            "angle",
            "weibull",
            "ddm_hddm_base",
            "ddm_legacy",  # AF-TODO what was DDM legacy?
            "levy",
            "levy_angle",
            "full_ddm",
            "full_ddm_legacy",
            "full_ddm_hddm_base",
            "ornstein",
            "ornstein_angle",
            "gamma_drift",
            "gamma_drift_angle",
        ]:
            pass

        # ----- Single particle models -----
        if model in ["ds_conflict_drift", "ds_conflict_drift_angle"]:
            theta["v"] = np.tile(np.array([0], dtype=np.float32), n_trials)

        if model in ["ddm_st"]:
            theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
            theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
            # turn st from param values to corresponding random variable
            theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
                theta["st"]
            )

        if model in ["ddm_rayleight"]:
            theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
            theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
            theta["t"] = (
                np.ones(n_trials) * model_config["simulator_fixed_params"]["t"]
            ).astype(np.float32)
            # turn st from param values to corresponding random variable
            theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
                theta["st"]
            )

        if model in ["ddm_truncnormt"]:
            theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
            theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
            # turn st from param values to corresponding random variable
            theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
                theta["mt"], theta["st"]
            )
            theta["t"] = np.array([0], dtype=np.float32)

        if model in ["ddm_sdv"]:
            theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
            theta["t_dist"] = model_config["simulator_fixed_params"]["t_dist"]
            # turn st from param values to corresponding random variable
            theta["v_dist"] = model_config["simulator_param_mappings"]["v_dist"](
                theta["sv"]
            )

        if model in ["full_ddm_rv"]:
            theta["z_dist"] = model_config["simulator_param_mappings"]["z_dist"](
                theta["sz"]
            )
            theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
                theta["st"]
            )
            theta["v_dist"] = model_config["simulator_param_mappings"]["v_dist"](
                theta["sv"]
            )

        if model in [
            "shrink_spot",
            "shrink_spot_simple",
            "shrink_spot_extended",
            "shrink_spot_extended_angle",
            "shrink_spot_simple_extended",
        ]:
            theta["v"] = np.tile(np.array([0], dtype=np.float32), n_trials)

        # Multi-particle models
        #   LBA-based models
        if model == "lba2":
            theta["nact"] = 2
            theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
            theta["z"] = np.expand_dims(theta["A"], axis=1)
            theta["a"] = np.expand_dims(theta["b"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

            del theta["A"]
            del theta["b"]

        if model == "lba3":
            theta["nact"] = 3
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])

            theta["z"] = np.expand_dims(theta["A"], axis=1)
            theta["a"] = np.expand_dims(theta["b"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

            del theta["A"]
            del theta["b"]

        if model == "lba_3_vs_constraint":
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["z"] = np.expand_dims(theta["z"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

        if model in ["lba_angle_3_vs_constraint", "lba_angle_3"]:
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["z"] = np.expand_dims(theta["z"], axis=1)
            theta["theta"] = np.expand_dims(theta["theta"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)
        
        if model in ["dev_lba_angle_3_v2"]:
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            # theta["a"] = np.expand_dims(theta["a"], axis=1)
            # theta["z"] = np.expand_dims(theta["z"], axis=1)
            # theta["theta"] = np.expand_dims(theta["theta"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

        if model in ["dev_rlwm_lba_race_v1", "dev_rlwm_lba_race_v2"]:
            theta["v_RL"] = np.column_stack(
                [theta["v_RL_0"], theta["v_RL_1"], theta["v_RL_2"]]
            )
            theta["v_WM"] = np.column_stack(
                [theta["v_WM_0"], theta["v_WM_1"], theta["v_WM_2"]]
            )
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["z"] = np.expand_dims(theta["z"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

        if model == "dev_rlwm_lba_pw_v1":
            theta["v_RL"] = np.column_stack(
                [theta["v_RL_0"], theta["v_RL_1"], theta["v_RL_2"]]
            )
            theta["v_WM"] = np.column_stack(
                [theta["v_WM_0"], theta["v_WM_1"], theta["v_WM_2"]]
            )
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["z"] = np.expand_dims(theta["z"], axis=1)
            theta["t_WM"] = np.expand_dims(theta["t_WM"], axis=1)
            theta["t"] = np.zeros(n_trials).astype(np.float32)

        # 2 Choice
        if model == "race_2":
            theta["z"] = np.column_stack([theta["z0"], theta["z1"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_bias_2", "race_no_bias_angle_2"]:
            theta["z"] = np.column_stack([theta["z"], theta["z"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_z_2", "race_no_z_angle_2"]:
            theta["z"] = np.tile(np.array([0.0] * 2, dtype=np.float32), (n_trials, 1))
            theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        # 3 Choice models

        if model == "race_3":
            theta["z"] = np.column_stack([theta["z0"], theta["z1"], theta["z2"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_bias_3", "race_no_bias_angle_3"]:
            theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_z_3", "race_no_z_angle_3"]:
            theta["z"] = np.tile(np.array([0.0] * 3, dtype=np.float32), (n_trials, 1))
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model == "lca_3":
            theta["z"] = np.column_stack([theta["z0"], theta["z1"], theta["z2"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        if model in ["lca_no_bias_3", "lca_no_bias_angle_3"]:
            theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"]])
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        if model in ["lca_no_z_3", "lca_no_z_angle_3"]:
            theta["z"] = np.tile(np.array([0.0] * 3, dtype=np.float32), (n_trials, 1))
            theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        # 4 Choice models
        if model == "race_4":
            theta["z"] = np.column_stack(
                [theta["z0"], theta["z1"], theta["z2"], theta["z3"]]
            )
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_bias_4", "race_no_bias_angle_4"]:
            theta["z"] = np.column_stack(
                [theta["z"], theta["z"], theta["z"], theta["z"]]
            )
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model in ["race_no_z_4", "race_no_z_angle_4"]:
            theta["z"] = np.tile(np.array([0.0] * 4, dtype=np.float32), (n_trials, 1))
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)

        if model == "lca_4":
            theta["z"] = np.column_stack(
                [theta["z0"], theta["z1"], theta["z2"], theta["z3"]]
            )
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        if model in ["lca_no_bias_4", "lca_no_bias_angle_4"]:
            theta["z"] = np.column_stack(
                [theta["z"], theta["z"], theta["z"], theta["z"]]
            )
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        if model in ["lca_no_z_4", "lca_no_z_angle_4"]:
            theta["z"] = np.tile(np.array([0.0] * 4, dtype=np.float32), (n_trials, 1))
            theta["v"] = np.column_stack(
                [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
            )
            theta["t"] = np.expand_dims(theta["t"], axis=1)
            theta["a"] = np.expand_dims(theta["a"], axis=1)
            theta["g"] = np.expand_dims(theta["g"], axis=1)
            theta["b"] = np.expand_dims(theta["b"], axis=1)

        # Seq / Parallel models (4 choice)
        z_vec = np.tile(np.array([0.5], dtype=np.float32), reps=n_trials)
        g_zero_vec = np.tile(np.array([0.0], dtype=np.float32), reps=n_trials)
        g_vec_leak = np.tile(np.array([2.0], dtype=np.float32), reps=n_trials)
        s_pre_high_level_choice_zero_vec = np.tile(
            np.array([0.0], dtype=np.float32), reps=n_trials
        )
        s_pre_high_level_choice_one_vec = np.tile(
            np.array([1.0], dtype=np.float32), reps=n_trials
        )

        # if model in ["ddm_seq2", "ddm_seq2_traj"]:
        #     sim_param_dict["s"] = noise_dict["1_particles"]

        if model in [
            "ddm_seq2_no_bias",
            "ddm_seq2_angle_no_bias",
            "ddm_seq2_weibull_no_bias",
            "ddm_seq2_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

        # if model == "ddm_par2":
        #     sim_param_dict["s"] = noise_dict["1_particles"]

        if model in [
            "ddm_par2_no_bias",
            "ddm_par2_angle_no_bias",
            "ddm_par2_weibull_no_bias",
            "ddm_par2_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

        if model == "ddm_mic2_adj":
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec
            theta["g"] = g_zero_vec

        if model in [
            "ddm_mic2_adj_no_bias",
            "ddm_mic2_adj_angle_no_bias",
            "ddm_mic2_adj_weibull_no_bias",
            "ddm_mic2_adj_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
            theta["g"] = g_zero_vec
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

        # ----- Ornstein version of mic2_adj ---------
        if model == "ddm_mic2_ornstein":
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

        if model in [
            "ddm_mic2_ornstein_no_bias",
            "ddm_mic2_ornstein_angle_no_bias",
            "ddm_mic2_ornstein_weibull_no_bias",
            "ddm_mic2_ornstein_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

        if model in [
            "ddm_mic2_ornstein_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_zero_vec

        # Leak version of mic2
        if model == "ddm_mic2_leak":
            theta["g"] = g_vec_leak
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

        if model in [
            "ddm_mic2_leak_no_bias",
            "ddm_mic2_leak_angle_no_bias",
            "ddm_mic2_leak_weibull_no_bias",
            "ddm_mic2_leak_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
            theta["g"] = g_vec_leak
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

        if model in [
            "ddm_mic2_leak_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_angle_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
            theta["g"] = g_vec_leak
            theta["s_pre_high_level_choice"] = s_pre_high_level_choice_zero_vec

        # ----------------- High level dependent noise scaling --------------
        if model in [
            "ddm_mic2_multinoise_no_bias",
            "ddm_mic2_multinoise_angle_no_bias",
            "ddm_mic2_multinoise_weibull_no_bias",
            "ddm_mic2_multinoise_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

        # ----------------- Tradeoff models -----------------
        if model in [
            "tradeoff_no_bias",
            "tradeoff_angle_no_bias",
            "tradeoff_weibull_no_bias",
            "tradeoff_conflict_gamma_no_bias",
        ]:
            theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        return theta
