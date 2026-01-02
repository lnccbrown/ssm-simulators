"""Simulation-based data generation strategy for KDE estimators."""

import numpy as np
from copy import deepcopy
from typing import Any
from scipy.stats import mode

from ssms.basic_simulators.simulator import _theta_dict_to_array
from ssms.basic_simulators.simulator_class import Simulator
from ssms.dataset_generators.protocols import (
    EstimatorBuilderProtocol,
    TrainingDataStrategyProtocol,
    ParameterSamplerProtocol,
)


class SimulationBasedGenerationStrategy:
    """Generate training data via simulation + KDE workflow.

    This strategy implements the traditional workflow:
    1. Sample parameters from parameter space
    2. Run Cython simulator to generate samples
    3. Filter simulations for quality (RT variance, choice proportions, etc.)
    4. Build KDE estimator from simulations
    5. Generate training data from KDE

    Used for: KDE-based likelihood estimation

    This strategy is necessary when analytical solutions are not available
    or when you want to validate analytical methods against simulations.
    """

    def __init__(
        self,
        generator_config: dict,
        model_config: dict,
        estimator_builder: EstimatorBuilderProtocol,
        training_strategy: TrainingDataStrategyProtocol,
    ):
        """Initialize simulation-based strategy.

        Args:
            generator_config: Configuration for data generation
            model_config: Model specification (must contain 'param_bounds_dict')
            estimator_builder: Builder for KDE estimator
            training_strategy: Strategy for generating training samples
        """
        self.generator_config = generator_config
        self.model_config = model_config
        self.estimator_builder = estimator_builder
        self.training_strategy = training_strategy

        # Construct simulator from model_config (single source of truth)
        # The same model_config is used for both simulation-based and analytical strategies
        self.simulator = Simulator(model_config["name"])

        # Create parameter sampler with model-specific transforms
        self._param_sampler = self._create_parameter_sampler()

    def generate_for_parameter_set(
        self,
        parameter_sampling_seed: int | None = None,
        simulator_seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate training data for one parameter set (simulation workflow).

        Workflow:
        1. Sample theta from parameter space
        2. Run simulations
        3. Filter simulations
        4. Build estimator
        5. Generate training data

        Args:
            parameter_sampling_seed: Seed for parameter sampling (ensures different workers sample different θ)
            simulator_seed: Random seed for simulations (controls RT/choice variability)

        Returns:
            Dictionary with keys:
                - 'data': Training data dictionary (or None if generation failed)
                - 'theta': Parameter values used
                - 'success': Whether generation succeeded
        """
        # Create isolated RNG for parameter sampling (no global state pollution!)
        param_rng = np.random.default_rng(parameter_sampling_seed)

        # Keep simulating until we get valid data
        keep = False
        max_attempts = 1000  # High threshold - parameter resampling is cheap
        attempt = 0

        while not keep and attempt < max_attempts:
            # 1. Sample parameters (with transforms applied automatically)
            # Pass RNG directly to sample() - clean and functional!
            theta_dict = self._param_sampler.sample(n_samples=1, rng=param_rng)

            # 2. Run simulations
            simulations = self._run_simulations(theta_dict, simulator_seed)

            # 3. Filter simulations
            keep, stats = self._is_valid_simulation(simulations)
            attempt += 1

        if not keep:
            return {"data": None, "theta": theta_dict, "success": False}

        # 4. Build estimator + generate training data
        try:
            estimator = self.estimator_builder.build(theta_dict, simulations)
            training_data = self.training_strategy.generate(theta_dict, estimator)
        except Exception as e:
            return {
                "data": None,
                "theta": theta_dict,
                "success": False,
                "error": str(e),
            }

        # 5. Prepare output dictionary
        theta_array = _theta_dict_to_array(theta_dict, self.model_config["params"])

        # Extract CPN and other labels
        if len(simulations["metadata"]["possible_choices"]) == 2:
            # For 2-choice models, extract probability of choice 1 and reshape to (1, 1)
            cpn_labels = simulations["choice_p"][0, 1:2].reshape(1, 1)
            cpn_no_omission_labels = simulations["choice_p_no_omission"][
                0, 1:2
            ].reshape(1, 1)
        else:
            cpn_labels = simulations["choice_p"]
            cpn_no_omission_labels = simulations["choice_p_no_omission"]

        result = {
            "lan_data": training_data[:, :-1],
            "lan_labels": training_data[:, -1],
            "cpn_data": theta_array,
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

        return {"data": result, "theta": theta_dict, "success": True}

    def _run_simulations(self, theta: dict, seed: int | None) -> dict:
        """Run class-based simulator.

        Args:
            theta: Parameter dictionary
            seed: Random seed for simulator

        Returns:
            Dictionary containing core simulation results (rts, choices, metadata)
            with computed auxiliary labels
        """
        # Run simulator (returns core output: rts, choices, metadata)
        simulations = self.simulator.simulate(
            theta=deepcopy(theta),
            n_samples=self.generator_config["n_samples"],
            delta_t=self.generator_config["delta_t"],
            max_t=self.generator_config["max_t"],
            random_state=seed,
        )

        # Compute auxiliary labels for training data
        auxiliary_labels = self._compute_auxiliary_labels(simulations)

        # Merge auxiliary labels into simulation results
        simulations.update(auxiliary_labels)

        return simulations

    def _is_valid_simulation(self, simulations: dict) -> tuple[bool, np.ndarray]:
        """Validate simulation quality.

        Checks:
        - RT variance (not too concentrated at mode)
        - Choice proportions (enough samples per choice)
        - Statistical properties (mean RT, std, etc.)

        Args:
            simulations: Dictionary containing simulation results

        Returns:
            Tuple of (keep: bool, stats: np.ndarray)
            - keep: Whether simulation passes all filters
            - stats: Array of statistics [mode, mean, std, mode_cnt_rel, tmp_n_c, n_sim]
        """
        if simulations is None:
            raise ValueError("No simulations provided")

        keep = True
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

            # Apply all filters
            # AF-TODO: Apply .get() pattern here to default to "no filter" on the
            # key if it isn't present.
            keep = (
                keep
                & (mode_ <= self.generator_config["simulation_filters"]["mode"])
                & (mean_ <= self.generator_config["simulation_filters"]["mean_rt"])
                & (std_ >= self.generator_config["simulation_filters"]["std"])
                & (
                    mode_cnt_rel_
                    <= self.generator_config["simulation_filters"]["mode_cnt_rel"]
                )
                & (tmp_n_c >= self.generator_config["simulation_filters"]["choice_cnt"])
            )

        stats = np.array(
            [mode_, mean_, std_, mode_cnt_rel_, tmp_n_c, n_sim], dtype=np.float32
        )

        return keep, stats

    def _compute_auxiliary_labels(self, simulations: dict) -> dict:
        """Compute auxiliary training labels from simulation results.

        These labels are used for multi-task learning networks (CPN, OPN, etc.).
        All labels are computed from the core simulation output (rts, choices).

        Args:
            simulations: Dictionary with 'rts', 'choices', 'metadata' keys

        Returns:
            Dictionary with auxiliary label keys:
                - choice_p: Choice probabilities
                - choice_p_no_omission: Choice probabilities excluding omissions
                - omission_p: Omission probability
                - nogo_p: No-go probability
                - go_p: Go probability
                - binned_128: RT histogram with 128 bins
                - binned_256: RT histogram with 256 bins
        """
        metadata = simulations["metadata"]
        rts = simulations["rts"]
        choices = simulations["choices"]
        possible_choices = metadata["possible_choices"]
        n_trials = 1  # Single trial per parameter set

        # Initialize arrays
        choice_p = np.zeros((n_trials, len(possible_choices)))
        choice_p_no_omission = np.zeros((n_trials, len(possible_choices)))
        omission_p = np.zeros((n_trials, 1))
        nogo_p = np.zeros((n_trials, 1))
        go_p = np.zeros((n_trials, 1))

        # Compute choice probabilities
        n_samples = len(choices)
        for i, choice in enumerate(possible_choices):
            choice_p[0, i] = np.sum(choices == choice) / n_samples

        # Compute choice probabilities excluding omissions
        non_omitted = choices != -999
        if np.any(non_omitted):
            choices_no_omission = choices[non_omitted]
            n_no_omission = len(choices_no_omission)
            for i, choice in enumerate(possible_choices):
                choice_p_no_omission[0, i] = (
                    np.sum(choices_no_omission == choice) / n_no_omission
                )
        else:
            # All omitted - uniform distribution
            choice_p_no_omission[0, :] = 1.0 / len(possible_choices)

        # Compute omission probability
        omission_p[0, 0] = np.sum(choices == -999) / n_samples

        # Compute go/nogo probabilities
        # nogo = not choosing max choice OR omission
        max_choice = max(possible_choices)
        nogo_p[0, 0] = np.sum((choices != max_choice) | (rts == -999)) / n_samples
        go_p[0, 0] = 1 - nogo_p[0, 0]

        # Compute RT histograms (separated by choice)
        max_t = metadata.get("max_t", 20.0)
        bins_128 = np.linspace(0, max_t, 129)  # 129 edges for 128 bins
        bins_256 = np.linspace(0, max_t, 257)  # 257 edges for 256 bins

        # Initialize histograms for each choice
        binned_128 = np.zeros((n_trials, 128, len(possible_choices)))
        binned_256 = np.zeros((n_trials, 256, len(possible_choices)))

        # Compute histogram for each choice separately
        for i, choice in enumerate(possible_choices):
            choice_mask = choices == choice
            rts_for_choice = rts[choice_mask]
            # Exclude omissions (RT == -999)
            rts_for_choice = rts_for_choice[rts_for_choice != -999]

            if len(rts_for_choice) > 0:
                binned_128[0, :, i], _ = np.histogram(rts_for_choice, bins=bins_128)
                binned_256[0, :, i], _ = np.histogram(rts_for_choice, bins=bins_256)

        return {
            "choice_p": choice_p,
            "choice_p_no_omission": choice_p_no_omission,
            "omission_p": omission_p,
            "nogo_p": nogo_p,
            "go_p": go_p,
            "binned_128": binned_128,
            "binned_256": binned_256,
        }

    def _create_parameter_sampler(self) -> ParameterSamplerProtocol:
        """Create parameter sampler with model-specific transforms.

        Extracts transforms from model config and creates a sampler that
        automatically applies them during sampling.

        Returns:
            Parameter sampler configured with model transforms
        """
        from ssms.dataset_generators.parameter_samplers import UniformParameterSampler
        from ssms.dataset_generators.parameter_samplers.constraints.factory import (
            get_constraints_from_model_config,
        )

        # Extract constraints from model config (empty list if none defined)
        constraints = get_constraints_from_model_config(self.model_config)

        return UniformParameterSampler(
            param_space=self.model_config["param_bounds_dict"],
            constraints=constraints,
        )

    def get_param_space(self) -> Any:
        """Get the parameter space dictionary."""
        return self._param_sampler.get_param_space()
