"""PyDDM-based data generation strategy using analytical solutions."""

import numpy as np
from typing import Any

from ssms.basic_simulators.simulator import _theta_dict_to_array
from ssms.dataset_generators.protocols import (
    EstimatorBuilderProtocol,
    TrainingDataStrategyProtocol,
    ParameterSamplerProtocol,
)


class PyDDMGenerationStrategy:
    """Generate training data via PyDDM analytical PDF workflow.

    This strategy implements an analytical workflow:
    1. Sample parameters from parameter space
    2. SKIP simulation (use analytical Fokker-Planck solution)
    3. Build PyDDM estimator (solves PDE)
    4. Compute auxiliary labels analytically
    5. Generate training data from analytical PDF

    Used for: PyDDM-based likelihood estimation

    Benefits:
    - Much faster (no simulation overhead)
    - No filtering needed (analytical solution always valid)
    - Deterministic for given parameters
    - Memory efficient (no large simulation arrays)
    - Computes choice probabilities, omission_p, and nogo_p analytically

    Limitations:
    - Only works for compatible models (single-particle, two-choice, Gaussian noise)
    - Cannot generate binned RT histograms (these require trajectory simulations)
    """

    def __init__(
        self,
        generator_config: dict,
        model_config: dict,
        estimator_builder: EstimatorBuilderProtocol,
        training_strategy: TrainingDataStrategyProtocol,
    ):
        """Initialize PyDDM-based strategy.

        Args:
            generator_config: Configuration for data generation
            model_config: Model specification (must contain 'param_bounds_dict')
            estimator_builder: Builder for PyDDM estimator
            training_strategy: Strategy for generating training samples
        """
        self.generator_config = generator_config
        self.model_config = model_config
        self.estimator_builder = estimator_builder
        self.training_strategy = training_strategy

        # Create parameter sampler with model-specific transforms
        self._param_sampler = self._create_parameter_sampler()

    def generate_for_parameter_set(
        self,
        parameter_sampling_seed: int,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate training data for one parameter set (analytical workflow).

        Workflow:
        1. Sample theta from parameter space
        2. Build analytical estimator (no simulations needed)
        3. Compute auxiliary labels analytically
        4. Generate training data

        Args:
            parameter_sampling_seed: Index for parameter sampling (used as seed)
            random_seed: Random seed (accepted for API compatibility,
                may not be used as analytical solutions are deterministic)

        Returns:
            Dictionary with keys:
                - 'data': Training data dictionary (or None if generation failed)
                - 'theta': Parameter values used
                - 'success': Whether generation succeeded
                - 'error': Error message if success=False (optional)

        Note:
            PyDDM can compute choice probabilities, omission_p, and nogo_p
            analytically without simulations. Only binned RT histograms require
            trajectory data and are set to None.
        """
        # Use parameter_sampling_seed as random seed for parameter sampling
        np.random.seed(parameter_sampling_seed)

        # Keep trying until we get valid parameters
        # (PyDDM may reject parameters with high P(undecided))
        max_attempts = 1000  # High threshold - parameter resampling is cheap
        attempt = 0
        success = False

        while not success and attempt < max_attempts:
            # 1. Sample parameters (with transforms applied automatically)
            theta_dict = self._param_sampler.sample(n_samples=1)

            # 2. Build analytical estimator (simulations=None)
            # PyDDMEstimatorBuilder may raise ValueError if P(undecided) too high
            try:
                estimator = self.estimator_builder.build(theta_dict, simulations=None)
                success = True
            except ValueError as e:
                # Parameter set rejected (e.g., high P(undecided))
                attempt += 1
                if attempt >= max_attempts:
                    return {
                        "data": None,
                        "theta": theta_dict,
                        "success": False,
                        "error": (
                            f"Failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        ),
                    }
                continue

        # 3. Compute auxiliary labels from PyDDM solution
        auxiliary_labels = self._compute_auxiliary_labels(estimator, theta_dict)

        # 4. Generate training data
        try:
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

        # PyDDM strategy generates LAN data + all auxiliary labels analytically
        # Only binned histograms are None (require trajectory data)
        # For 2-choice models, extract probability of choice 1 (index 1) for consistency
        result = {
            "lan_data": training_data[:, :-1],
            "lan_labels": training_data[:, -1],
            "theta": theta_array,
            # Auxiliary labels computed analytically from PyDDM solution
            "cpn_data": theta_array,
            "cpn_labels": auxiliary_labels["choice_p"][
                :, 1:2
            ],  # Extract P(choice=1), shape (1,1)
            "cpn_no_omission_data": theta_array,
            "cpn_no_omission_labels": auxiliary_labels["choice_p_no_omission"][
                :, 1:2
            ],  # Shape (1,1)
            "opn_data": theta_array,
            "opn_labels": auxiliary_labels["omission_p"],
            "gonogo_data": theta_array,
            "gonogo_labels": auxiliary_labels["nogo_p"],
            # Binned histograms require trajectories (None for PyDDM)
            "binned_128": None,
            "binned_256": None,
        }

        return {"data": result, "theta": theta_dict, "success": True}

    def _compute_auxiliary_labels(
        self, estimator, theta_dict: dict
    ) -> dict[str, np.ndarray]:
        """Compute auxiliary labels from PyDDM solution.

        Computes choice probabilities, omission probabilities, and go/nogo
        probabilities analytically using PyDDM's solution object and numerical
        integration. This follows the same logic as the simulation-based strategy
        but computed analytically.

        Args:
            estimator: PyDDMLikelihoodEstimator with solved PyDDM model
            theta_dict: Parameter dictionary (used to extract deadline if present)

        Returns:
            Dictionary with:
                - choice_p: Choice probabilities [error, correct], shape (1, 2)
                - choice_p_no_omission: Choice probs excluding omissions, shape (1, 2)
                - omission_p: Probability of RT beyond deadline, shape (1, 1)
                - nogo_p: Probability of error OR omitting correct, shape (1, 1)
        """
        from scipy import integrate

        # Get PyDDM solution and time domain
        solution = estimator.solution
        t_dom = estimator.t_domain

        # Get PDFs for both choices
        sol_corr = estimator.pdf_correct
        sol_err = estimator.pdf_error

        # Initialize output arrays (n_trials=1 for single parameter set)
        n_trials = 1
        choice_p = np.zeros((n_trials, 2), dtype=np.float32)
        choice_p_pre_deadline = np.zeros((n_trials, 2), dtype=np.float32)
        choice_p_post_deadline = np.zeros((n_trials, 2), dtype=np.float32)
        choice_p_no_omission = np.zeros((n_trials, 2), dtype=np.float32)

        # Get deadline (default to beyond max_t if not specified)
        deadline = theta_dict.get("deadline", t_dom[-1] + 1)

        if deadline >= t_dom[-1]:
            # No deadline or deadline beyond domain: all responses are pre-deadline
            choice_p[0, 0] = solution.prob("error")
            choice_p[0, 1] = solution.prob("correct")

            choice_p_no_omission[0, 0] = choice_p[0, 0]
            choice_p_no_omission[0, 1] = choice_p[0, 1]

            nogo_p = choice_p[0, 0]
            _go_p = choice_p[0, 1]  # Calculated for potential future use

            omission_p = 0.0
        else:
            # Deadline is within domain: split into pre/post deadline

            # Pre-deadline probabilities (integrate PDFs before deadline)
            choice_p_pre_deadline[0, 0] = integrate.simpson(
                sol_err[t_dom < deadline], t_dom[t_dom < deadline]
            )
            choice_p_pre_deadline[0, 1] = integrate.simpson(
                sol_corr[t_dom < deadline], t_dom[t_dom < deadline]
            )

            # Post-deadline probabilities (remainder)
            choice_p_post_deadline[0, 0] = (
                solution.prob("error") - choice_p_pre_deadline[0, 0]
            )
            choice_p_post_deadline[0, 1] = (
                solution.prob("correct") - choice_p_pre_deadline[0, 1]
            )

            # Overall choice probabilities (pre-deadline only)
            choice_p[0, 0] = choice_p_pre_deadline[0, 0]
            choice_p[0, 1] = choice_p_pre_deadline[0, 1]

            # Choice probabilities excluding omissions (renormalize pre-deadline)
            total_pre_deadline = (
                choice_p_pre_deadline[0, 0] + choice_p_pre_deadline[0, 1]
            )
            if total_pre_deadline > 0:
                choice_p_no_omission[0, 0] = (
                    choice_p_pre_deadline[0, 0] / total_pre_deadline
                )
                choice_p_no_omission[0, 1] = (
                    choice_p_pre_deadline[0, 1] / total_pre_deadline
                )
            else:
                # All omitted - uniform distribution
                choice_p_no_omission[0, 0] = 0.5
                choice_p_no_omission[0, 1] = 0.5

            # Nogo probability: error boundary OR post-deadline correct
            # (matching simulation logic: not choosing max choice OR omission)
            nogo_p = choice_p_post_deadline[0, 1] + choice_p[0, 0]
            _go_p = 1 - nogo_p  # Calculated for potential future use

            # Omission probability: any response beyond deadline
            omission_p = choice_p_post_deadline[0, 0] + choice_p_post_deadline[0, 1]

        # Format outputs to match simulation-based strategy
        return {
            "choice_p": choice_p,
            "choice_p_no_omission": choice_p_no_omission,
            "omission_p": np.array([[omission_p]], dtype=np.float32),
            "nogo_p": np.array([[nogo_p]], dtype=np.float32),
        }

    def _create_parameter_sampler(self) -> ParameterSamplerProtocol:
        """Create parameter sampler with model-specific transforms.

        Extracts transforms from model config and creates a sampler that
        automatically applies them during sampling.

        Returns:
            Parameter sampler configured with model transforms
        """
        from ssms.dataset_generators.parameter_samplers import UniformParameterSampler
        from ssms.dataset_generators.parameter_samplers.transforms.factory import (
            get_transforms_from_model_config,
        )

        # Extract transforms from model config (empty list if none defined)
        transforms = get_transforms_from_model_config(self.model_config)

        return UniformParameterSampler(
            param_space=self.model_config["param_bounds_dict"],
            transforms=transforms,
        )

    def get_param_space(self) -> Any:
        """Get the parameter space dictionary."""
        return self._param_sampler.get_param_space()
