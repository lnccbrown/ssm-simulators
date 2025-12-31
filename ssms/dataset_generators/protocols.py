"""Protocols for data generation components.

This module defines the interfaces (protocols) for:
- Likelihood estimators: Objects that estimate P(RT, choice | theta)
- Estimator builders: Objects that create and configure likelihood estimators
- Training data strategies: Objects that generate training samples from estimators
- Data generation strategies: Objects that orchestrate the end-to-end workflow
- Parameter samplers: Objects that sample parameters from parameter spaces

Using protocols allows for dependency injection and clean separation of concerns.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


class ParameterSamplerProtocol(Protocol):
    """Protocol for parameter sampling strategies.

    Parameter samplers generate parameter sets from a parameter space,
    handling dependencies between parameters and applying transforms.
    """

    def sample(
        self, n_samples: int = 1, random_state: np.random.Generator | None = None
    ) -> dict[str, np.ndarray]:
        """Sample n_samples parameter sets.

        Args:
            n_samples: Number of parameter sets to sample

        Returns:
            Dictionary mapping parameter names to sampled values (arrays)
        """
        ...

    def get_param_space(self) -> dict[str, tuple]:
        """Get the parameter space bounds.

        Returns:
            Dictionary mapping parameter names to (lower, upper) bound tuples
        """
        ...


class LikelihoodEstimatorProtocol(Protocol):
    """Protocol for likelihood estimation.

    Likelihood estimators compute P(RT, choice | theta) and can sample
    from this distribution. They may use different methods:
    - KDE: Build kernel density estimate from simulated samples
    - Analytical: Use closed-form PDF (e.g., from PyDDM solver)
    - Other: Custom methods
    """

    def fit(self, simulations: dict[str, Any]) -> None:
        """Fit estimator to simulation data.

        For KDE: Builds the kernel density estimate.
        For analytical: May be a no-op (PDF already available).

        Arguments
        ---------
        simulations : dict
            Dictionary containing simulation data with keys:
            - 'rts': np.ndarray of reaction times
            - 'choices': np.ndarray of choices
            - 'metadata': dict with model info
        """
        ...

    def evaluate(self, rts: np.ndarray, choices: np.ndarray) -> np.ndarray:
        """Evaluate log-likelihood at given (RT, choice) pairs.

        Arguments
        ---------
        rts : np.ndarray
            Reaction times to evaluate, shape (n_samples,)
        choices : np.ndarray
            Choices to evaluate, shape (n_samples,)

        Returns
        -------
        log_likelihoods : np.ndarray
            Log-likelihood for each (RT, choice) pair, shape (n_samples,)
        """
        ...

    def sample(
        self, n_samples: int, random_state: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample (RT, choice) pairs from the estimated likelihood.

        Arguments
        ---------
        n_samples : int
            Number of samples to generate
        random_state : int | None, optional
            Random seed for reproducibility. If None, uses non-reproducible random behavior.

        Returns
        -------
        samples : dict
            Dictionary with keys:
            - 'rts': np.ndarray of sampled reaction times
            - 'choices': np.ndarray of sampled choices
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata about the estimator.

        Returns
        -------
        metadata : dict
            Dictionary containing:
            - 'max_t': Maximum time value
            - 'possible_choices': List of possible choice values
            - Other model-specific metadata
        """
        ...


class EstimatorBuilderProtocol(Protocol):
    """Protocol for building likelihood estimators.

    Builders encapsulate all the logic for creating and configuring
    likelihood estimators. This keeps the TrainingDataGenerator class generic
    and free of type-specific logic.
    """

    def build(
        self, theta: dict[str, Any], simulations: dict[str, Any] | None = None
    ) -> LikelihoodEstimatorProtocol:
        """Build and return a fitted likelihood estimator for given theta.

        Arguments
        ---------
        theta : dict
            Model parameters (e.g., {'v': 1.0, 'a': 2.0, 'z': 0.5, 't': 0.3})
        simulations : dict | None
            Simulation data (required for KDE, may be None for analytical estimators)

        Returns
        -------
        estimator : LikelihoodEstimatorProtocol
            A fitted likelihood estimator ready for use

        Raises
        ------
        ValueError
            If required data (e.g., simulations for KDE) is missing
        """
        ...


class TrainingDataStrategyProtocol(Protocol):
    """Protocol for training data generation strategies.

    Strategies define how to generate training samples from a likelihood
    estimator. Different strategies might:
    - Resample from estimator + add uniform samples (current approach)
    - Evaluate on a uniform grid
    - Use importance sampling
    - Other methods
    """

    def generate(
        self,
        theta: dict[str, Any],
        likelihood_estimator: LikelihoodEstimatorProtocol,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate training data array.

        Arguments
        ---------
        theta : dict
            Model parameters
        likelihood_estimator : LikelihoodEstimatorProtocol
            Fitted likelihood estimator
        random_state : int | None, optional
            Random seed for reproducibility. If None, uses non-reproducible random behavior.
            Controls all random operations including sampling and row shuffling.

        Returns
        -------
        training_data : np.ndarray
            Array of shape (n_samples, n_params + 3) containing:
            - Columns 0:n_params: Parameter values (tiled for all samples)
            - Column -3: Reaction times
            - Column -2: Choices
            - Column -1: Log-likelihoods

            Note: Rows are shuffled to avoid ordering bias in training batches.
        """
        ...


@runtime_checkable
class DataGenerationPipelineProtocol(Protocol):
    """Protocol for end-to-end data generation workflows.

    Data generation pipelines orchestrate the complete process from parameter
    sampling to training data generation. Different pipelines handle different
    workflows:
    - Simulation-based: Run simulations, filter, build KDE, generate data
    - Analytical (PyDDM): Skip simulations, solve Fokker-Planck, generate data
    - Hybrid: Mix simulation and analytical approaches
    - Other: Custom workflows

    This protocol enables clean separation between simulation-dependent logic
    (SimulationPipeline) and analytical approaches (PyDDMPipeline),
    eliminating wasteful computation.

    Attributes
    ----------
    generator_config : dict
        Configuration for data generation (samples, bins, etc.)
    model_config : dict
        Model specification (parameters, bounds, transforms, etc.)
    """

    # Required attributes
    generator_config: dict
    model_config: dict

    def generate_for_parameter_set(
        self,
        parameter_sampling_seed: int,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate training data for one parameter set.

        Arguments
        ---------
        parameter_sampling_seed : int
            Index of parameter set to generate (used to sample from param space)
        random_seed : int | None
            Random seed for reproducibility (may be unused for deterministic methods)

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'data': np.ndarray of training data (or None if generation failed)
            - 'theta': dict of parameter values used
            - 'success': bool indicating whether generation succeeded
            - 'error': str (optional) error message if success=False
        """
        ...

    def get_param_space(self) -> Any:
        """Get the parameter space object used by this strategy.

        Returns
        -------
        param_space : Any
            Parameter space object for sampling theta values
        """
        ...
