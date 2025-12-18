"""Builder for PyDDM-based likelihood estimators."""

from typing import Any, Dict

from ssms.external_simulators import SSMSToPyDDMMapper
from ssms.dataset_generators.likelihood_estimators.pyddm_estimator import (
    PyDDMLikelihoodEstimator,
)


class PyDDMEstimatorBuilder:
    """Builder for PyDDM analytical PDF estimators.

    This builder:
    1. Validates model compatibility with PyDDM at construction
    2. Builds PyDDM models using SSMSToPyDDMMapper for each theta
    3. Solves the Fokker-Planck equation analytically
    4. Wraps the solution in a PyDDMLikelihoodEstimator

    Key advantages:
    - No simulation required (purely analytical)
    - Deterministic results
    - Fast for compatible models

    Compatible models:
    - Single-particle, two-choice models
    - Gaussian noise only
    - Examples: ddm, ornstein, angle, weibull, gamma_drift, conflict_*

    Incompatible models:
    - Multi-particle: race_*, lca_*, lba*
    - Non-Gaussian noise: levy
    - Inter-trial variability: full_ddm, ddm_sdv, ddm_st
    """

    def __init__(self, generator_config: dict, model_config: dict):
        """Initialize builder with configuration.

        Validates model compatibility at construction time (fail-fast approach).

        Args:
            generator_config: Generator settings including:
                - delta_t: Time step for solving
                - max_t: Maximum time
                - pdf_interpolation: 'linear' or 'cubic' (optional, defaults to 'cubic')
            model_config: Model specification from ssms.config

        Raises:
            ValueError: If model is not compatible with PyDDM
            ImportError: If pyddm package is not installed
        """
        self.generator_config = generator_config
        self.model_config = model_config

        # Validate compatibility at construction (fail fast)
        is_compat, reason = SSMSToPyDDMMapper.is_compatible(model_config)
        if not is_compat:
            raise ValueError(
                f"Model '{model_config['name']}' not compatible with PyDDM: {reason}. "
                f"Compatible models require: single-particle, two-choice, Gaussian noise. "
                f"Use 'estimator_type': 'kde' for this model."
            )

    def build(
        self, theta: Dict[str, Any], simulations: Dict[str, Any] | None = None
    ) -> PyDDMLikelihoodEstimator:
        """Build PyDDM likelihood estimator for given parameters.

        Unlike KDE builder, this does NOT require simulation data. The estimator
        is built by solving the Fokker-Planck equation analytically.

        Args:
            theta: Model parameters (v, a, z, t, etc.)
            simulations: Ignored (PyDDM doesn't need simulations)

        Returns:
            PyDDMLikelihoodEstimator instance fitted to theta

        Raises:
            ValueError: If PyDDM model cannot be built (e.g., invalid parameters)

        Note:
            The simulations parameter is accepted for protocol compliance
            but is not used by PyDDM estimators.
        """
        # Build PyDDM model using the mapper (handles all complexity)
        pyddm_model = SSMSToPyDDMMapper.build_pyddm_model(
            model_config=self.model_config,
            theta=theta,
            generator_config=self.generator_config,
        )

        # Solve Fokker-Planck equation analytically
        solution = pyddm_model.solve()

        # Get time domain
        t_domain = pyddm_model.t_domain()

        # Check if undecided probability is too high
        # High P(undecided) makes sampling inefficient/intractable
        max_undecided_prob = self.generator_config.get("max_undecided_prob", 0.5)
        p_undecided = solution.prob_undecided()

        if p_undecided > max_undecided_prob:
            raise ValueError(
                f"P(undecided)={p_undecided:.4f} exceeds threshold {max_undecided_prob}. "
                f"Parameters produce too many undecided trials for efficient sampling. "
                f"Theta: {theta}"
            )

        # Get interpolation method from config (default to cubic)
        interpolation = self.generator_config.get("pdf_interpolation", "cubic")

        # Create estimator (extracts PDFs internally)
        estimator = PyDDMLikelihoodEstimator(
            pyddm_solution=solution,
            t_domain=t_domain,
            interpolation=interpolation,
        )

        return estimator
