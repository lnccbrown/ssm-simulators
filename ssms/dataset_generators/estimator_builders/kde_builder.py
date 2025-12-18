"""KDE Estimator Builder.

This module implements the builder pattern for KDE-based likelihood estimators.
The builder extracts relevant parameters from the generator_config and instantiates
a KDELikelihoodEstimator with explicit parameters.

This separation of concerns ensures that:
1. KDELikelihoodEstimator remains focused on likelihood estimation
2. Configuration parsing logic is centralized in the builder
3. New estimator parameters can be added with minimal changes
"""

from typing import Any, Dict

from ssms.dataset_generators.likelihood_estimators.kde_estimator import (
    KDELikelihoodEstimator,
)


class KDEEstimatorBuilder:
    """Builder for KDE-based likelihood estimators.

    This builder is responsible for:
    1. Extracting relevant parameters from the generator_config
    2. Instantiating KDELikelihoodEstimator with explicit parameters
    3. Fitting the estimator with provided simulations

    Attributes
    ----------
    generator_config : dict
        Configuration dictionary containing KDE settings
    displace_t : bool
        Extracted parameter: whether to displace time by the t parameter

    Examples
    --------
    >>> builder = KDEEstimatorBuilder(generator_config)
    >>> estimator = builder.build(theta, simulations)
    >>> log_liks = estimator.evaluate(rts, choices)

    Notes
    -----
    Implements EstimatorBuilderProtocol from protocols.py
    Extracted from lan_mlp.py as part of Phase 1 refactoring
    """

    def __init__(self, generator_config: dict):
        """Initialize KDE estimator builder.

        Arguments
        ---------
        generator_config : dict
            Configuration dictionary containing:
            - 'kde_displace_t' (optional): Whether to displace time by t parameter.
              Defaults to False if not specified.
        """
        self.generator_config = generator_config

        # Extract and cache the displace_t parameter
        self.displace_t = generator_config.get("kde_displace_t", False)

    def build(
        self, theta: Dict[str, Any], simulations: Dict[str, Any] | None = None
    ) -> KDELikelihoodEstimator:
        """Build and fit a KDE likelihood estimator.

        This method:
        1. Creates a KDELikelihoodEstimator with extracted parameters
        2. Fits it with the provided simulations

        Arguments
        ---------
        theta : dict
            Parameter dictionary (not directly used by KDE, but part of protocol)
        simulations : dict | None
            Simulation data containing 'rts', 'choices', and 'metadata' keys.
            Required for KDE estimators (cannot be None).

        Returns
        -------
        KDELikelihoodEstimator
            A fitted likelihood estimator ready for evaluate() or sample() calls

        Raises
        ------
        ValueError
            If simulations is None (KDE requires simulation data)

        Examples
        --------
        >>> builder = KDEEstimatorBuilder({"kde_displace_t": False})
        >>> estimator = builder.build(theta, simulations)
        >>> assert estimator._kde is not None  # Verifies it's fitted

        Notes
        -----
        The theta parameter is included in the signature to conform to
        EstimatorBuilderProtocol. Future estimator types (e.g., PyDDM) will
        use theta to configure model-specific parameters.
        """
        if simulations is None:
            raise ValueError(
                "KDE estimator requires simulations data. "
                "Cannot build KDE estimator with simulations=None."
            )

        # Instantiate estimator with explicit parameters (cleaner design!)
        estimator = KDELikelihoodEstimator(displace_t=self.displace_t)

        # Fit the estimator
        estimator.fit(simulations)

        return estimator
