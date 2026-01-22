"""KDE-based likelihood estimator.

This module implements likelihood estimation via Kernel Density Estimation (KDE).
The estimator builds a KDE from simulated samples and uses it to evaluate
likelihoods and generate new samples.
"""

from typing import Any

import numpy as np

from ssms.support_utils import kde_class


class KDELikelihoodEstimator:
    """KDE-based likelihood estimator.

    This estimator builds a kernel density estimate from simulated (RT, choice)
    samples and uses it to estimate the likelihood P(RT, choice | theta).

    Attributes
    ----------
    displace_t : bool
        Whether to displace time by the t parameter (non-decision time)
    _kde : kde_class.LogKDE | None
        The fitted KDE object (None before fit() is called)
    _metadata : dict | None
        Metadata from simulations (None before fit() is called)

    Examples
    --------
    >>> estimator = KDELikelihoodEstimator(displace_t=False)
    >>> estimator.fit(simulations)
    >>> log_liks = estimator.evaluate(rts, choices)
    >>> samples = estimator.sample(n_samples=1000)
    """

    def __init__(self, displace_t: bool = False):
        """Initialize KDE likelihood estimator.

        Arguments
        ---------
        displace_t : bool, optional
            Whether to displace time by the t parameter (non-decision time).
            Only works if all simulations have the same t value.
            Default is False.
        """
        self.displace_t = displace_t
        self._kde = None
        self._metadata = None

    def fit(self, simulations: dict[str, Any]) -> None:
        """Build KDE from simulation data.

        This method constructs a kernel density estimate from the provided
        simulations. Must be called before evaluate() or sample().

        Arguments
        ---------
        simulations : dict
            Dictionary containing simulation data with keys:
            - 'rts': np.ndarray of reaction times
            - 'choices': np.ndarray of choices
            - 'metadata': dict with model information

        Notes
        -----
        Extracted from lan_mlp.py lines 305-308
        """
        self._kde = kde_class.LogKDE(
            simulations,
            displace_t=self.displace_t,
        )
        self._metadata = simulations["metadata"]

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

        Raises
        ------
        ValueError
            If fit() has not been called yet

        Notes
        -----
        Extracted from lan_mlp.py lines 312, 342-344
        """
        if self._kde is None:
            raise ValueError("Must call fit() before evaluate()")

        return self._kde.kde_eval(data={"rts": rts, "choices": choices}).ravel()

    def sample(
        self, n_samples: int, random_state: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample (RT, choice) pairs from the KDE.

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
            - 'rts': np.ndarray of sampled reaction times, shape (n_samples,)
            - 'choices': np.ndarray of sampled choices, shape (n_samples,)

        Raises
        ------
        ValueError
            If fit() has not been called yet

        Notes
        -----
        Extracted from lan_mlp.py line 311
        """
        if self._kde is None:
            raise ValueError("Must call fit() before sample()")

        samples = self._kde.kde_sample(n_samples=n_samples, random_state=random_state)

        # Flatten arrays to ensure shape (n_samples,) instead of (n_samples, 1)
        return {
            "rts": samples["rts"].ravel(),
            "choices": samples["choices"].ravel(),
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata from the simulations.

        Returns
        -------
        metadata : dict
            Dictionary containing simulation metadata including:
            - 'max_t': Maximum time value
            - 'possible_choices': List of possible choice values
            - Other model-specific metadata

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self._metadata is None:
            raise ValueError("Must call fit() before get_metadata()")

        return self._metadata
