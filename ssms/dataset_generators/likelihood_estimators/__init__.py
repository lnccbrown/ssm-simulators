"""Likelihood estimators for training data generation.

This module provides implementations of LikelihoodEstimatorProtocol:
- KDELikelihoodEstimator: KDE-based estimator from simulated samples
  - Works with all models
  - Requires simulation data for fitting

- PyDDMLikelihoodEstimator: Analytical PDF-based estimator using PyDDM
  - Only for compatible models (single-particle, two-choice, Gaussian noise)
  - Does not require simulation data (uses Fokker-Planck solution)
  - Deterministic results
  - Requires optional 'pyddm' package: pip install pyddm
"""

from .kde_estimator import KDELikelihoodEstimator

# PyDDM is optional dependency
try:
    from .pyddm_estimator import PyDDMLikelihoodEstimator

    __all__ = ["KDELikelihoodEstimator", "PyDDMLikelihoodEstimator"]
except ImportError:
    __all__ = ["KDELikelihoodEstimator"]
