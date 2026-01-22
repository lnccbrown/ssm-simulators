"""Training data generation strategies.

This module contains strategy classes for training data generation:

- ResampleMixtureStrategy: Generates training data by mixing samples from a
  likelihood estimator (KDE or PyDDM) with uniform samples in positive and
  negative RT space. This helps networks learn both the likelihood surface
  and its boundaries.

- MixtureTrainingStrategy: Alias for ResampleMixtureStrategy (for backward
  compatibility).

Note: For end-to-end data generation workflows, see the `pipelines` module which
contains SimulationPipeline and PyDDMPipeline.
"""

from ssms.dataset_generators.strategies.resample_mixture_strategy import (
    ResampleMixtureStrategy,
)

# Backward compatibility alias
MixtureTrainingStrategy = ResampleMixtureStrategy

__all__ = [
    "ResampleMixtureStrategy",
    "MixtureTrainingStrategy",  # Alias for backward compatibility
]
