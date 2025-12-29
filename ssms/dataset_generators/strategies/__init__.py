"""Training data generation strategies.

This module contains strategy classes for training data generation:

- MixtureTrainingStrategy: Mixture of KDE samples + uniform samples

Note: For end-to-end data generation workflows, see the `pipelines` module which
contains SimulationPipeline and PyDDMPipeline.
"""

from ssms.dataset_generators.strategies.mixture_training_strategy import (
    MixtureTrainingStrategy,
)

__all__ = [
    "MixtureTrainingStrategy",
]
