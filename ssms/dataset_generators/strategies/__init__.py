"""Training data generation strategies.

This module contains strategy classes that generate training data from
likelihood estimators. Strategies follow the TrainingDataStrategyProtocol
and implement different approaches to sampling and data augmentation.

Available strategies:
- ResampleMixtureStrategy: Mixture of KDE samples + uniform samples (positive and negative RT)
"""

from ssms.dataset_generators.strategies.resample_mixture_strategy import (
    ResampleMixtureStrategy,
)

__all__ = ["ResampleMixtureStrategy"]
