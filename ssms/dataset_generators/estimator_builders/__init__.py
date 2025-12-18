"""Estimator builder implementations.

This module contains builder classes that construct likelihood estimators
from configuration dictionaries. Builders follow the EstimatorBuilderProtocol
and handle the responsibility of extracting relevant parameters from configs.

Available builders:
- KDEEstimatorBuilder: Builds KDE-based likelihood estimators
  - Works with all models
  - Requires simulation data

- PyDDMEstimatorBuilder: Builds PyDDM analytical PDF estimators
  - Only for compatible models (single-particle, two-choice, Gaussian noise)
  - Does not require simulation data (purely analytical)
  - Requires optional 'pyddm' package: pip install pyddm

Factory:
- create_estimator_builder: Factory function for creating builders based on config
  - Automatically selects appropriate builder based on 'estimator_type'
  - Handles compatibility checking and error messages
"""

from ssms.dataset_generators.estimator_builders.kde_builder import KDEEstimatorBuilder
from ssms.dataset_generators.estimator_builders.builder_factory import (
    create_estimator_builder,
)

# PyDDM is optional dependency
try:
    from ssms.dataset_generators.estimator_builders.pyddm_builder import (
        PyDDMEstimatorBuilder,
    )

    __all__ = [
        "KDEEstimatorBuilder",
        "PyDDMEstimatorBuilder",
        "create_estimator_builder",
    ]
except ImportError:
    __all__ = ["KDEEstimatorBuilder", "create_estimator_builder"]
