"""Parameter transforms for the training data generation workflow.

These transforms are applied during the parameter sampling stage of the
training data generation workflow. They enforce parameter relationships
when generating synthetic training data for likelihood approximation networks.

Note: These are NOT directly relevant for basic Simulator usage, which uses
simulation transforms via ParameterSimulatorAdapters instead.

Examples:
    - SwapIfLessConstraint: Ensure a > z by swapping when violated
    - NormalizeToSumConstraint: Normalize parameters to sum to 1
"""

from ssms.transforms.sampling.swap import SwapIfLessConstraint
from ssms.transforms.sampling.normalize import NormalizeToSumConstraint

__all__ = [
    "SwapIfLessConstraint",
    "NormalizeToSumConstraint",
]
