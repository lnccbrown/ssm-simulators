"""Unified parameter transforms for the training data generation and simulation workflows.

This module provides a unified system for parameter transformations that can be
applied at two different stages:

1. **Sampling transforms** (for training data generation):
   Applied during the parameter sampling stage of the training data generation
   workflow. These enforce parameter relationships (e.g., a > z) when generating
   synthetic training data for likelihood approximation networks.
   - Examples: SwapIfLessConstraint, NormalizeToSumConstraint

2. **Simulation transforms** (for basic Simulator):
   Applied via ParameterSimulatorAdapters when running the basic Simulator.
   These prepare user-provided parameters for the low-level C/Cython simulators
   (e.g., stacking v0, v1, v2 into a single v array).
   - Examples: ColumnStackParameters, ExpandDimension, SetZeroArray

All transforms inherit from ParameterTransform and use the same interface,
making it easy to compose them into pipelines.

Example usage in model config:

    from ssms.transforms import (
        SwapIfLessConstraint,
        ColumnStackParameters,
        ExpandDimension,
        SetZeroArray,
    )

    def get_lba_angle_3_config():
        return {
            "name": "lba_angle_3",
            "params": ["v0", "v1", "v2", "a", "z", "theta"],
            # ... other fields ...

            "parameter_transforms": {
                "sampling": [
                    SwapIfLessConstraint("a", "z"),
                ],
                "simulation": [
                    ColumnStackParameters(["v0", "v1", "v2"], "v"),
                    ExpandDimension(["a", "z", "theta"]),
                    SetZeroArray("t"),
                ],
            }
        }
"""

# Base class
from ssms.transforms.base import ParameterTransform

# Sampling transforms (for training data generation workflow)
from ssms.transforms.sampling import (
    SwapIfLessConstraint,
    NormalizeToSumConstraint,
)

# Simulation transforms (for basic Simulator via ParameterSimulatorAdapters)
from ssms.transforms.simulation import (
    SetDefaultValue,
    ExpandDimension,
    ColumnStackParameters,
    RenameParameter,
    DeleteParameters,
    SetZeroArray,
    TileArray,
    ApplyMapping,
    ConditionalAdaptation,
    LambdaAdaptation,
)

__all__ = [
    # Base
    "ParameterTransform",
    # Sampling
    "SwapIfLessConstraint",
    "NormalizeToSumConstraint",
    # Simulation
    "SetDefaultValue",
    "ExpandDimension",
    "ColumnStackParameters",
    "RenameParameter",
    "DeleteParameters",
    "SetZeroArray",
    "TileArray",
    "ApplyMapping",
    "ConditionalAdaptation",
    "LambdaAdaptation",
]
