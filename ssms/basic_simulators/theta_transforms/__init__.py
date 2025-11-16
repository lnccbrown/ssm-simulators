"""
Theta transformation system for modular parameter processing.

This module provides a composable system for transforming theta parameters
before they are passed to simulators. Transformations are small, focused
classes that can be combined to handle complex parameter processing logic.

Main Components:
    - ThetaTransformation: Base class for all transformations
    - Common transformations: SetDefaultValue, ExpandDimension, etc.
    - ThetaProcessorRegistry: Maps models to transformation pipelines
    - ModularThetaProcessor: Applies transformations in sequence

Example:
    >>> from ssms.basic_simulators.theta_transforms import (
    ...     ExpandDimension, ColumnStackParameters
    ... )
    >>>
    >>> transforms = [
    ...     ColumnStackParameters(["v0", "v1", "v2"], "v"),
    ...     ExpandDimension(["a", "t"])
    ... ]
    >>>
    >>> for transform in transforms:
    ...     theta = transform.apply(theta, model_config, n_trials)
"""

from .base import ThetaTransformation
from .common import (
    SetDefaultValue,
    ExpandDimension,
    ColumnStackParameters,
    RenameParameter,
    DeleteParameters,
    SetZeroArray,
    TileArray,
    ApplyMapping,
    ConditionalTransform,
    LambdaTransformation,
)
from .registry import ThetaProcessorRegistry

__all__ = [
    # Base class
    "ThetaTransformation",
    # Common transformations
    "SetDefaultValue",
    "ExpandDimension",
    "ColumnStackParameters",
    "RenameParameter",
    "DeleteParameters",
    "SetZeroArray",
    "TileArray",
    "ApplyMapping",
    "ConditionalTransform",
    "LambdaTransformation",
    # Registry
    "ThetaProcessorRegistry",
]
