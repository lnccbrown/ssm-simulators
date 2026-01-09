"""
Parameter adaptation system for modular parameter processing.

This module provides a composable system for adapting parameters
before they are passed to simulators. Adaptations are small, focused
classes that can be combined to handle complex parameter preparation logic.

Main Components:
    - ParameterAdaptation: Alias for ParameterTransform (backward compatibility)
    - Common adaptations: SetDefaultValue, ExpandDimension, etc.
    - ParameterAdapterRegistry: Maps models to adaptation pipelines
    - ModularParameterSimulatorAdapter: Applies adaptations in sequence

Example:
    >>> from ssms.basic_simulators.parameter_adapters import (
    ...     ExpandDimension, ColumnStackParameters
    ... )
    >>>
    >>> adaptations = [
    ...     ColumnStackParameters(["v0", "v1", "v2"], "v"),
    ...     ExpandDimension(["a", "t"])
    ... ]
    >>>
    >>> for adaptation in adaptations:
    ...     theta = adaptation.apply(theta, model_config, n_trials)

Note:
    New code should import directly from ssms.transforms.simulation for the
    common adaptation classes. This module re-exports them for backward
    compatibility.
"""

from .base import ParameterAdaptation

# Re-export common adaptations from transforms module (single source of truth)
from ssms.transforms.simulation.common import (
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
from .registry import (
    ParameterAdapterRegistry,
    register_adapter_to_model,
    register_adapter_to_model_family,
    get_adapter_registry,
)

__all__ = [
    # Base class
    "ParameterAdaptation",
    # Common adaptations (re-exported from transforms.simulation)
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
    # Registry
    "ParameterAdapterRegistry",
    # Module-level registry functions
    "register_adapter_to_model",
    "register_adapter_to_model_family",
    "get_adapter_registry",
]
