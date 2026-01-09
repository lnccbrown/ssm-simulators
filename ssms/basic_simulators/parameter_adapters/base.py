"""
Base classes for parameter adaptations.

This module provides backward compatibility by re-exporting ParameterTransform
as ParameterAdaptation.

Note: New code should use ParameterTransform from ssms.transforms.base directly.
"""

from ssms.transforms.base import ParameterTransform

# Backward compatibility alias
ParameterAdaptation = ParameterTransform

__all__ = ["ParameterAdaptation", "ParameterTransform"]
