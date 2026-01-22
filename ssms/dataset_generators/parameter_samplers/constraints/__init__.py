"""Parameter sampling constraint classes.

NOTE: This module re-exports from ssms.transforms for backwards compatibility.
Import directly from ssms.transforms instead:

    from ssms.transforms import SwapIfLessConstraint, NormalizeToSumConstraint
"""

# Re-export from canonical location for backwards compatibility
from ssms.transforms.sampling import (
    SwapIfLessConstraint,
    NormalizeToSumConstraint,
)

__all__ = [
    "SwapIfLessConstraint",
    "NormalizeToSumConstraint",
]
