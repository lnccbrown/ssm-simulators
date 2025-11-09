"""
CSSM - Cython Sequential Sampling Models

This package provides high-performance Cython implementations of various
sequential sampling models used in cognitive science and neuroscience.

Backward compatibility layer for the refactored modules.
Users can import functions as before: `import cssm; cssm.levy_flexbound(...)`
Or use the new modular structure: `from cssm.levy_models import levy_flexbound`
"""

# Import simulators from their respective modules
from .ddm_models import (
    full_ddm_hddm_base,
    ddm,
    ddm_flexbound,
    ddm_flex,
    ddm_flex_leak,
    ddm_flex_leak2,
    full_ddm_rv,
    full_ddm,
    ddm_sdv,
    ddm_flexbound_tradeoff,
)
from .race_models import race_model, lca
from .lba_models import lba_vanilla, lba_angle, rlwm_lba_pw_v1, rlwm_lba_race
from .sequential_models import (
    ddm_flexbound_seq2,
    ddm_flexbound_mic2_ornstein,
    ddm_flexbound_mic2_multinoise,
    ddm_flexbound_mic2_ornstein_multinoise,
    ddm_flexbound_mic2_unnormalized_ornstein_multinoise,
)
from .parallel_models import ddm_flexbound_par2
from .levy_models import levy_flexbound
from .ornstein_models import ornstein_uhlenbeck

__all__ = [
    # DDM models
    "full_ddm_hddm_base",
    "ddm",
    "ddm_flexbound",
    "ddm_flex",
    "ddm_flex_leak",
    "ddm_flex_leak2",
    "full_ddm_rv",
    "full_ddm",
    "ddm_sdv",
    "ddm_flexbound_tradeoff",
    # Race models
    "race_model",
    "lca",
    # LBA models
    "lba_vanilla",
    "lba_angle",
    "rlwm_lba_pw_v1",
    "rlwm_lba_race",
    # Sequential models
    "ddm_flexbound_seq2",
    "ddm_flexbound_mic2_ornstein",
    "ddm_flexbound_mic2_multinoise",
    "ddm_flexbound_mic2_ornstein_multinoise",
    "ddm_flexbound_mic2_unnormalized_ornstein_multinoise",
    # Parallel models
    "ddm_flexbound_par2",
    # Levy models
    "levy_flexbound",
    # Ornstein models
    "ornstein_uhlenbeck",
]

__version__ = "0.11.3"
