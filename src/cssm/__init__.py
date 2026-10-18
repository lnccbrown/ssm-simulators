"""
CSSM - Cython Sequential Sampling Models

This package provides high-performance Cython implementations of various
sequential sampling models used in cognitive science and neuroscience.

Backward compatibility layer for the refactored modules.
Users can import functions as before: `import cssm; cssm.levy_flexbound(...)`
Or use the new modular structure: `from cssm.levy_models import levy_flexbound`
"""

import importlib.metadata

from .addm_models import addm

# Import simulators from their respective modules
from .ddm_models import (
    ddm,
    ddm_flex,
    ddm_flex_leak,
    ddm_flex_leak2,
    ddm_flexbound,
    ddm_flexbound_tradeoff,
    ddm_sdv,
    full_ddm,
    full_ddm_hddm_base,
    full_ddm_rv,
)
from .lba_models import lba_angle, lba_vanilla, rlwm_lba_pw_v1, rlwm_lba_race
from .levy_models import levy_flexbound
from .ornstein_models import ornstein_uhlenbeck
from .parallel_models import ddm_flexbound_par2
from .poisson_race_models import poisson_race
from .race_models import lca, race_model, racing_diffusion_model
from .race_multistage_models import race_multistage
from .sequential_models import (
    ddm_flexbound_mic2_multinoise,
    ddm_flexbound_mic2_ornstein,
    ddm_flexbound_mic2_ornstein_multinoise,
    ddm_flexbound_mic2_unnormalized_ornstein_multinoise,
    ddm_flexbound_seq2,
)

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
    # ADDM model
    "addm",
    "race_multistage",
    # Race models
    "race_model",
    "lca",
    "racing_diffusion_model",
    # Poisson race models
    "poisson_race",
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

# Single-source the version from package metadata (mirrors ssms/__init__.py) so it
# never drifts from pyproject.toml across releases.
__version__ = importlib.metadata.version("ssm-simulators")
