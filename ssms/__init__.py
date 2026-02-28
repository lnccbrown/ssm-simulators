import importlib.metadata
import os

# Configure multiprocessing start method for OpenMP safety BEFORE any other imports
# 'spawn' is safer with OpenMP (used by Cython simulators) and works across all platforms
# 'fork' can cause deadlocks when combined with OpenMP threads
# Users can override via SSMS_MP_START_METHOD environment variable
_MP_START_METHOD = os.environ.get("SSMS_MP_START_METHOD", "spawn")
try:
    import multiprocess

    multiprocess.set_start_method(_MP_START_METHOD, force=False)
except (RuntimeError, ImportError):
    # Already set or multiprocess not installed - this is fine
    pass

from . import basic_simulators
from . import dataset_generators
from . import config
from . import support_utils
from . import hssm_support
from .basic_simulators import Simulator, OMISSION_SENTINEL
from .config import get_default_generator_config

__version__ = importlib.metadata.version("ssm-simulators")

__all__ = [
    "basic_simulators",
    "dataset_generators",
    "config",
    "support_utils",
    "hssm_support",
    "Simulator",
    "OMISSION_SENTINEL",
    "get_default_generator_config",
]
