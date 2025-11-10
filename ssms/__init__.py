import importlib.metadata

from . import basic_simulators
from . import dataset_generators
from . import config
from . import support_utils
from . import hssm_support
from .basic_simulators import Simulator

__version__ = importlib.metadata.version("ssm-simulators")

__all__ = [
    "basic_simulators",
    "dataset_generators",
    "config",
    "support_utils",
    "hssm_support",
    "Simulator",
]
