# import importlib.metadata
from . import basic_simulators
from . import dataset_generators
from . import config
from . import support_utils

__version__ = "0.10.2"  # importlib.metadata.version(__package__ or __name__)

__all__ = ["basic_simulators", "dataset_generators", "config", "support_utils"]
