# import importlib.metadata
from . import basic_simulators, config, dataset_generators, support_utils

__version__ = "0.9.0"  # importlib.metadata.version(__package__ or __name__)

__all__ = ["basic_simulators", "dataset_generators", "config", "support_utils"]
