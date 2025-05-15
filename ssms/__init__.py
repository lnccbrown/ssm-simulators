"""SSM Simulators package."""

__version__ = "0.9.0"  # importlib.metadata.version(__package__ or __name__)

# Import submodules after version to avoid circular imports
from . import basic_simulators  # noqa: F401
from . import config  # noqa: F401
from . import dataset_generators  # noqa: F401
from . import support_utils  # noqa: F401

__all__ = [
    "basic_simulators",
    "dataset_generators",
    "config",
    "support_utils",
]
