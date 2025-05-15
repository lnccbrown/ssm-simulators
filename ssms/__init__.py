"""SSM Simulators package."""

__version__ = "0.9.0"  # importlib.metadata.version(__package__ or __name__)

# Import core functionality first
from . import boundary_functions  # noqa: F401
from . import drift_functions  # noqa: F401
from . import core  # noqa: F401

# Then import config which depends on core
from . import config  # noqa: F401

# Then import simulators which depend on config
from . import basic_simulators  # noqa: F401

# Finally import higher level modules
from . import dataset_generators  # noqa: F401
from . import support_utils  # noqa: F401

__all__ = [
    "basic_simulators",
    "dataset_generators",
    "config",
    "support_utils",
    "boundary_functions",
    "drift_functions",
    "core",
]
