# import importlib.metadata
__version__ = "0.8.3"  # importlib.metadata.version(__package__ or __name__)

# Import main modules in a way that avoids circular imports
import sys
from typing import Any

# Define __all__ for explicit exports
__all__ = ["basic_simulators", "dataset_generators", "config", "support_utils", "main"]

# Import main directly since it's needed for the CLI
from .generate import main

# Lazy loader for other modules to prevent circular imports
class _LazyModule:
    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, item):
        if self._module is None:
            self._module = __import__(f"ssms.{self._name}", fromlist=['*'])
        return getattr(self._module, item)

# Set up lazy loading for submodules
for module_name in ["basic_simulators", "dataset_generators", "config", "support_utils"]:
    setattr(sys.modules[__name__], module_name, _LazyModule(module_name))
