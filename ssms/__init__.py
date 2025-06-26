# import importlib.metadata
# Import only the package names without importing the modules immediately
__all__ = ["basic_simulators", "dataset_generators", "config", "support_utils", "main"]

# Lazy loading of submodules
import importlib
import sys
from typing import Any

class _LazyLoader:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(f"ssms.{self._module_name}")
        return getattr(self._module, name)

# Set up lazy loading for submodules
for module_name in __all__:
    if module_name != "main":
        setattr(sys.modules[__name__], module_name, _LazyLoader(module_name))

# Import main directly since it's likely needed immediately
from .generate import main

__version__ = "0.8.3"  # importlib.metadata.version(__package__ or __name__)
