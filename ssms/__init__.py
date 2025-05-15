"""SSM Simulators package."""

__version__ = "0.9.0"  # importlib.metadata.version(__package__ or __name__)

# Define what should be available in the public API
__all__ = [
    "basic_simulators",
    "dataset_generators",
    "config",
    "support_utils",
    "boundary_functions",
    "drift_functions",
]


# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__package__} has no attribute {name}")
