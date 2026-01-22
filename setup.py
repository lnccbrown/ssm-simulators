"""
Minimal setup.py for Cython extensions only.

All package metadata, dependencies, and configuration are defined in pyproject.toml.
This file exists solely to configure the Cython extension modules.
"""

from setuptools import setup, Extension
import numpy

# Define all Cython extension modules
CYTHON_MODULES = [
    "_utils",
    "ddm_models",
    "race_models",
    "poisson_race_models",
    "lba_models",
    "sequential_models",
    "parallel_models",
    "levy_models",
    "ornstein_models",
]


def create_extensions(modules):
    """Create Extension objects for all Cython modules."""
    return [
        Extension(
            f"cssm.{module}",
            [f"src/cssm/{module}.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        )
        for module in modules
    ]


# Try to build with Cython if available
try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        create_extensions(CYTHON_MODULES),
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    # Fallback to pre-compiled .c files if Cython is not available
    ext_modules = create_extensions(CYTHON_MODULES)

# Minimal setup call - all metadata comes from pyproject.toml
setup(
    ext_modules=ext_modules,
)
