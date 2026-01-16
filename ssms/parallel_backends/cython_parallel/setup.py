"""
Setup script for the Cython nogil parallel extension.

Build with:
    python setup.py build_ext --inplace

Or using pip:
    pip install -e '.[dev]'
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Compiler flags for OpenMP
extra_compile_args = ["-fopenmp", "-O3", "-ffast-math"]
extra_link_args = ["-fopenmp"]

# Handle macOS specifics
if os.uname().sysname == "Darwin":
    # macOS with Apple Clang doesn't have native OpenMP
    # Try to use libomp from Homebrew
    try:
        import subprocess

        brew_prefix = subprocess.check_output(["brew", "--prefix"]).decode().strip()
        extra_compile_args = [
            f"-I{brew_prefix}/opt/libomp/include",
            "-Xpreprocessor",
            "-fopenmp",
            "-O3",
            "-ffast-math",
        ]
        extra_link_args = [f"-L{brew_prefix}/opt/libomp/lib", "-lomp"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: OpenMP not found. Parallel extensions may not work.")
        print("Install with: brew install libomp")
        extra_compile_args = ["-O3", "-ffast-math"]
        extra_link_args = []

extensions = [
    Extension(
        "ssms.parallel_backends.cython_parallel.ddm_nogil",
        ["ddm_nogil.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="ssms_cython_parallel",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
