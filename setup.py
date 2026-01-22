from setuptools import setup, Extension, find_packages
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

# Use find_packages to automatically discover all packages
packages = find_packages(include=["ssms", "ssms.*"])

setup(
    name="ssm-simulators",
    version="0.10.2",
    packages=packages,
    package_dir={"cssm": "src/cssm"},  # Map cssm package to source directory
    package_data={
        "ssms": ["**/*.py", "**/*.pyx", "**/*.pxd", "**/*.so", "**/*.pyd"],
        "cssm": ["*.py", "*.pyx", "*.pxd"],  # Include cssm source files
    },
    include_package_data=True,
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "typer",
    ],
)
