from setuptools import setup, Extension, find_packages
import numpy

# Try to build with Cython if available
try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            # New modular structure
            Extension(
                "cssm._utils",
                ["src/cssm/_utils.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.ddm_models",
                ["src/cssm/ddm_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.race_models",
                ["src/cssm/race_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.lba_models",
                ["src/cssm/lba_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.sequential_models",
                ["src/cssm/sequential_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.parallel_models",
                ["src/cssm/parallel_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.levy_models",
                ["src/cssm/levy_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
            Extension(
                "cssm.ornstein_models",
                ["src/cssm/ornstein_models.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/cssm"],
            ),
        ],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = [
        Extension(
            "cssm._utils",
            ["src/cssm/_utils.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.ddm_models",
            ["src/cssm/ddm_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.race_models",
            ["src/cssm/race_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.lba_models",
            ["src/cssm/lba_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.sequential_models",
            ["src/cssm/sequential_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.parallel_models",
            ["src/cssm/parallel_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.levy_models",
            ["src/cssm/levy_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
        Extension(
            "cssm.ornstein_models",
            ["src/cssm/ornstein_models.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), "src/cssm"],
        ),
    ]

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
