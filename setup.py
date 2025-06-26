from setuptools import setup, Extension, find_packages
import numpy

# Try to build with Cython if available
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [Extension("cssm", ["src/cssm.pyx"], language="c++")],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = [Extension("cssm", ["src/cssm.pyx"], language="c++")]

# Use find_packages to automatically discover all packages
packages = find_packages(include=['ssms', 'ssms.*'])

setup(
    name="ssm-simulators",
    version="0.8.3",
    packages=packages,
    package_data={
        'ssms': ['**/*.py', '**/*.pyx', '**/*.pxd', '**/*.so', '**/*.pyd'],
    },
    include_package_data=True,
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'tqdm',
        'pyyaml',
        'typer',
    ],
    entry_points={
        'console_scripts': [
            'generate=ssms.generate:main',
        ],
    },
)
