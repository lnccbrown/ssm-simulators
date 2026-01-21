"""
Minimal setup.py for Cython extensions only.

All package metadata, dependencies, and configuration are defined in pyproject.toml.
This file exists solely to configure the Cython extension modules.

OpenMP support is OPTIONAL:
- If OpenMP is available, parallel simulators will use multiple threads
- If OpenMP is not available, the package still builds and works (single-threaded)
- Runtime detection via cssm._openmp_status.is_openmp_available()

GSL support is OPTIONAL (but required for correct parallel RNG):
- If GSL is available, parallel execution uses GSL's validated Ziggurat
- If GSL is not available, parallel requests fall back to single-threaded NumPy
- Runtime detection via cssm._openmp_status.is_gsl_available()
"""

import os
import subprocess
import sys
import tempfile
from setuptools import setup, Extension
import numpy

# Define all Cython extension modules (sequential only - no OpenMP/GSL needed)
CYTHON_MODULES = [
    "_utils",
    "lba_models",  # LBA models (no timestep loop, uses NumPy vectorized ops)
]

# Modules that benefit from OpenMP (will still build without it, but with OpenMP flags)
# These modules use prange/parallel for multi-threading
OPENMP_MODULES = [
    "_openmp_status",  # Runtime OpenMP/GSL detection
    "_c_rng",  # C random number generator wrapper (for testing)
    "ddm_models",  # DDM simulators with n_threads support
    "levy_models",  # Levy simulators with n_threads support
    "ornstein_models",  # Ornstein-Uhlenbeck with n_threads support
    "race_models",  # Race models with n_threads support
    "parallel_models",  # Parallel decision models with n_threads support
    "sequential_models",  # Sequential two-stage models with n_threads support
]

# Note: GSL is linked via gsl_rng.h header included in OPENMP_MODULES
# No separate GSL-only modules needed

# Cache detection results
_OPENMP_AVAILABLE = None
_GSL_AVAILABLE = None
_GSL_FLAGS = None


def gsl_available():
    """
    Check if GSL is available by running gsl-config.

    Returns True if GSL can be found and linked, False otherwise.
    Result is cached for performance.
    """
    global _GSL_AVAILABLE, _GSL_FLAGS

    if _GSL_AVAILABLE is not None:
        return _GSL_AVAILABLE

    # Allow forcing GSL on/off via environment variable
    force_gsl = os.environ.get("SSMS_FORCE_GSL", "").lower()
    if force_gsl == "0" or force_gsl == "false":
        print("GSL disabled via SSMS_FORCE_GSL=0")
        _GSL_AVAILABLE = False
        return False
    elif force_gsl == "1" or force_gsl == "true":
        print("GSL forced on via SSMS_FORCE_GSL=1")
        _GSL_AVAILABLE = True
        # Still need to get flags

    # Try to get GSL configuration
    try:
        gsl_cflags = (
            subprocess.check_output(
                ["gsl-config", "--cflags"], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
            .split()
        )
        gsl_libs = (
            subprocess.check_output(
                ["gsl-config", "--libs"], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
            .split()
        )

        _GSL_FLAGS = {
            "cflags": gsl_cflags,
            "libs": gsl_libs,
        }
        _GSL_AVAILABLE = True
        print(f"GSL available: {' '.join(gsl_libs)}")

    except (subprocess.CalledProcessError, FileNotFoundError):
        _GSL_AVAILABLE = False
        _GSL_FLAGS = None
        print("GSL not available (gsl-config not found)")
        print("  Install GSL for parallel RNG support:")
        print("    macOS:  brew install gsl")
        print("    Ubuntu: apt install libgsl-dev")
        print("    Conda:  conda install -c conda-forge gsl")

    return _GSL_AVAILABLE


def get_gsl_flags():
    """
    Get compiler and linker flags for GSL.

    Returns:
        dict with keys: extra_compile_args, extra_link_args, include_dirs
        or None if GSL is not available
    """
    if not gsl_available():
        return None

    # Parse gsl-config output
    cflags = _GSL_FLAGS.get("cflags", [])
    libs = _GSL_FLAGS.get("libs", [])

    include_dirs = []
    compile_args = []
    link_args = []

    for flag in cflags:
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])
        else:
            compile_args.append(flag)

    for flag in libs:
        link_args.append(flag)

    return {
        "include_dirs": include_dirs,
        "extra_compile_args": compile_args,
        "extra_link_args": link_args,
        "define_macros": [("HAVE_GSL", "1")],
    }


def openmp_available():
    """
    Check if OpenMP is available by trying to compile a test program.

    Returns True if OpenMP can be compiled and linked, False otherwise.
    Result is cached for performance.
    """
    global _OPENMP_AVAILABLE

    if _OPENMP_AVAILABLE is not None:
        return _OPENMP_AVAILABLE

    # Allow forcing OpenMP on/off via environment variable
    force_openmp = os.environ.get("SSMS_FORCE_OPENMP", "").lower()
    if force_openmp == "0" or force_openmp == "false":
        print("OpenMP disabled via SSMS_FORCE_OPENMP=0")
        _OPENMP_AVAILABLE = False
        return False
    elif force_openmp == "1" or force_openmp == "true":
        print("OpenMP forced on via SSMS_FORCE_OPENMP=1")
        _OPENMP_AVAILABLE = True
        return True

    # Try to compile a simple OpenMP program
    test_code = """
    #include <omp.h>
    #include <stdio.h>
    int main() {
        #pragma omp parallel
        {
            printf("Thread %d of %d\\n", omp_get_thread_num(), omp_get_num_threads());
        }
        return 0;
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "test_openmp.c")
        out_file = os.path.join(tmpdir, "test_openmp")

        with open(src_file, "w") as f:
            f.write(test_code)

        # Determine compiler and flags based on platform
        if sys.platform == "darwin":
            # macOS: try to find libomp from Homebrew
            try:
                prefix = subprocess.check_output(
                    ["brew", "--prefix", "libomp"], text=True, stderr=subprocess.DEVNULL
                ).strip()
                compile_cmd = [
                    "clang",
                    "-Xpreprocessor",
                    "-fopenmp",
                    f"-I{prefix}/include",
                    f"-L{prefix}/lib",
                    "-lomp",
                    src_file,
                    "-o",
                    out_file,
                ]
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Homebrew not available or libomp not installed
                _OPENMP_AVAILABLE = False
                print(
                    "OpenMP not available (libomp not found, install with: brew install libomp)"
                )
                return False
        elif sys.platform == "win32":
            compile_cmd = ["cl", "/openmp", src_file, f"/Fe{out_file}"]
        else:
            # Linux and others: assume GCC
            compile_cmd = ["gcc", "-fopenmp", src_file, "-o", out_file]

        try:
            subprocess.check_call(
                compile_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _OPENMP_AVAILABLE = True
            print("OpenMP available: parallel support enabled")
        except (subprocess.CalledProcessError, FileNotFoundError):
            _OPENMP_AVAILABLE = False
            print("OpenMP not available: building without parallel support")

    return _OPENMP_AVAILABLE


def get_openmp_flags(with_openmp=True):
    """
    Get compiler and linker flags for OpenMP based on platform.

    Args:
        with_openmp: If True, include OpenMP flags. If False, return optimized
                     flags without OpenMP (for graceful degradation).

    Returns:
        dict with keys: extra_compile_args, extra_link_args, include_dirs, define_macros
    """
    # Base optimization flags (always applied)
    base_flags = {
        "darwin": ["-O3", "-ffast-math"],  # Skip -march=native for portability on macOS
        "win32": ["/O2"],
        "linux": ["-O3", "-flto", "-ffast-math", "-march=native", "-funroll-loops"],
    }

    platform_key = "linux" if sys.platform not in ("darwin", "win32") else sys.platform

    if not with_openmp:
        # Return optimized flags WITHOUT OpenMP
        return {
            "extra_compile_args": base_flags.get(platform_key, ["-O3"]),
            "extra_link_args": ["-flto"] if platform_key == "linux" else [],
            "include_dirs": [],
            "define_macros": [],  # No HAVE_OPENMP
        }

    # OpenMP flags by platform
    if sys.platform == "darwin":
        # macOS with clang - needs special handling
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            prefix = "/opt/homebrew/opt/libomp"  # Default for Apple Silicon

        return {
            "extra_compile_args": [
                "-Xpreprocessor",
                "-fopenmp",
                f"-I{prefix}/include",
                "-O3",
                "-ffast-math",
            ],
            "extra_link_args": [
                f"-L{prefix}/lib",
                "-lomp",
            ],
            "include_dirs": [f"{prefix}/include"],
            "define_macros": [("HAVE_OPENMP", "1")],
        }
    elif sys.platform == "win32":
        return {
            "extra_compile_args": ["/openmp", "/O2"],
            "extra_link_args": [],
            "include_dirs": [],
            "define_macros": [("HAVE_OPENMP", "1")],
        }
    else:
        # Linux and others with GCC
        return {
            "extra_compile_args": [
                "-fopenmp",
                "-O3",
                "-flto",
                "-ffast-math",
                "-march=native",
                "-funroll-loops",
            ],
            "extra_link_args": ["-fopenmp", "-flto"],
            "include_dirs": [],
            "define_macros": [("HAVE_OPENMP", "1")],
        }


def create_extensions(modules, openmp=False, gsl=False):
    """
    Create Extension objects for Cython modules.

    Args:
        modules: List of module names (without cssm. prefix)
        openmp: If True, add OpenMP flags (if available)
        gsl: If True, add GSL flags (if available)

    Returns:
        List of Extension objects
    """
    extensions = []

    # Get flags based on OpenMP availability
    if openmp:
        has_openmp = openmp_available()
        omp_flags = get_openmp_flags(with_openmp=has_openmp)
    else:
        omp_flags = get_openmp_flags(with_openmp=False)

    # Get GSL flags if requested
    gsl_flags = get_gsl_flags() if gsl else None

    for module in modules:
        include_dirs = [numpy.get_include(), "src/cssm"]
        include_dirs.extend(omp_flags.get("include_dirs", []))

        compile_args = list(omp_flags.get("extra_compile_args", []))
        link_args = list(omp_flags.get("extra_link_args", []))
        define_macros = list(omp_flags.get("define_macros", []))

        # Add GSL flags if available
        if gsl_flags:
            include_dirs.extend(gsl_flags.get("include_dirs", []))
            compile_args.extend(gsl_flags.get("extra_compile_args", []))
            link_args.extend(gsl_flags.get("extra_link_args", []))
            define_macros.extend(gsl_flags.get("define_macros", []))

        ext_kwargs = {
            "name": f"cssm.{module}",
            "sources": [f"src/cssm/{module}.pyx"],
            "language": "c++",
            "include_dirs": include_dirs,
            "extra_compile_args": compile_args,
            "extra_link_args": link_args,
            "define_macros": define_macros,
        }

        extensions.append(Extension(**ext_kwargs))

    return extensions


# Try to build with Cython if available
try:
    from Cython.Build import cythonize

    # Check OpenMP and GSL availability once at the start
    HAS_OPENMP = openmp_available()
    HAS_GSL = gsl_available()

    # Standard modules (no OpenMP/GSL needed)
    standard_extensions = create_extensions(CYTHON_MODULES, openmp=False, gsl=False)

    # OpenMP modules (will use OpenMP if available, otherwise graceful degradation)
    # These also get GSL flags so they can detect GSL at runtime
    openmp_extensions = create_extensions(OPENMP_MODULES, openmp=True, gsl=HAS_GSL)

    ext_modules = cythonize(
        standard_extensions + openmp_extensions,
        compiler_directives={"language_level": "3"},
    )

    # Print build summary
    print(f"\n{'=' * 60}")
    print("SSMS Build Configuration")
    print(f"{'=' * 60}")
    print(f"  Platform:     {sys.platform}")
    print(
        f"  OpenMP:       {'enabled' if HAS_OPENMP else 'disabled (parallel features unavailable)'}"
    )
    print(
        f"  GSL:          {'enabled' if HAS_GSL else 'disabled (parallel RNG falls back to NumPy)'}"
    )
    print(
        f"  Modules:      {len(CYTHON_MODULES)} standard + {len(OPENMP_MODULES)} parallel"
    )
    print(f"{'=' * 60}\n")

except ImportError:
    # Fallback to pre-compiled .c files if Cython is not available
    print("Warning: Cython not available, using pre-compiled sources")
    ext_modules = create_extensions(CYTHON_MODULES, openmp=False, gsl=False)

# Minimal setup call - all metadata comes from pyproject.toml
setup(
    ext_modules=ext_modules,
)
