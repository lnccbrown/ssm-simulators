# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False

"""
OpenMP and GSL runtime status detection.

This module provides runtime detection of whether OpenMP and GSL support was
compiled into the ssm-simulators package. It is used to:

1. Warn users when they request n_threads > 1 but OpenMP/GSL is not available
2. Allow introspection of parallel capabilities
3. Provide installation guidance

Usage:
    from cssm._openmp_status import is_openmp_available, is_gsl_available
    from cssm._openmp_status import check_parallel_request

    # Check capabilities
    if is_openmp_available() and is_gsl_available():
        # Full parallel support available
        pass

    # In simulator functions
    n_threads = check_parallel_request(n_threads)  # Validates and warns
"""

import os

# =============================================================================
# OpenMP Detection
# =============================================================================

# Detect OpenMP at compile time using preprocessor
cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define COMPILED_WITH_OPENMP 1
    #else
        #define COMPILED_WITH_OPENMP 0
        // Stub functions when OpenMP not available
        static int omp_get_max_threads(void) { return 1; }
        static int omp_get_num_procs(void) { return 1; }
    #endif
    """
    int COMPILED_WITH_OPENMP
    int omp_get_max_threads() nogil
    int omp_get_num_procs() nogil


# =============================================================================
# GSL Detection
# =============================================================================

# Detect GSL at compile time using preprocessor
cdef extern from *:
    """
    #ifdef HAVE_GSL
        #define COMPILED_WITH_GSL 1
    #else
        #define COMPILED_WITH_GSL 0
    #endif
    """
    int COMPILED_WITH_GSL


# Cache the results at module load time
_OPENMP_AVAILABLE = bool(COMPILED_WITH_OPENMP)
_GSL_AVAILABLE = bool(COMPILED_WITH_GSL)


# =============================================================================
# Public API - OpenMP
# =============================================================================

def is_openmp_available():
    """
    Check if OpenMP support was compiled into the package.

    Returns:
        bool: True if OpenMP is available, False otherwise.

    Example:
        >>> from cssm._openmp_status import is_openmp_available
        >>> if is_openmp_available():
        ...     print("Parallel simulation available!")
        ... else:
        ...     print("Running in single-threaded mode")
    """
    return _OPENMP_AVAILABLE


def get_max_threads():
    """
    Get the maximum number of OpenMP threads available.

    Returns:
        int: Maximum threads if OpenMP available, 1 otherwise.
    """
    if _OPENMP_AVAILABLE:
        return omp_get_max_threads()
    return 1


def get_num_procs():
    """
    Get the number of processors available to OpenMP.

    Returns:
        int: Number of processors if OpenMP available, 1 otherwise.
    """
    if _OPENMP_AVAILABLE:
        return omp_get_num_procs()
    return 1


# =============================================================================
# Public API - GSL
# =============================================================================

def is_gsl_available():
    """
    Check if GSL support was compiled into the package.

    GSL is required for correct parallel random number generation.
    Without GSL, parallel execution falls back to single-threaded NumPy.

    Returns:
        bool: True if GSL is available, False otherwise.

    Example:
        >>> from cssm._openmp_status import is_gsl_available
        >>> if is_gsl_available():
        ...     print("GSL RNG available for parallel execution!")
        ... else:
        ...     print("Parallel execution will use NumPy (single-threaded)")
    """
    return _GSL_AVAILABLE


# =============================================================================
# Combined Status
# =============================================================================

def get_openmp_info():
    """
    Get detailed information about OpenMP and GSL status and configuration.

    Returns:
        dict: Dictionary with configuration information.

    Example:
        >>> from cssm._openmp_status import get_openmp_info
        >>> info = get_openmp_info()
        >>> print(f"OpenMP available: {info['openmp_available']}")
        >>> print(f"GSL available: {info['gsl_available']}")
        >>> print(f"Parallel ready: {info['parallel_ready']}")
    """
    return {
        "openmp_available": _OPENMP_AVAILABLE,
        "gsl_available": _GSL_AVAILABLE,
        "parallel_ready": _OPENMP_AVAILABLE and _GSL_AVAILABLE,
        "max_threads": get_max_threads(),
        "num_procs": get_num_procs(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
    }


def check_parallel_request(n_threads, warn=True):
    """
    Check if a parallel request can be fulfilled.

    For parallel execution to work correctly, BOTH OpenMP AND GSL must be
    available. If either is missing, this function returns 1 (sequential)
    and optionally warns the user.

    Args:
        n_threads: Requested number of threads
        warn: Whether to emit a warning if falling back to sequential

    Returns:
        int: Actual number of threads to use (may be 1 if requirements not met)

    Example:
        >>> # In a simulator function:
        >>> n_threads = check_parallel_request(n_threads)
        >>> if n_threads > 1:
        ...     # Use parallel path with GSL RNG
        ... else:
        ...     # Use sequential path with NumPy RNG
    """
    if n_threads <= 1:
        return 1

    # Check OpenMP
    if not _OPENMP_AVAILABLE:
        if warn:
            import warnings
            warnings.warn(
                f"Requested n_threads={n_threads} but OpenMP is not available. "
                f"Running with n_threads=1.\n"
                f"To enable parallel support:\n"
                f"  - macOS: brew install libomp && pip install --force-reinstall ssm-simulators\n"
                f"  - Linux: Ensure OpenMP dev packages are installed (libgomp-dev)\n"
                f"  - Conda: conda install -c conda-forge ssm-simulators",
                RuntimeWarning,
                stacklevel=3
            )
        return 1

    # Check GSL (required for correct parallel RNG)
    if not _GSL_AVAILABLE:
        if warn:
            import warnings
            warnings.warn(
                f"Requested n_threads={n_threads} but GSL is not available. "
                f"Running with n_threads=1 (using NumPy RNG).\n"
                f"GSL is required for correct parallel random number generation.\n"
                f"To enable parallel support:\n"
                f"  - macOS: brew install gsl && pip install --force-reinstall ssm-simulators\n"
                f"  - Linux: apt install libgsl-dev && pip install --force-reinstall ssm-simulators\n"
                f"  - Conda: conda install -c conda-forge ssm-simulators (recommended)",
                RuntimeWarning,
                stacklevel=3
            )
        return 1

    return n_threads


def print_status():
    """Print OpenMP and GSL status information to stdout."""
    info = get_openmp_info()

    print("=" * 60)
    print("SSMS Parallel Execution Status")
    print("=" * 60)
    print(f"  OpenMP compiled:    {'Yes' if info['openmp_available'] else 'No'}")
    print(f"  GSL compiled:       {'Yes' if info['gsl_available'] else 'No'}")
    print(f"  Parallel ready:     {'Yes' if info['parallel_ready'] else 'No'}")
    print(f"  Max threads:        {info['max_threads']}")
    print(f"  CPU processors:     {info['num_procs']}")
    print(f"  OMP_NUM_THREADS:    {info['omp_num_threads_env'] or '(not set)'}")
    print(f"  Python cpu_count:   {info['cpu_count']}")
    print("=" * 60)

    if not info['parallel_ready']:
        print("\nTo enable full parallel support:")
        if not info['openmp_available']:
            print("  OpenMP:")
            print("    macOS:  brew install libomp")
            print("    Linux:  apt install libgomp-dev (or equivalent)")
        if not info['gsl_available']:
            print("  GSL:")
            print("    macOS:  brew install gsl")
            print("    Linux:  apt install libgsl-dev")
            print("    Conda:  conda install -c conda-forge gsl")
        print("\n  Then reinstall: pip install --force-reinstall ssm-simulators")
        print("  Or use conda:   conda install -c conda-forge ssm-simulators")


# Run status check if executed directly
if __name__ == "__main__":
    print_status()
