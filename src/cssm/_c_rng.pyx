# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
C-Level Random Number Generator - GIL-free implementation

This module provides high-quality random number generation that can be called
from nogil Cython code, enabling true parallel execution.

Uses GSL's validated implementations:
- gsl_ran_gaussian_ziggurat: Fast Gaussian sampling (variance = 1.0)
- gsl_ran_levy: Alpha-stable (Lévy) distributions

When GSL is not available, provides stub implementations for testing.

Reference:
- GSL Manual: https://www.gnu.org/software/gsl/doc/html/randist.html
"""

from libc.stdint cimport uint64_t

# Use the GSL header file implementation
cdef extern from "gsl_rng.h" nogil:
    ctypedef struct ssms_rng_state:
        pass

    void ssms_rng_init(ssms_rng_state* state, uint64_t seed)
    void ssms_rng_cleanup(ssms_rng_state* state)
    float ssms_gaussian_f32(ssms_rng_state* state)
    float ssms_gaussian_f32_sigma(ssms_rng_state* state, float sigma)
    float ssms_levy_f32(ssms_rng_state* state, float c, float alpha)
    double ssms_uniform(ssms_rng_state* state)
    uint64_t ssms_mix_seed(uint64_t base, uint64_t t1, uint64_t t2)

# Type alias for backward compatibility
ctypedef ssms_rng_state Xoroshiro128PlusState

# =============================================================================
# WRAPPER FUNCTIONS (for internal Cython use)
# =============================================================================

cdef void rng_seed(Xoroshiro128PlusState* state, uint64_t seed) noexcept nogil:
    """Initialize RNG state from a seed."""
    ssms_rng_init(state, seed)

cdef uint64_t rng_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id) noexcept nogil:
    """Create unique seed for thread/trial combination."""
    return ssms_mix_seed(base_seed, thread_id, trial_id)

cdef double rng_uniform(Xoroshiro128PlusState* state) noexcept nogil:
    """Generate uniform random number in (0, 1)."""
    return ssms_uniform(state)

cdef double rng_gaussian(Xoroshiro128PlusState* state) noexcept nogil:
    """Generate Gaussian random number using GSL Ziggurat algorithm."""
    return <double>ssms_gaussian_f32(state)

cdef float rng_gaussian_f32(Xoroshiro128PlusState* state) noexcept nogil:
    """Generate Gaussian random float."""
    return ssms_gaussian_f32(state)

cdef double rng_levy(Xoroshiro128PlusState* state, double alpha) noexcept nogil:
    """Generate alpha-stable (Lévy) random variate using GSL."""
    return <double>ssms_levy_f32(state, 1.0, <float>alpha)

cdef float rng_levy_f32(Xoroshiro128PlusState* state, float alpha) noexcept nogil:
    """Generate alpha-stable random float."""
    return ssms_levy_f32(state, 1.0, alpha)

# =============================================================================
# PYTHON-ACCESSIBLE TEST AND VALIDATION FUNCTIONS
# =============================================================================

def test_rng(seed=42, n=1000000):
    """Test the C RNG from Python."""
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    # Generate samples
    cdef double[:] uniforms = np.zeros(n, dtype=np.float64)
    cdef double[:] gaussians = np.zeros(n, dtype=np.float64)

    cdef int i
    for i in range(n):
        uniforms[i] = ssms_uniform(&state)
        gaussians[i] = <double>ssms_gaussian_f32(&state)

    ssms_rng_cleanup(&state)

    uniforms_np = np.asarray(uniforms)
    gaussians_np = np.asarray(gaussians)

    print(f"Uniform:  mean={uniforms_np.mean():.4f} (expect 0.5), std={uniforms_np.std():.4f} (expect 0.289)")
    print(f"Gaussian: mean={gaussians_np.mean():.4f} (expect 0.0), std={gaussians_np.std():.4f} (expect 1.0)")

    return uniforms_np, gaussians_np


def generate_gaussian_samples(int n, uint64_t seed=42):
    """
    Generate Gaussian samples using GSL's Ziggurat algorithm.

    Parameters
    ----------
    n : int
        Number of samples to generate
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of Gaussian random samples (mean=0, std=1)
    """
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    cdef double[:] samples = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        samples[i] = <double>ssms_gaussian_f32(&state)

    ssms_rng_cleanup(&state)

    return np.asarray(samples)


def generate_uniform_samples(int n, uint64_t seed=42):
    """
    Generate uniform (0, 1) samples using GSL.

    Parameters
    ----------
    n : int
        Number of samples to generate
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of uniform random samples in (0, 1)
    """
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    cdef double[:] samples = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        samples[i] = ssms_uniform(&state)

    ssms_rng_cleanup(&state)

    return np.asarray(samples)


def generate_levy_samples(int n, double alpha, uint64_t seed=42):
    """
    Generate alpha-stable (Lévy) samples using GSL.

    Parameters
    ----------
    n : int
        Number of samples to generate
    alpha : float
        Stability parameter (0 < alpha <= 2)
        alpha=2: Gaussian (variance=2), alpha=1: Cauchy
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of alpha-stable random samples
    """
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    cdef double[:] samples = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        samples[i] = <double>ssms_levy_f32(&state, 1.0, <float>alpha)

    ssms_rng_cleanup(&state)

    return np.asarray(samples)


def py_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id):
    """
    Python wrapper for seed mixing function.

    Parameters
    ----------
    base_seed : int
        Base random seed
    thread_id : int
        Thread identifier
    trial_id : int
        Trial/sample identifier

    Returns
    -------
    int
        Mixed seed value
    """
    return ssms_mix_seed(base_seed, thread_id, trial_id)
