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

Reference:
- GSL Manual: https://www.gnu.org/software/gsl/doc/html/randist.html
"""

from libc.stdint cimport uint64_t

# Raw C function declarations
cdef extern from "gsl_rng.h" nogil:
    # Struct must be declared here (not just in .pxd) for cross-version Cython compatibility
    ctypedef struct ssms_rng_state:
        void* rng

    void ssms_rng_alloc(ssms_rng_state* state)
    void ssms_rng_free(ssms_rng_state* state)
    void ssms_rng_seed(ssms_rng_state* state, uint64_t seed)
    void ssms_rng_init(ssms_rng_state* state, uint64_t seed)
    void ssms_rng_cleanup(ssms_rng_state* state)
    float ssms_rng_gaussian_f32(ssms_rng_state* state)
    float ssms_rng_gaussian_f32_sigma(ssms_rng_state* state, float sigma)
    float ssms_rng_levy_f32(ssms_rng_state* state, float c, float alpha)
    float ssms_rng_gamma_f32(ssms_rng_state* state, float shape, float scale)
    float ssms_rng_uniform_f32(ssms_rng_state* state)
    uint64_t ssms_rng_mix_seed(uint64_t base, uint64_t t1, uint64_t t2)

# =============================================================================
# WRAPPER FUNCTIONS (for internal Cython use)
# =============================================================================
# Note: These use ssms_rng_state directly (not the Xoroshiro128PlusState/RngState
# typedefs from the .pxd) so the .pyx is self-contained and doesn't depend on
# the .pxd being discoverable by all Cython versions and build environments.

cdef void rng_seed(ssms_rng_state* state, uint64_t seed) noexcept nogil:
    """Re-seed an already allocated RNG state."""
    ssms_rng_seed(state, seed)

cdef uint64_t rng_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id) noexcept nogil:
    """Create unique seed for thread/trial combination."""
    return ssms_rng_mix_seed(base_seed, thread_id, trial_id)

cdef float rng_uniform_f32(ssms_rng_state* state) noexcept nogil:
    """Generate uniform random number in (0, 1)."""
    return ssms_rng_uniform_f32(state)

cdef float rng_gaussian_f32(ssms_rng_state* state) noexcept nogil:
    """Generate Gaussian random float."""
    return ssms_rng_gaussian_f32(state)

cdef float rng_levy_f32(ssms_rng_state* state, float alpha) noexcept nogil:
    """Generate alpha-stable random float."""
    return ssms_rng_levy_f32(state, 1.0, alpha)

cdef float rng_gamma_f32(ssms_rng_state* state, float shape, float scale) noexcept nogil:
    """Generate Gamma(shape, scale) random float using GSL."""
    return ssms_rng_gamma_f32(state, shape, scale)

cdef void rng_alloc(ssms_rng_state* state) noexcept nogil:
    """Allocate GSL RNG - call once per thread before parallel block."""
    ssms_rng_alloc(state)

cdef void rng_free(ssms_rng_state* state) noexcept nogil:
    """Free GSL RNG - call once per thread after parallel block."""
    ssms_rng_free(state)

# =============================================================================
# PYTHON-ACCESSIBLE TEST AND VALIDATION FUNCTIONS
# =============================================================================

def test_rng(seed=42, n=1000000):
    """Test the C RNG from Python."""
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    # Generate samples
    cdef float[:] uniforms = np.zeros(n, dtype=np.float32)
    cdef float[:] gaussians = np.zeros(n, dtype=np.float32)

    cdef int i
    for i in range(n):
        uniforms[i] = ssms_rng_uniform_f32(&state)
        gaussians[i] = ssms_rng_gaussian_f32(&state)

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

    cdef float[:] samples = np.zeros(n, dtype=np.float32)
    cdef int i

    for i in range(n):
        samples[i] = ssms_rng_gaussian_f32(&state)

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

    cdef float[:] samples = np.zeros(n, dtype=np.float32)
    cdef int i

    for i in range(n):
        samples[i] = ssms_rng_uniform_f32(&state)

    ssms_rng_cleanup(&state)

    return np.asarray(samples)


def generate_levy_samples(int n, float alpha, uint64_t seed=42):
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

    cdef float[:] samples = np.zeros(n, dtype=np.float32)
    cdef int i

    for i in range(n):
        samples[i] = ssms_rng_levy_f32(&state, 1.0, alpha)

    ssms_rng_cleanup(&state)

    return np.asarray(samples)


def generate_gamma_samples(int n, float shape, float scale, uint64_t seed=42):
    """
    Generate Gamma(shape, scale) samples using GSL.

    Parameters
    ----------
    n : int
        Number of samples to generate
    shape : float
        Shape parameter (must be > 0)
    scale : float
        Scale parameter (must be > 0)
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of Gamma random samples
    """
    import numpy as np

    cdef ssms_rng_state state
    ssms_rng_init(&state, seed)

    cdef float[:] samples = np.zeros(n, dtype=np.float32)
    cdef int i

    for i in range(n):
        samples[i] = ssms_rng_gamma_f32(&state, shape, scale)

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
    return ssms_rng_mix_seed(base_seed, thread_id, trial_id)


# =============================================================================
# TEST ARRAY ALLOCATION (for parallel execution pattern)
# =============================================================================

# Include shared constants (MAX_THREADS, etc.)
include "_constants.pxi"

def test_parallel_alloc(int n_threads=4):
    """
    Test the allocation pattern used in parallel execution.

    This tests:
    1. Allocating an array of RNG states
    2. Seeding each one
    3. Generating samples
    4. Freeing them
    """
    import numpy as np

    cdef ssms_rng_state[MAX_THREADS] rng_states
    cdef int i
    cdef float val

    print(f"Testing parallel allocation pattern with {n_threads} threads...")

    # Step 1: Allocate
    print("  Allocating RNG states...")
    for i in range(n_threads):
        ssms_rng_alloc(&rng_states[i])
        print(f"    Allocated state {i}")

    # Step 2: Seed and generate
    print("  Seeding and generating samples...")
    for i in range(n_threads):
        ssms_rng_seed(&rng_states[i], 42 + i)
        val = ssms_rng_gaussian_f32(&rng_states[i])
        print(f"    State {i}: gaussian = {val:.4f}")

    # Step 3: Free
    print("  Freeing RNG states...")
    for i in range(n_threads):
        ssms_rng_free(&rng_states[i])
        print(f"    Freed state {i}")

    print("SUCCESS: Parallel allocation pattern works!")
    return True
