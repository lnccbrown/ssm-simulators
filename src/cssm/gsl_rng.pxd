# cython: language_level=3
"""
GSL Random Number Generation Interface for Cython

This provides access to GSL's verified implementations of:
- Gaussian (Ziggurat method)
- Lévy / Alpha-Stable (Chambers-Mallows-Stuck method)
- Many other distributions

All functions are nogil-safe for use in parallel code.

Usage:
    brew install gsl  # or apt-get install libgsl-dev

    Then link with: -lgsl -lgslcblas
"""

from libc.stdint cimport uint64_t

# =============================================================================
# GSL RNG Types and Functions
# =============================================================================

cdef extern from "gsl/gsl_rng.h" nogil:
    # RNG type definitions
    ctypedef struct gsl_rng_type:
        const char* name
        unsigned long int max
        unsigned long int min
        size_t size

    ctypedef struct gsl_rng:
        const gsl_rng_type* type
        void* state

    # Available RNG algorithms (all high-quality)
    const gsl_rng_type* gsl_rng_mt19937      # Mersenne Twister
    const gsl_rng_type* gsl_rng_taus2        # Tausworthe (fast)
    const gsl_rng_type* gsl_rng_ranlxd2      # RANLUX (highest quality)

    # RNG lifecycle
    gsl_rng* gsl_rng_alloc(const gsl_rng_type* T)
    void gsl_rng_set(gsl_rng* r, unsigned long int seed)
    void gsl_rng_free(gsl_rng* r)

    # Basic uniform generation
    unsigned long int gsl_rng_get(const gsl_rng* r)
    double gsl_rng_uniform(const gsl_rng* r)
    double gsl_rng_uniform_pos(const gsl_rng* r)  # Excludes 0


# =============================================================================
# GSL Random Distributions
# =============================================================================

cdef extern from "gsl/gsl_randist.h" nogil:
    # --- GAUSSIAN (Normal) Distribution ---
    # Uses verified Ziggurat algorithm
    double gsl_ran_gaussian(const gsl_rng* r, double sigma)
    double gsl_ran_gaussian_ziggurat(const gsl_rng* r, double sigma)
    double gsl_ran_ugaussian(const gsl_rng* r)  # Standard normal (mu=0, sigma=1)
    double gsl_ran_ugaussian_ratio_method(const gsl_rng* r)

    # --- LÉVY / ALPHA-STABLE Distribution ---
    # Uses Chambers-Mallows-Stuck algorithm
    #
    # For alpha in (0, 2]:
    #   alpha = 2: Gaussian
    #   alpha = 1: Cauchy
    #   alpha = 0.5: Lévy distribution
    #
    # Parameters:
    #   c: scale parameter
    #   alpha: stability parameter (0 < alpha <= 2)
    #   beta: skewness parameter (-1 <= beta <= 1)
    #
    double gsl_ran_levy(const gsl_rng* r, double c, double alpha)
    double gsl_ran_levy_skew(const gsl_rng* r, double c, double alpha, double beta)

    # --- Other useful distributions ---
    double gsl_ran_exponential(const gsl_rng* r, double mu)
    double gsl_ran_cauchy(const gsl_rng* r, double a)
    double gsl_ran_flat(const gsl_rng* r, double a, double b)  # Uniform [a, b)
