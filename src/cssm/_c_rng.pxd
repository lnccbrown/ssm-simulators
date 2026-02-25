# cython: language_level=3
"""
C-Level Random Number Generator - GIL-free implementation

Header file declaring the C-level RNG interface backed by GSL.
"""

from libc.stdint cimport uint64_t

# RNG state structure - wraps GSL's gsl_rng pointer
cdef extern from "gsl_rng.h" nogil:
    ctypedef struct ssms_rng_state:
        void* rng

# Type aliases for backward compatibility and clarity
ctypedef ssms_rng_state Xoroshiro128PlusState
ctypedef ssms_rng_state RngState

# Function declarations
cdef void rng_seed(Xoroshiro128PlusState* state, uint64_t seed) noexcept nogil
cdef uint64_t rng_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id) noexcept nogil
cdef double rng_uniform(Xoroshiro128PlusState* state) noexcept nogil
cdef double rng_gaussian(Xoroshiro128PlusState* state) noexcept nogil
cdef float rng_gaussian_f32(Xoroshiro128PlusState* state) noexcept nogil

# Alpha-stable (Lévy) random variates
cdef double rng_levy(Xoroshiro128PlusState* state, double alpha) noexcept nogil
cdef float rng_levy_f32(Xoroshiro128PlusState* state, float alpha) noexcept nogil

# Gamma distribution
cdef float rng_gamma_f32(Xoroshiro128PlusState* state, float shape, float scale) noexcept nogil

# Allocation / deallocation
cdef void rng_alloc(RngState* state) noexcept nogil
cdef void rng_free(RngState* state) noexcept nogil
