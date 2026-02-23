# cython: language_level=3
"""
Pure C Random Number Generator - usable with nogil

Header file declaring the C-level RNG interface.
"""

from libc.stdint cimport uint64_t

# RNG state structure
ctypedef struct Xoroshiro128PlusState:
    uint64_t s0
    uint64_t s1

# Function declarations
cdef void rng_seed(Xoroshiro128PlusState* state, uint64_t seed) noexcept nogil
cdef uint64_t rng_mix_seed(uint64_t base_seed, uint64_t thread_id, uint64_t trial_id) noexcept nogil
cdef double rng_uniform(Xoroshiro128PlusState* state) noexcept nogil
cdef float rng_uniform_f32(Xoroshiro128PlusState* state) noexcept nogil
cdef double rng_gaussian(Xoroshiro128PlusState* state) noexcept nogil
cdef float rng_gaussian_f32(Xoroshiro128PlusState* state) noexcept nogil

# Alpha-stable (Lévy) random variates using Chambers-Mallows-Stuck algorithm
cdef double rng_levy(Xoroshiro128PlusState* state, double alpha) noexcept nogil
cdef float rng_levy_f32(Xoroshiro128PlusState* state, float alpha) noexcept nogil
