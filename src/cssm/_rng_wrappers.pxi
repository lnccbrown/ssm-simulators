# _rng_wrappers.pxi - Shared RNG declarations and inline wrappers
# This file is textually included by Cython .pyx files using: include "_rng_wrappers.pxi"

from libc.stdint cimport uint64_t

cdef extern from "gsl_rng.h" nogil:
    ctypedef struct ssms_rng_state:
        void* rng

    void ssms_rng_alloc(ssms_rng_state* state)
    void ssms_rng_free(ssms_rng_state* state)
    void ssms_rng_seed(ssms_rng_state* state, uint64_t seed)
    float ssms_rng_gaussian_f32(ssms_rng_state* state)
    float ssms_rng_levy_f32(ssms_rng_state* state, float c, float alpha)
    float ssms_rng_gamma_f32(ssms_rng_state* state, float shape, float scale)
    float ssms_rng_uniform_f32(ssms_rng_state* state)
    uint64_t ssms_rng_mix_seed(uint64_t base, uint64_t t1, uint64_t t2)

ctypedef ssms_rng_state RngState

cdef inline void rng_alloc(RngState* state) noexcept nogil:
    ssms_rng_alloc(state)

cdef inline void rng_free(RngState* state) noexcept nogil:
    ssms_rng_free(state)

cdef inline void rng_seed(RngState* state, uint64_t seed) noexcept nogil:
    ssms_rng_seed(state, seed)

cdef inline uint64_t rng_mix_seed(uint64_t base, uint64_t t, uint64_t n) noexcept nogil:
    return ssms_rng_mix_seed(base, t, n)

cdef inline float rng_gaussian_f32(RngState* state) noexcept nogil:
    return ssms_rng_gaussian_f32(state)

cdef inline float rng_uniform_f32(RngState* state) noexcept nogil:
    return ssms_rng_uniform_f32(state)

cdef inline float rng_levy_f32(RngState* state, float alpha) noexcept nogil:
    return ssms_rng_levy_f32(state, 1.0, alpha)

cdef inline float rng_gamma_f32(RngState* state, float shape, float scale) noexcept nogil:
    return ssms_rng_gamma_f32(state, shape, scale)
