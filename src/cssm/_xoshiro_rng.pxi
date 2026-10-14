# _xoshiro_rng.pxi - Shared GSL-free generator for prange kernels
#
# Textually included by Cython .pyx files with: include "_xoshiro_rng.pxi"
#
# xoshiro256++ seeded per trial through SplitMix64, with a Box-Muller Gaussian
# transform.  Vendored verbatim from efficient-fpt @ d97a451
# (src/efficient_fpt/cython/simulator.pyx), MIT (c) 2025 Sicheng Liu; see
# ADDM_ENGINE_NOTICE.md and LICENSE.efpt in this directory.
#
# Why this exists next to the GSL generator in _rng_wrappers.pxi: that one is
# live only on GSL builds and only inside the parallel path (PyPI wheels build
# without GSL, and its stubs return 0.0), so kernels using it keep a second,
# sequential NumPy path.  This generator needs neither GSL nor the GIL, so one
# kernel body can run under prange on every build, and seeding one state per
# (sample, trial) makes the output independent of n_threads and bit-identical
# to efficient-fpt's own simulator on the same seeds.
#
# Usage inside a nogil trial function:
#     cdef Xoshiro256State rng
#     cdef BoxMullerState bm
#     seed_xoshiro256(&rng, seed)
#     bm.has_spare = 0
#     z = box_muller_next(&rng, &bm)   # one standard normal
#
# Do not change the arithmetic: tests/test_addm_simulator.py pins the stream.

from libc.math cimport sqrt, log, cos, sin, M_PI
from libc.stdint cimport uint64_t

cdef int UNIFORM_BITS = 11                               # keep the top 53 bits
cdef double UINT64_TO_DOUBLE = 1.0 / 9007199254740992.0  # 2^-53
cdef double MIN_UNIFORM = 1e-300                         # keeps log(u1) finite
cdef double TWO_PI = 2.0 * M_PI


cdef struct Xoshiro256State:
    uint64_t s0
    uint64_t s1
    uint64_t s2
    uint64_t s3


cdef inline uint64_t _rotl(uint64_t x, int k) noexcept nogil:
    return (x << k) | (x >> (64 - k))


cdef inline uint64_t xoshiro256pp_next(Xoshiro256State *state) noexcept nogil:
    cdef uint64_t result = _rotl(state.s0 + state.s3, 23) + state.s0
    cdef uint64_t t = state.s1 << 17
    state.s2 ^= state.s0
    state.s3 ^= state.s1
    state.s1 ^= state.s2
    state.s0 ^= state.s3
    state.s2 ^= t
    state.s3 = _rotl(state.s3, 45)
    return result


cdef inline uint64_t splitmix64_next(uint64_t *state) noexcept nogil:
    state[0] += <uint64_t>0x9e3779b97f4a7c15
    cdef uint64_t z = state[0]
    z = (z ^ (z >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * <uint64_t>0x94d049bb133111eb
    return z ^ (z >> 31)


cdef inline void seed_xoshiro256(Xoshiro256State *state, uint64_t seed) noexcept nogil:
    cdef uint64_t sm_state = seed
    state.s0 = splitmix64_next(&sm_state)
    state.s1 = splitmix64_next(&sm_state)
    state.s2 = splitmix64_next(&sm_state)
    state.s3 = splitmix64_next(&sm_state)


cdef inline double uint64_to_double(uint64_t x) noexcept nogil:
    """Uniform in [0, 1) from the top 53 bits of ``x``."""
    return <double>(x >> UNIFORM_BITS) * UINT64_TO_DOUBLE


cdef struct BoxMullerState:
    # Box-Muller produces two normals from one pair of uniforms. ``spare``
    # keeps the second one and ``has_spare`` marks it for the next call.
    double spare
    int has_spare


cdef inline double box_muller_next(Xoshiro256State *rng_state, BoxMullerState *bm_state) noexcept nogil:
    """Return one standard normal, caching its Box-Muller companion."""
    cdef double u1, u2, mag
    if bm_state.has_spare:
        bm_state.has_spare = 0
        return bm_state.spare
    u1 = uint64_to_double(xoshiro256pp_next(rng_state))
    u2 = uint64_to_double(xoshiro256pp_next(rng_state))
    if u1 < MIN_UNIFORM:
        u1 = MIN_UNIFORM
    mag = sqrt(-2.0 * log(u1))
    bm_state.spare = mag * sin(TWO_PI * u2)
    bm_state.has_spare = 1
    return mag * cos(TWO_PI * u2)
