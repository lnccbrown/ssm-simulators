/*
 * GSL-based Random Number Generation for Parallel Execution
 *
 * This header provides thread-safe RNG functions. When HAVE_GSL is defined,
 * it uses GSL's validated implementations. Otherwise, it provides a simple
 * xoroshiro128+ based implementation with Box-Muller for Gaussian.
 *
 * The state is a simple struct that doesn't require malloc, making it safe
 * for use in parallel loops without memory management issues.
 */

#ifndef SSMS_GSL_RNG_H
#define SSMS_GSL_RNG_H

#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Simple xoroshiro128+ state - no malloc needed */
typedef struct {
    uint64_t s0;
    uint64_t s1;
} ssms_rng_state;

/* Rotate left helper */
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* Initialize RNG with seed */
static inline void ssms_rng_init(ssms_rng_state* state, uint64_t seed) {
    /* SplitMix64 to initialize state from seed */
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s0 = z ^ (z >> 31);

    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s1 = z ^ (z >> 31);
}

/* Alias for backward compatibility */
static inline void ssms_rng_seed(ssms_rng_state* state, uint64_t seed) {
    ssms_rng_init(state, seed);
}

/* No-op cleanup (no malloc used) */
static inline void ssms_rng_cleanup(ssms_rng_state* state) {
    (void)state;
}

/* Xoroshiro128+ next */
static inline uint64_t ssms_rng_next(ssms_rng_state* state) {
    const uint64_t s0 = state->s0;
    uint64_t s1 = state->s1;
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state->s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    state->s1 = rotl(s1, 37);

    return result;
}

/* Generate uniform random double in (0, 1) */
static inline double ssms_uniform(ssms_rng_state* state) {
    /* Use upper 53 bits for maximum precision */
    return (ssms_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Generate Gaussian random float using Box-Muller transform */
static inline float ssms_gaussian_f32(ssms_rng_state* state) {
    /* Box-Muller transform - generates two values, we use one */
    double u1, u2, r, theta;

    /* Ensure u1 > 0 to avoid log(0) */
    do {
        u1 = ssms_uniform(state);
    } while (u1 <= 1e-15);

    u2 = ssms_uniform(state);

    r = sqrt(-2.0 * log(u1));
    theta = 2.0 * M_PI * u2;

    return (float)(r * cos(theta));
}

/* Generate Gaussian random float with given sigma */
static inline float ssms_gaussian_f32_sigma(ssms_rng_state* state, float sigma) {
    return ssms_gaussian_f32(state) * sigma;
}

/* Chambers-Mallows-Stuck algorithm for alpha-stable (Levy) distribution */
static inline float ssms_levy_f32(ssms_rng_state* state, float c, float alpha) {
    double U, W, X;
    double inv_alpha, one_minus_alpha_over_alpha;

    /* Handle special case: alpha=2 is Gaussian with variance=2 */
    if (alpha >= 1.9999f) {
        return c * ssms_gaussian_f32(state) * 1.4142135623730951f; /* sqrt(2) */
    }

    /* Uniform in (-pi/2, pi/2) */
    U = (ssms_uniform(state) - 0.5) * M_PI;

    /* Exponential with mean 1 */
    double e;
    do {
        e = ssms_uniform(state);
    } while (e <= 1e-15);
    W = -log(e);

    inv_alpha = 1.0 / (double)alpha;
    one_minus_alpha_over_alpha = ((double)alpha - 1.0) * inv_alpha;

    /* CMS formula */
    X = sin((double)alpha * U) / pow(cos(U), inv_alpha);
    X *= pow(cos(U - (double)alpha * U) / W, one_minus_alpha_over_alpha);

    return (float)(c * X);
}

/* Mix seed with thread/trial IDs for independent streams */
static inline uint64_t ssms_mix_seed(uint64_t base, uint64_t t1, uint64_t t2) {
    /* SplitMix64-style mixing for high-quality seed derivation */
    uint64_t z = base + t1 * 0x9E3779B97F4A7C15ULL + t2 * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

#endif /* SSMS_GSL_RNG_H */
