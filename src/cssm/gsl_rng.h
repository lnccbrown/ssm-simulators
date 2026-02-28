/*
 * GSL-based Random Number Generation for Parallel Execution
 *
 * This header provides thread-safe RNG functions using GSL's validated
 * Ziggurat implementation for Gaussian sampling.
 *
 * USAGE FOR PARALLEL CODE:
 * 1. Declare array: cdef RngState[MAX_THREADS] rng_states
 * 2. Before parallel block: call ssms_rng_alloc(&rng_states[i]) for each thread
 * 3. Inside parallel: use ssms_rng_seed(&rng_states[tid], seed) to re-seed
 * 4. After parallel block: call ssms_rng_free(&rng_states[i]) for each thread
 *
 * Functions:
 * - ssms_rng_alloc: Allocate GSL RNG (call once per thread, before parallel)
 * - ssms_rng_free: Free GSL RNG (call once per thread, after parallel)
 * - ssms_rng_seed: Re-seed an allocated RNG (no allocation, safe in parallel)
 * - ssms_rng_gaussian_f32: Generate Gaussian using GSL Ziggurat (variance = 1.0)
 * - ssms_rng_levy_f32: Generate Levy alpha-stable
 * - ssms_rng_uniform_f32: Generate uniform in (0, 1)
 * - ssms_rng_mix_seed: Mix seed with thread/trial IDs
 */

#ifndef SSMS_GSL_RNG_H
#define SSMS_GSL_RNG_H

#include <stdint.h>
#include <stdlib.h>

#ifdef HAVE_GSL

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*
 * RNG state wrapper - contains pointer to GSL RNG.
 * This struct has KNOWN SIZE (just a pointer) so Cython can allocate arrays of it.
 */
typedef struct {
    gsl_rng *rng;
} ssms_rng_state;

/* Allocate GSL RNG - call ONCE per thread BEFORE parallel block */
static inline void ssms_rng_alloc(ssms_rng_state* state) {
    state->rng = gsl_rng_alloc(gsl_rng_mt19937);
}

/* Free GSL RNG - call ONCE per thread AFTER parallel block */
static inline void ssms_rng_free(ssms_rng_state* state) {
    if (state->rng != NULL) {
        gsl_rng_free(state->rng);
        state->rng = NULL;
    }
}

/* Re-seed an already allocated RNG - safe to call in parallel (no allocation) */
static inline void ssms_rng_seed(ssms_rng_state* state, uint64_t seed) {
    gsl_rng_set(state->rng, (unsigned long)seed);
}

/* Legacy init function - allocates AND seeds (for backward compat, avoid in loops) */
static inline void ssms_rng_init(ssms_rng_state* state, uint64_t seed) {
    state->rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(state->rng, (unsigned long)seed);
}

/* Legacy cleanup function */
static inline void ssms_rng_cleanup(ssms_rng_state* state) {
    ssms_rng_free(state);
}

/* Generate Gaussian using GSL's validated Ziggurat - variance = 1.0 */
static inline float ssms_rng_gaussian_f32(ssms_rng_state* state) {
    return (float)gsl_ran_gaussian_ziggurat(state->rng, 1.0);
}

/* Generate Gaussian with given sigma */
static inline float ssms_rng_gaussian_f32_sigma(ssms_rng_state* state, float sigma) {
    return (float)gsl_ran_gaussian_ziggurat(state->rng, (double)sigma);
}

/* Generate Levy alpha-stable using GSL */
static inline float ssms_rng_levy_f32(ssms_rng_state* state, float c, float alpha) {
    return (float)gsl_ran_levy(state->rng, (double)c, (double)alpha);
}

/* Generate Gamma(shape, scale) variate using GSL */
static inline float ssms_rng_gamma_f32(ssms_rng_state* state, float shape, float scale) {
    return (float)gsl_ran_gamma(state->rng, (double)shape, (double)scale);
}

/* Generate uniform in (0, 1) */
static inline float ssms_rng_uniform_f32(ssms_rng_state* state) {
    return (float)gsl_rng_uniform_pos(state->rng);
}

/* Mix seed with thread/trial IDs for independent streams */
static inline uint64_t ssms_rng_mix_seed(uint64_t base, uint64_t t1, uint64_t t2) {
    uint64_t z = base + t1 * 0x9E3779B97F4A7C15ULL + t2 * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

#else /* !HAVE_GSL */

/* Stub implementations for compilation without GSL.
 * These are never called at runtime — the simulator checks is_gsl_available()
 * before entering any parallel path that uses per-thread RNG states.
 * They exist solely to allow compilation on systems without GSL installed.
 */
typedef struct { void* rng; } ssms_rng_state;

static inline void ssms_rng_alloc(ssms_rng_state* state) { (void)state; }
static inline void ssms_rng_free(ssms_rng_state* state) { (void)state; }
static inline void ssms_rng_seed(ssms_rng_state* state, uint64_t seed) {
    (void)state; (void)seed;
}
static inline void ssms_rng_init(ssms_rng_state* state, uint64_t seed) {
    (void)state; (void)seed;
}
static inline void ssms_rng_cleanup(ssms_rng_state* state) { (void)state; }
static inline float ssms_rng_gaussian_f32(ssms_rng_state* state) {
    (void)state; return 0.0f;
}
static inline float ssms_rng_gaussian_f32_sigma(ssms_rng_state* state, float sigma) {
    (void)state; (void)sigma; return 0.0f;
}
static inline float ssms_rng_levy_f32(ssms_rng_state* state, float c, float alpha) {
    (void)state; (void)c; (void)alpha; return 0.0f;
}
static inline float ssms_rng_gamma_f32(ssms_rng_state* state, float shape, float scale) {
    (void)state; (void)shape; (void)scale; return 1.0f;
}
static inline float ssms_rng_uniform_f32(ssms_rng_state* state) {
    (void)state; return 0.5f;
}
static inline uint64_t ssms_rng_mix_seed(uint64_t base, uint64_t t1, uint64_t t2) {
    (void)t1; (void)t2; return base;
}

#endif /* HAVE_GSL */

#endif /* SSMS_GSL_RNG_H */
