# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Shared utility functions for CSSM simulators.

Random-number generation — design notes
----------------------------------------
The sequential simulation path (n_threads=1) uses three distinct RNG mechanisms.
Understanding why each was chosen helps future developers make informed changes.

**1. Gaussian draws — `draw_gaussian(n)`**
   Uses NumPy's ``standard_normal`` (Ziggurat algorithm) via a module-level
   cached ``numpy.Generator`` (``_global_rng``).  Benchmarks show all three
   alternatives (NumPy, GSL Ziggurat, hand-rolled Marsaglia polar) are within
   ~50 % of each other at ~180–270 M samples/s; NumPy is the obvious choice
   because it is battle-tested and respects the ``random_state`` seed.

**2. Uniform draws — `draw_uniform(n)`**
   Generates batched uniform(0,1) samples using ``_global_rng.uniform()``
   (NumPy PCG64), the same Generator as ``draw_gaussian``.  Follows the
   identical batch-refill pattern: pre-generate ``num_draws`` values, consume
   with an index ``mu``, refill when exhausted.  Used in:
     - DDM ``st``/``sz`` parameter variability  (2 calls per trial)
     - ``smooth_u`` end-of-walk RT jitter        (via ``compute_smooth_unif``)
     - Sequential probabilistic choices          (2–4 calls per sample)
   Performance: ~236 M/s batched vs ~20 M/s for the old scalar C stdlib
   ``random_uniform()`` (11.8× faster per draw); absolute saving ~1.9 ms per
   100k calls.  Primary benefit is reproducibility: uniform draws now share
   the NumPy seed path set by ``set_seed()``.

   ``random_uniform()`` (C stdlib ``rand()``) is retained only for scalar
   helpers such as ``random_exponential()`` and non-parallel call sites.
   The OpenMP parallel paths now use per-thread GSL uniforms via
   ``rng_uniform_f32()``.

**3. Alpha-stable draws — `draw_random_stable(n, alpha)`**
   Implements the Chambers–Mallows–Stuck (CMS) algorithm using vectorised
   NumPy operations.  Three alternatives were benchmarked:

   +--------------------------+--------------------+------------------+
   | Method                   | Throughput (M/s)   | Notes            |
   +==========================+====================+==================+
   | SciPy ``levy_stable``    | ~12–23             | Slowest: ~6.7 ms |
   |                          |                    | of Python         |
   |                          |                    | dispatch overhead|
   |                          |                    | per 100k samples |
   +--------------------------+--------------------+------------------+
   | Hand-rolled CMS (this)   | ~37–62             | Fastest or tied; |
   |                          |                    | full control over|
   |                          |                    | numerical guards |
   +--------------------------+--------------------+------------------+
   | GSL ``ssms_rng_levy_f32``    | ~21–64             | Used in parallel |
   |                          |                    | path; comparable |
   |                          |                    | at most alpha    |
   +--------------------------+--------------------+------------------+

   SciPy's ``levy_stable.rvs`` runs the same CMS math but goes through the
   ``rv_continuous`` Python dispatch layer (parameter validation, broadcasting,
   multiple parameterisation handling), adding ~6.7 ms of overhead above the
   raw NumPy operations — making it 3–7× slower overall.  The hand-rolled
   version is kept because it is the fastest option and SciPy's overhead is
   not justified here.  See ``draw_random_stable_scipy`` for the SciPy
   alternative that is available if correctness verification is needed.
"""

import cython
from libc.stdlib cimport rand, RAND_MAX, srand

from libc.math cimport (
    log,
    sqrt,
    pow,
    fmax,
    atan,
    sin,
    cos,
    tan,
    M_PI,
    M_PI_2
)

from libc.time cimport time

import numpy as np
cimport numpy as np
import numbers

DTYPE = np.float32

cdef object _global_rng = None

cpdef void set_seed(random_state):
    """
    Set the random seed for the simulation.

    Args:
        random_state: An integer seed or None. If None, the current time is used as seed.

    This function sets a random state globally for the simulation.
    """

    cdef long seed_value
    global _global_rng

    if random_state is None:
        srand(time(NULL))
        _global_rng = np.random.default_rng()
        return

    if isinstance(random_state, numbers.Integral):
        seed_value = <long>random_state
    else:
        try:
            seed_value = <long>int(random_state)
        except (TypeError, ValueError):
            raise ValueError("random_state must be an integer or None")

    srand(seed_value)
    _global_rng = np.random.default_rng(seed_value)

# Method to draw random samples from a gaussian
cpdef float random_uniform():
    """
    Generate a single uniform float in [0, 1) using C stdlib ``rand()``.

    This is a *scalar*, one-at-a-time call.  It is seeded separately from
    ``_global_rng`` via ``srand()`` inside ``set_seed()``, so it does not
    share the NumPy seed path used by ``draw_gaussian`` and
    ``draw_random_stable``.

    Performance: ~20 M/s (scalar call overhead from Python dispatch).
    Batched NumPy ``uniform`` achieves ~236 M/s.  The per-run overhead is
    acceptable (~3–5 ms for ~100k calls) but full seed unification would
    require switching to pre-generated NumPy batches.

    Returns:
        float: A random float in [0, 1).
    """
    cdef float r = rand()
    return r / RAND_MAX

cpdef float random_exponential():
    """
    Generate a random float from an exponential distribution with rate 1.

    Returns:
        float: A random float from an exponential distribution.
    """
    return - log(random_uniform())

# OLD IMPLEMENTATION (C-based random_stable) - Kept for reference
# cpdef float random_stable(float alpha):
#     """
#     Generate a random float from a stable distribution.
#     DEPRECATED: This C-based implementation is ~1.8x slower than the NumPy version.
#
#     Args:
#         alpha (float): The stability parameter of the distribution.
#
#     Returns:
#         float: A random float from a stable distribution.
#     """
#     cdef float eta, u, w, x
#
#     u = M_PI * (random_uniform() - 0.5)
#     w = random_exponential()
#
#     if alpha == 1.0:
#         eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
#         x = (1.0 / eta) * ((M_PI_2) * tan(u))
#     else:
#         x = (sin(alpha * u) / (pow(cos(u), 1 / alpha))) * pow(cos(u - (alpha * u)) / w, (1.0 - alpha) / alpha)
#     return x
#
# cpdef float[:] draw_random_stable(int n, float alpha):
#     """
#     Generate an array of random floats from a stable distribution.
#     DEPRECATED: This C-based implementation is ~1.8x slower than the NumPy version.
#
#     Args:
#         n (int): The number of random floats to generate.
#         alpha (float): The stability parameter of the distribution.
#
#     Returns:
#         float[:]: An array of random floats from a stable distribution.
#     """
#     cdef int i
#     cdef float[:] result = np.zeros(n, dtype = DTYPE)
#
#     for i in range(n):
#         result[i] = random_stable(alpha)
#     return result

# OLD IMPLEMENTATION (Marsaglia polar method) - Kept for reference
# cpdef float random_gaussian():
#     """
#     Generate a random float from a standard normal distribution.
#     DEPRECATED: This Marsaglia polar method is ~2.4x slower than NumPy's Ziggurat.
#
#     Returns:
#         float: A random float from a standard normal distribution.
#     """
#     cdef float x1, x2, w
#     w = 2.0
#
#     while(w >= 1.0):
#         x1 = 2.0 * random_uniform() - 1.0
#         x2 = 2.0 * random_uniform() - 1.0
#         w = x1 * x1 + x2 * x2
#
#     w = ((-2.0 * log(w)) / w) ** 0.5
#     return x1 * w

cpdef float[:] draw_gaussian(int n):
    """
    Generate Gaussian samples using NumPy's Ziggurat algorithm.
    ~2.4x faster than the old Marsaglia polar method.

    Args:
        n (int): The number of random floats to generate.

    Returns:
        float[:]: An array of random floats from a standard normal distribution.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()

    # Generate with NumPy's fast Ziggurat, return as memoryview
    cdef np.ndarray[float, ndim=1] samples = _global_rng.standard_normal(
        n, dtype=np.float32
    )
    return samples

cpdef float[:] draw_uniform(int n):
    """
    Generate uniform(0, 1) samples using ``_global_rng`` (NumPy PCG64).

    Returns a ``float32`` memoryview for zero-copy use in Cython hot loops.
    Follows the same batch-refill pattern as ``draw_gaussian``: callers
    pre-generate ``num_draws`` values, consume with an index ``mu``, and
    call this function again when exhausted.

    **NOT thread-safe**: do not call from parallel OpenMP threads.
    For parallel contexts, use ``gsl_rng_uniform(per_thread_state)`` instead.
    The 4 boundary-decision coin flips in ``parallel_models.pyx`` retain
    C stdlib ``random_uniform()`` for this reason.

    Performance: ~236 M/s (11.8× faster than scalar ``random_uniform()``
    at ~20 M/s); absolute saving ~1.9 ms per 100 k calls.
    Primary benefit is reproducibility: shares the NumPy Generator seed
    set by ``set_seed()``, eliminating the separate ``srand()`` path.

    Args:
        n (int): Number of samples to draw.

    Returns:
        float[:]: Memoryview of ``n`` float32 uniform samples in [0, 1).
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()

    cdef np.ndarray[float, ndim=1] samples = np.asarray(
        _global_rng.uniform(0.0, 1.0, n), dtype=np.float32
    )
    return samples

cpdef float[:] draw_random_stable_scipy(int n, float alpha):
    """
    Generate alpha-stable variates via ``scipy.stats.levy_stable``.

    This function exists as a reference implementation for correctness
    cross-checks against ``draw_random_stable``.  It is NOT used in the
    production simulation path because benchmarks show it is 3–7× slower
    than the hand-rolled CMS implementation (``draw_random_stable``):

    * SciPy runs the same CMS algorithm internally but goes through the
      ``rv_continuous`` Python dispatch layer (parameter validation,
      broadcasting, multi-parameterisation handling), adding ~6.7 ms of
      fixed overhead per 100k samples on top of the ~2.5 ms of raw math.
    * Throughput: ~12–23 M/s vs ~37–62 M/s for the hand-rolled version.

    When to prefer this function
    ----------------------------
    * Validating that ``draw_random_stable`` produces the correct distribution
      (compare moments, KS test).
    * One-off analysis where convenience matters more than speed.
    * If SciPy ever inlines its CMS path and removes the dispatch overhead,
      this becomes the preferred implementation — replace the import in
      ``levy_models.pyx`` and delete ``draw_random_stable``.

    Note: ``scipy`` is already a hard dependency of ``ssm-simulators``, so
    there is no added dependency cost from using this function.
    """
    from scipy.stats import levy_stable

    # SciPy uses different parameterization:
    # beta=0 for symmetric, scale and loc for standardization
    cdef np.ndarray[float, ndim=1] samples = levy_stable.rvs(
        alpha=alpha,
        beta=0, # symmetric
        loc=0, # location
        scale=1, # scale
        size=n,
        random_state=_global_rng
    ).astype(np.float32)

    return samples

# cpdef float[:] draw_random_stable_numpy(int n, float alpha):
#     """
#     Generate alpha-stable random variates using NumPy's fast RNG.
#     ~2-3x faster than the Marsaglia-based version.
#
#     Args:
#         n (int): Number of samples to generate
#         alpha (float): Stability parameter (0 < alpha <= 2)
#
#     Returns:
#         float[:]: Array of stable random variates
#     """
#     global _global_rng
#     if _global_rng is None:
#         _global_rng = np.random.default_rng()
#
#     # Generate all random numbers at once (FAST!)
#     cdef np.ndarray[float, ndim=1] u_vals = _global_rng.uniform(
#         -np.pi/2, np.pi/2, n
#     ).astype(np.float32)
#
#     cdef np.ndarray[float, ndim=1] w_vals = _global_rng.exponential(
#         1.0, n
#     ).astype(np.float32)
#
#     # Prepare output
#     cdef np.ndarray[float, ndim=1] result = np.empty(n, dtype=np.float32)
#     cdef float[:] result_view = result
#     cdef float[:] u_view = u_vals
#     cdef float[:] w_view = w_vals
#
#     cdef int i
#     cdef float u, w, x
#
#     # Special case: alpha = 1 (Cauchy)
#     if alpha == 1.0:
#         for i in range(n):
#             result_view[i] = tan(u_view[i])
#         return result
#
#     # General case
#     cdef float alpha_inv = 1.0 / alpha
#     cdef float one_minus_alpha = 1.0 - alpha
#     cdef float scale = one_minus_alpha / alpha
#
#     for i in range(n):
#         u = u_view[i]
#         w = w_view[i]
#         x = (sin(alpha * u) / pow(cos(u), alpha_inv)) * \
#             pow(cos(u - alpha * u) / w, scale)
#         result_view[i] = x
#
#     return result

cpdef float[:] draw_random_stable(int n, float alpha):
    """
    Generate alpha-stable variates using the Chambers–Mallows–Stuck (CMS)
    algorithm with vectorised NumPy operations.

    Algorithm
    ---------
    Chambers, Mallows & Stuck (1976) show that for a symmetric stable
    distribution S_alpha(0, 1, 0) with 0 < alpha <= 2, the representation::

        X = sin(alpha * U) / cos(U)^(1/alpha)
            * (cos(U * (1 - alpha)) / W)^((1 - alpha) / alpha)

    where U ~ Uniform(-pi/2, pi/2) and W ~ Exponential(1), produces exact
    samples.  For alpha = 1 (Cauchy) the formula reduces to tan(U).

    Sampler choice
    --------------
    Three options were benchmarked (N=100k, best of 20 runs, float32 output):

    * SciPy ``levy_stable.rvs``:  ~12–23 M/s  — runs the same CMS math but
      incurs ~6.7 ms of Python-level overhead per 100k samples from the
      ``rv_continuous`` dispatch layer.  3–7x slower overall.
    * This function (hand-rolled CMS):  ~37–62 M/s  — fastest or tied.
    * GSL ``ssms_rng_levy_f32`` (``_c_rng``):  ~21–64 M/s  — used in the parallel
      path; comparable at most alpha values.

    SciPy's overhead dominates and is not justified in a per-trial simulation
    loop; this implementation is kept as the sequential-path sampler.
    See ``draw_random_stable_scipy`` for a SciPy-backed alternative that can
    be used for correctness cross-checks.

    Numerical stability guards
    --------------------------
    ``cos(pi/2 - eps) ≈ eps`` (first-order Taylor), so U values near ±pi/2
    drive ``cos(U)`` toward zero, causing ``cos(U)^(1/alpha)`` to blow up.
    The clamp ``eps = 1e-7`` is chosen to equal the float32 machine epsilon
    (``np.finfo(np.float32).eps ≈ 1.19e-7``): it is the smallest value where
    ``cos(U)`` is still reliably non-zero in float32 arithmetic.  Going
    smaller would silently round to zero in float32; going larger (e.g. 1e-4)
    would truncate the distribution tails.  In float64 the safe floor would
    be ~1e-15.  The ``w`` exponential is clamped at 1e-10 for the same reason.
    Output is clipped to [-1e10, 1e10] and NaN/inf replaced with 0 to prevent
    downstream propagation of degenerate values.

    Args:
        n (int): Number of variates to generate.
        alpha (float): Stability parameter; 0 < alpha <= 2.
            alpha = 2  → Gaussian,  alpha = 1  → Cauchy.

    Returns:
        float[:]: Array of n alpha-stable variates (float32).
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()

    # Generate uniform values and clamp away from boundaries
    # to avoid cos(u) ≈ 0 issues
    cdef np.ndarray[float, ndim=1] u_vals = _global_rng.uniform(
        -np.pi/2 + 1e-7, np.pi/2 - 1e-7, n
    ).astype(np.float32)

    # Generate exponential values with minimum to avoid division issues
    cdef np.ndarray[float, ndim=1] w_vals = _global_rng.exponential(
        1.0, n
    ).astype(np.float32)
    w_vals = np.maximum(w_vals, 1e-10)

    # Declare result array and intermediate values
    cdef np.ndarray[float, ndim=1] result
    cdef np.ndarray[float, ndim=1] cos_u
    cdef np.ndarray[float, ndim=1] cos_u_alpha
    cdef float alpha_inv
    cdef float scale

    # Vectorized computation with numerical stability
    if alpha == 1.0:
        # Cauchy case
        result = np.tan(u_vals).astype(np.float32)
        # Clamp extreme values
        result = np.clip(result, -1e10, 1e10)
        return result
    else:
        alpha_inv = 1.0 / alpha
        scale = (1.0 - alpha) / alpha

        # Compute cosines with clamping to avoid pow(0, x) issues
        cos_u = np.cos(u_vals)
        cos_u = np.sign(cos_u) * np.maximum(np.abs(cos_u), 1e-10)

        cos_u_alpha = np.cos(u_vals * (1.0 - alpha))
        cos_u_alpha = np.sign(cos_u_alpha) * np.maximum(np.abs(cos_u_alpha), 1e-10)

        # Use np.abs for power operations (CMS formula with symmetric stable)
        # This avoids "invalid value in power" for negative bases
        result = ((np.sin(alpha * u_vals) / (np.abs(cos_u) ** alpha_inv)) * \
                  ((np.abs(cos_u_alpha) / w_vals) ** scale)).astype(np.float32)

        # Clamp output and replace NaN with 0
        result = np.clip(result, -1e10, 1e10)
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

        return result

cpdef int sign(float x):
    """
    Determine the sign of a float.

    Args:
        x (float): The input float.

    Returns:
        int: 1 if x is positive, -1 if x is negative, 0 if x is zero.
    """
    return (x > 0) - (x < 0)

cpdef float csum(float[:] x):
    """
    Calculate the sum of elements in an array.

    Args:
        x (float[:]): The input array.

    Returns:
        float: The sum of all elements in the array.
    """
    cdef int i
    cdef int n = x.shape[0]
    cdef float total = 0

    for i in range(n):
        total += x[i]
    return total

# OLD IMPLEMENTATION (Marsaglia polar method helpers) - Kept for reference
# ## @cythonboundscheck(False)
# cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
#     """
#     Generate a pair of random floats from a standard normal distribution and assign them to an array.
#     DEPRECATED: Part of old Marsaglia-based implementation.
#
#     Args:
#         out (float[:]): The output array to store the generated values.
#         assign_ix (int): The starting index in the output array to assign the values.
#     """
#     cdef float x1, x2, w
#     w = 2.0
#
#     while(w >= 1.0):
#         x1 = (2.0 * random_uniform()) - 1.0
#         x2 = (2.0 * random_uniform()) - 1.0
#         w = (x1 * x1) + (x2 * x2)
#
#     w = ((-2.0 * log(w)) / w) ** 0.5
#     out[assign_ix] = x1 * w
#     out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0
#
# # @cythonboundscheck(False)
# cpdef float[:] draw_gaussian(int n):
#     """
#     Generate an array of random floats from a standard normal distribution.
#     DEPRECATED: This Marsaglia-based implementation is ~2.4x slower than NumPy's Ziggurat.
#
#     Args:
#         n (int): The number of random floats to generate.
#
#     Returns:
#         float[:]: An array of random floats from a standard normal distribution.
#     """
#     # Draws standard normal variables - need to have the variance rescaled
#     cdef int i
#     cdef float[:] result = np.zeros(n, dtype=DTYPE)
#     for i in range(n // 2):
#
#         assign_random_gaussian_pair(result, i * 2)
#     if n % 2 == 1:
#         result[n - 1] = random_gaussian()
#     return result


# OLD IMPLEMENTATION (C-based setup_simulation) - Kept for reference
# cpdef dict setup_simulation(
#       int n_samples,
#       int n_trials,
#       float max_t,
#       float delta_t,
#       random_state
# ):
#     """
#     Set up common simulation data structures.
#     DEPRECATED: Uses old C-based draw_gaussian which is ~2.4x slower.
#     """
#     set_seed(random_state)
#
#     cdef int num_draws = int((max_t / delta_t) + 1)
#
#     # Trajectory storage
#     traj = np.zeros((num_draws, 1), dtype=DTYPE)
#     traj[:, :] = -999
#
#     # Output arrays
#     rts = np.zeros((n_samples,
#                     n_trials,
#                     1),
#                     dtype=DTYPE)
#     choices = np.zeros((n_samples,
#                         n_trials,
#                         1),
#                         dtype=np.intc)
#
#     # Time array
#     t_s = np.arange(0,
#                     max_t + delta_t,
#                     delta_t)\
#                     .astype(DTYPE)
#
#     # Gaussian values
#     gaussian_values = draw_gaussian(num_draws)  # OLD VERSION
#     return {
#         'traj': traj,
#         'rts': rts,
#         'choices': choices,
#         't_s': t_s,
#         'gaussian_values': gaussian_values,
#         'num_draws': num_draws,
#         'delta_t_sqrt': sqrt(delta_t)
#     }

cpdef dict setup_simulation(
      int n_samples,
      int n_trials,
      float max_t,
      float delta_t,
      random_state
):
    """
    Set up common simulation data structures using NumPy's fast RNG.
    ~2.4x faster than the old C-based implementation.
    """
    set_seed(random_state)

    cdef int num_draws = int((max_t / delta_t) + 1)

    # Initialize NumPy RNG with same seed
    global _global_rng
    if random_state is not None:
        _global_rng = np.random.default_rng(random_state)
    else:
        _global_rng = np.random.default_rng()

    # Trajectory storage
    traj = np.zeros((num_draws, 1), dtype=DTYPE)
    traj[:, :] = -999

    # Output arrays
    rts = np.zeros((n_samples,
                    n_trials,
                    1),
                    dtype=DTYPE)
    choices = np.zeros((n_samples,
                        n_trials,
                        1),
                        dtype=np.intc)

    # Time array
    t_s = np.arange(0,
                    max_t + delta_t,
                    delta_t)\
                    .astype(DTYPE)

    # Gaussian values (using new fast NumPy-based draw_gaussian)
    gaussian_values = draw_gaussian(num_draws)
    # Uniform values (using new fast NumPy-based draw_uniform, same _global_rng seed)
    uniform_values = draw_uniform(num_draws)
    return {
        'traj': traj,
        'rts': rts,
        'choices': choices,
        't_s': t_s,
        'gaussian_values': gaussian_values,
        'uniform_values': uniform_values,
        'num_draws': num_draws,
        'delta_t_sqrt': sqrt(delta_t)
    }

cpdef void compute_boundary(
    boundary,
    t_s,
    boundary_fun,
    dict boundary_params
):
    """Compute boundary values for given time points.

    Boundary functions now receive all parameters via boundary_params,
    including 'a'. No special treatment needed.
    """
    boundary[:] = boundary_fun(t=t_s, **boundary_params).astype(DTYPE)

cpdef float compute_smooth_unif(
    bint smooth_unif,
    float t_particle,
    float deadline_tmp,
    float delta_t,
    float u,
):
    """Compute uniform smoothing adjustment for reaction times.

    Args:
        smooth_unif: Whether smooth-uniform RT jitter is enabled.
        t_particle: Current particle time.
        deadline_tmp: Effective deadline for this trial.
        delta_t: Simulation time step.
        u: Pre-drawn uniform(0, 1) value from the caller's ``draw_uniform()``
           batch.  Always required — the caller increments its batch index
           regardless of ``smooth_unif`` so the batch consumption pattern
           stays consistent.  The value is ignored when ``smooth_unif=False``.
    """
    if not smooth_unif:
        return 0.0

    if t_particle == 0.0:
        return u * 0.5 * delta_t
    elif t_particle < deadline_tmp:
        return (0.5 - u) * delta_t
    else:
        return 0.0

cpdef void enforce_deadline(
    float[:, :, :] rts_view,
    float[:] deadline_view,
    int n,
    int k,
    int idx=0
):
    """Set RT to OMISSION_SENTINEL (-999) if it exceeds deadline or deadline is invalid.

    The value -999 is the sentinel indicating an omission (no response within deadline).
    This corresponds to ssms.OMISSION_SENTINEL in the Python API.
    """
    if (rts_view[n, k, idx] >= deadline_view[k]) or (deadline_view[k] <= 0):
        rts_view[n, k, idx] = -999

cpdef inline float compute_deadline_tmp(
    float max_t,
    float deadline_k,
    float t_k
):
    """Compute effective deadline for simulation loop."""
    return min(max_t, deadline_k - t_k)

cpdef dict build_param_dict_from_2d_array(
    np.ndarray arr,
    str prefix,
    int n_cols
):
    """
    Build dictionary from 2D array columns for race/LBA models.
    Example: v[:, 0], v[:, 1] -> {'v0': v[:, 0], 'v1': v[:, 1]}
    """
    result = {}
    for i in range(n_cols):
        result[f'{prefix}{i}'] = arr[:, i]
    return result

cpdef dict build_minimal_metadata(
    str simulator_name,
    list possible_choices,
    int n_samples,
    int n_trials,
    str boundary_fun_name=None
):
    """Build minimal metadata dictionary - the BASE for all metadata."""
    metadata = {
        'simulator': simulator_name,
        'possible_choices': possible_choices,
        'n_samples': n_samples,
        'n_trials': n_trials,
    }
    if boundary_fun_name is not None:
        metadata['boundary_fun_type'] = boundary_fun_name
    return metadata

cpdef dict build_full_metadata(
    dict minimal_metadata,
    dict params,
    dict sim_config,
    boundary_fun=None,
    np.ndarray boundary=None,
    np.ndarray traj=None,
    dict boundary_params=None,
    dict drift_params=None,
    dict extra_params=None
):
    """
    Build full metadata by AUGMENTING minimal_metadata.
    This ensures consistency - full always includes everything from minimal.
    """
    # Start with a copy of minimal metadata
    metadata = dict(minimal_metadata)

    # Add all the full metadata fields
    metadata.update(params)
    metadata.update(sim_config)

    # Add optional fields
    if boundary_fun is not None:
        metadata['boundary_fun_type'] = boundary_fun.__name__
    if boundary is not None:
        metadata['boundary'] = boundary
    if traj is not None:
        metadata['trajectory'] = traj
    if boundary_params:
        metadata.update(boundary_params)
    if drift_params:
        metadata.update(drift_params)
    if extra_params:
        metadata.update(extra_params)

    return metadata

cpdef dict build_return_dict(
    np.ndarray rts,
    np.ndarray choices,
    dict metadata,
    dict extra_arrays=None
):
    """Build return dictionary with rts, choices, and metadata."""
    result = {'rts': rts, 'choices': choices, 'metadata': metadata}
    if extra_arrays:
        result.update(extra_arrays)
    return result
