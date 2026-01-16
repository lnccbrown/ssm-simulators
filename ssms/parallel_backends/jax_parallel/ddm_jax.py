"""
JAX-based DDM Simulators with Vectorization and XLA Compilation

This module implements DDM simulators using JAX's transformations:
- jit: XLA compilation for fast execution
- vmap: Automatic vectorization across samples/trials
- lax.while_loop: Efficient compiled loops

Key optimizations:
1. Pure functional implementation (no side effects)
2. XLA compilation for optimal CPU/GPU execution
3. Automatic batching with vmap
4. Efficient random number handling with JAX's PRNG

Note on JAX DDM Simulation:
    Unlike traditional while-loop approaches, JAX benefits from fixed-iteration
    approaches due to XLA's compilation model. We provide both:
    1. A while-loop version (more memory efficient)
    2. A vectorized/scan version (can be faster on GPU)
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, vmap, lax
from functools import partial
import numpy as np
from typing import Tuple, Dict, Any


def get_jax_device_info() -> Dict[str, Any]:
    """Get information about available JAX devices."""
    try:
        devices = jax.devices()
        return {
            "available": True,
            "devices": [str(d) for d in devices],
            "default_backend": jax.default_backend(),
            "device_count": len(devices),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ============================================================================
# Core DDM Simulation (Single Sample) using while_loop
# ============================================================================


@partial(jit, static_argnums=(6, 7))
def _ddm_single_while(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    key: jrandom.PRNGKey,
) -> Tuple[float, int]:
    """
    Simulate a single DDM sample using lax.while_loop.

    This is more memory efficient for long simulations.
    """
    sqrt_dt = jnp.sqrt(delta_t)
    sqrt_st = sqrt_dt * s
    deadline_tmp = jnp.minimum(max_t, deadline - t_ndt)

    # Initial state: (y, t_particle, key, done)
    y_init = z * a

    def cond_fn(state):
        y, t_particle, _, done = state
        return ~done

    def body_fn(state):
        y, t_particle, key, done = state
        key, subkey = jrandom.split(key)
        noise = jrandom.normal(subkey, dtype=jnp.float32)

        y_new = y + v * delta_t + sqrt_st * noise
        t_new = t_particle + delta_t

        # Check if we hit boundary or deadline
        hit_boundary = (y_new <= 0) | (y_new >= a)
        hit_deadline = t_new > deadline_tmp
        done_new = hit_boundary | hit_deadline

        return (y_new, t_new, key, done_new)

    y_final, t_particle, _, _ = lax.while_loop(
        cond_fn, body_fn, (y_init, jnp.float32(0.0), key, False)
    )

    # Compute RT and choice
    rt = t_particle + t_ndt

    # Choice: 1 if upper, -1 if lower
    choice = jnp.where(y_final > 0, 1, -1)

    # Apply deadline
    rt = jnp.where(rt >= deadline, -999.0, rt)
    rt = jnp.where(deadline <= 0, -999.0, rt)

    return rt, choice


# ============================================================================
# Vectorized DDM using scan (better for GPU)
# ============================================================================


@partial(jit, static_argnums=(6, 7, 8))
def _ddm_single_scan(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    num_steps: int,
    key: jrandom.PRNGKey,
) -> Tuple[float, int]:
    """
    Simulate a single DDM sample using lax.scan.

    This pre-generates all random numbers and uses scan for the update.
    Better GPU utilization but uses more memory.
    """
    sqrt_dt = jnp.sqrt(delta_t)
    sqrt_st = sqrt_dt * s
    deadline_tmp = jnp.minimum(max_t, deadline - t_ndt)

    # Pre-generate all random numbers
    noise = jrandom.normal(key, shape=(num_steps,), dtype=jnp.float32)

    # Initial state
    y_init = z * a

    def step_fn(carry, noise_i):
        y, t_particle, done = carry

        # Update only if not done
        y_new = jnp.where(done, y, y + v * delta_t + sqrt_st * noise_i)
        t_new = jnp.where(done, t_particle, t_particle + delta_t)

        # Check boundaries
        hit_boundary = (y_new <= 0) | (y_new >= a)
        hit_deadline = t_new > deadline_tmp
        done_new = done | hit_boundary | hit_deadline

        return (y_new, t_new, done_new), None

    (y_final, t_particle, _), _ = lax.scan(
        step_fn, (y_init, jnp.float32(0.0), False), noise
    )

    rt = t_particle + t_ndt
    choice = jnp.where(y_final > 0, 1, -1)

    rt = jnp.where(rt >= deadline, -999.0, rt)
    rt = jnp.where(deadline <= 0, -999.0, rt)

    return rt, choice


# ============================================================================
# Batched simulation using vmap
# ============================================================================


@partial(jit, static_argnums=(6, 7, 8, 9))
def _ddm_batch_samples(
    v: float,
    a: float,
    z: float,
    t_ndt: float,
    deadline: float,
    s: float,
    delta_t: float,
    max_t: float,
    num_steps: int,
    n_samples: int,
    key: jrandom.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate n_samples for a single trial using vmap.
    """
    keys = jrandom.split(key, n_samples)

    # vmap over samples
    simulate_one = partial(
        _ddm_single_scan, v, a, z, t_ndt, deadline, s, delta_t, max_t, num_steps
    )

    rts, choices = vmap(simulate_one)(keys)
    return rts, choices


@partial(jit, static_argnums=(6, 7, 8, 9, 10))
def _ddm_batch_all(
    v: jnp.ndarray,
    a: jnp.ndarray,
    z: jnp.ndarray,
    t_ndt: jnp.ndarray,
    deadline: jnp.ndarray,
    s: jnp.ndarray,
    delta_t: float,
    max_t: float,
    num_steps: int,
    n_samples: int,
    n_trials: int,
    key: jrandom.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate all samples for all trials using nested vmap.
    """
    # Split keys for each trial
    trial_keys = jrandom.split(key, n_trials)

    # vmap over trials
    def simulate_trial(trial_params, trial_key):
        v_t, a_t, z_t, t_t, dl_t, s_t = trial_params
        return _ddm_batch_samples(
            v_t,
            a_t,
            z_t,
            t_t,
            dl_t,
            s_t,
            delta_t,
            max_t,
            num_steps,
            n_samples,
            trial_key,
        )

    trial_params = (v, a, z, t_ndt, deadline, s)
    rts, choices = vmap(simulate_trial, in_axes=(0, 0))(
        jax.tree.map(lambda x: x, trial_params),  # Stack along axis 0
        trial_keys,
    )

    # Shape: (n_trials, n_samples) -> (n_samples, n_trials)
    rts = rts.T
    choices = choices.T

    return rts, choices


# ============================================================================
# Public API
# ============================================================================


def ddm_jax(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    smooth_unif: bool = False,  # Note: not implemented in JAX version
    use_scan: bool = True,  # Use scan version (better for GPU)
    **kwargs,
) -> dict:
    """
    DDM simulator using JAX with automatic vectorization.

    This function uses JAX's vmap to automatically parallelize across
    samples and trials, with XLA compilation for fast execution on
    CPU, GPU, or TPU.

    Parameters
    ----------
    v : np.ndarray
        Drift rate for each trial
    a : np.ndarray
        Boundary separation for each trial
    z : np.ndarray
        Starting point (as proportion of a) for each trial
    t : np.ndarray
        Non-decision time for each trial
    deadline : np.ndarray, optional
        Maximum allowed RT for each trial (default: 999)
    s : np.ndarray, optional
        Noise standard deviation for each trial (default: 1.0)
    delta_t : float
        Time step size (default: 0.001)
    max_t : float
        Maximum simulation time (default: 20.0)
    n_samples : int
        Number of samples per trial (default: 20000)
    n_trials : int, optional
        Number of trials (inferred from parameter arrays if not given)
    random_state : int, optional
        Random seed for reproducibility
    return_option : str
        'full' or 'minimal' (default: 'full')
    smooth_unif : bool
        Apply uniform smoothing (not implemented, default: False)
    use_scan : bool
        Use scan-based simulation (default: True)

    Returns
    -------
    dict
        Dictionary with 'rts', 'choices', and 'metadata'
    """
    # Convert to JAX arrays
    v = jnp.asarray(v, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    t_ndt = jnp.asarray(t, dtype=jnp.float32)

    # Determine n_trials from array length if not provided
    if n_trials is None:
        n_trials = len(v)
    else:
        # If n_trials is provided and smaller, slice arrays
        if n_trials < len(v):
            v = v[:n_trials]
            a = a[:n_trials]
            z = z[:n_trials]
            t_ndt = t_ndt[:n_trials]

    if deadline is None:
        deadline = jnp.full(n_trials, 999.0, dtype=jnp.float32)
    else:
        deadline = jnp.asarray(deadline, dtype=jnp.float32)
        if len(deadline) > n_trials:
            deadline = deadline[:n_trials]

    if s is None:
        s = jnp.ones(n_trials, dtype=jnp.float32)
    else:
        s = jnp.asarray(s, dtype=jnp.float32)
        if len(s) > n_trials:
            s = s[:n_trials]

    # Initialize PRNG
    if random_state is None:
        random_state = np.random.default_rng().integers(0, 2**31)
    key = jrandom.PRNGKey(random_state)

    # Compute num_steps
    num_steps = int((max_t / delta_t) + 1)

    # Run simulation
    rts_jax, choices_jax = _ddm_batch_all(
        v, a, z, t_ndt, deadline, s, delta_t, max_t, num_steps, n_samples, n_trials, key
    )

    # Convert back to numpy and reshape
    rts = np.asarray(rts_jax).reshape(n_samples, n_trials, 1).astype(np.float32)
    choices = np.asarray(choices_jax).reshape(n_samples, n_trials, 1).astype(np.int32)

    # Build metadata
    metadata = {
        "simulator": "ddm_jax",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "device": str(jax.devices()[0]),
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": np.asarray(v),
                "a": np.asarray(a),
                "z": np.asarray(z),
                "t": np.asarray(t_ndt),
                "deadline": np.asarray(deadline),
                "s": np.asarray(s),
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


# ============================================================================
# Alternative: Fully vectorized approach (no loops)
# ============================================================================


@partial(jit, static_argnums=(6, 7, 8, 9, 10))
def _ddm_fully_vectorized(
    v: jnp.ndarray,
    a: jnp.ndarray,
    z: jnp.ndarray,
    t_ndt: jnp.ndarray,
    deadline: jnp.ndarray,
    s: jnp.ndarray,
    delta_t: float,
    max_t: float,
    num_steps: int,
    n_samples: int,
    n_trials: int,
    key: jrandom.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fully vectorized DDM simulation.

    Pre-generates all trajectories and finds first crossing.
    Most memory intensive but potentially fastest on GPU.
    """
    sqrt_dt = jnp.sqrt(delta_t)

    # Generate all noise at once: (n_samples, n_trials, num_steps)
    noise = jrandom.normal(
        key, shape=(n_samples, n_trials, num_steps), dtype=jnp.float32
    )

    # Broadcast parameters: (1, n_trials, 1)
    v_bc = v[None, :, None]
    a_bc = a[None, :, None]
    z_bc = z[None, :, None]
    s_bc = s[None, :, None]
    t_ndt_bc = t_ndt[None, :]
    deadline_bc = deadline[None, :]

    # Starting position: (n_samples, n_trials)
    y0 = (z_bc * a_bc).squeeze(-1)

    # Increments: (n_samples, n_trials, num_steps)
    increments = v_bc * delta_t + sqrt_dt * s_bc * noise

    # Cumulative sum to get trajectories
    trajectories = y0[:, :, None] + jnp.cumsum(increments, axis=-1)

    # Find first crossing (vectorized)
    # Crossing occurs when y <= 0 or y >= a
    a_2d = a[None, :, None]  # (1, n_trials, 1)
    crossed_lower = trajectories <= 0
    crossed_upper = trajectories >= a_2d
    crossed = crossed_lower | crossed_upper

    # Find first crossing time index
    # Use argmax on the crossed array; if no crossing, returns 0
    # We need to handle the case of no crossing carefully
    has_crossed = crossed.any(axis=-1)  # (n_samples, n_trials)
    first_cross_idx = jnp.argmax(crossed, axis=-1)  # (n_samples, n_trials)

    # For samples that never crossed, set to num_steps
    first_cross_idx = jnp.where(has_crossed, first_cross_idx, num_steps - 1)

    # Compute RT
    t_particle = (first_cross_idx + 1) * delta_t
    rt = t_particle + t_ndt_bc

    # Determine choice based on final position at crossing
    sample_idx = jnp.arange(n_samples)[:, None]
    trial_idx = jnp.arange(n_trials)[None, :]
    y_at_cross = trajectories[sample_idx, trial_idx, first_cross_idx]
    choice = jnp.where(y_at_cross > 0, 1, -1)

    # Apply deadline
    rt = jnp.where(rt >= deadline_bc, -999.0, rt)
    rt = jnp.where(deadline_bc <= 0, -999.0, rt)

    return rt.astype(jnp.float32), choice.astype(jnp.int32)


def ddm_jax_vectorized(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    **kwargs,
) -> dict:
    """
    Fully vectorized DDM simulator using JAX.

    This version pre-generates all random numbers and computes
    trajectories in a fully vectorized manner. Best for GPU
    when memory is not a concern.

    WARNING: Memory usage is O(n_samples * n_trials * num_steps).
    For large simulations, use ddm_jax() instead.
    """
    # Convert to JAX arrays
    v = jnp.asarray(v, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    t_ndt = jnp.asarray(t, dtype=jnp.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = jnp.full(n_trials, 999.0, dtype=jnp.float32)
    else:
        deadline = jnp.asarray(deadline, dtype=jnp.float32)

    if s is None:
        s = jnp.ones(n_trials, dtype=jnp.float32)
    else:
        s = jnp.asarray(s, dtype=jnp.float32)

    if random_state is None:
        random_state = np.random.default_rng().integers(0, 2**31)
    key = jrandom.PRNGKey(random_state)

    num_steps = int((max_t / delta_t) + 1)

    # Run fully vectorized simulation
    rts_jax, choices_jax = _ddm_fully_vectorized(
        v, a, z, t_ndt, deadline, s, delta_t, max_t, num_steps, n_samples, n_trials, key
    )

    # Convert back to numpy
    rts = np.asarray(rts_jax).reshape(n_samples, n_trials, 1).astype(np.float32)
    choices = np.asarray(choices_jax).reshape(n_samples, n_trials, 1).astype(np.int32)

    metadata = {
        "simulator": "ddm_jax_vectorized",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": "constant",
        "device": str(jax.devices()[0]),
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": np.asarray(v),
                "a": np.asarray(a),
                "z": np.asarray(z),
                "t": np.asarray(t_ndt),
                "deadline": np.asarray(deadline),
                "s": np.asarray(s),
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


# ============================================================================
# Flexible Boundary DDM
# ============================================================================


def ddm_flexbound_jax(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    **kwargs,
) -> dict:
    """
    DDM simulator with flexible boundaries using JAX.

    The boundary is precomputed as a JAX array before simulation.
    """
    if boundary_params is None:
        boundary_params = {}

    # Convert to JAX arrays
    v = jnp.asarray(v, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    t_ndt = jnp.asarray(t, dtype=jnp.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = jnp.full(n_trials, 999.0, dtype=jnp.float32)
    else:
        deadline = jnp.asarray(deadline, dtype=jnp.float32)

    if s is None:
        s = jnp.ones(n_trials, dtype=jnp.float32)
    else:
        s = jnp.asarray(s, dtype=jnp.float32)

    if random_state is None:
        random_state = np.random.default_rng().integers(0, 2**31)
    key = jrandom.PRNGKey(random_state)

    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)

    # Precompute boundary for all trials
    boundary_np = np.zeros((n_trials, num_steps), dtype=np.float32)
    for k in range(n_trials):
        boundary_params_tmp = {
            key_: boundary_params[key_][k] for key_ in boundary_params.keys()
        }
        boundary_np[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    boundary = jnp.asarray(boundary_np)

    # Run simulation with flexible boundary
    rts_jax, choices_jax = _ddm_flexbound_batch(
        v,
        z,
        t_ndt,
        deadline,
        s,
        boundary,
        delta_t,
        max_t,
        num_steps,
        n_samples,
        n_trials,
        key,
    )

    rts = np.asarray(rts_jax).reshape(n_samples, n_trials, 1).astype(np.float32)
    choices = np.asarray(choices_jax).reshape(n_samples, n_trials, 1).astype(np.int32)

    metadata = {
        "simulator": "ddm_flexbound_jax",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": boundary_fun.__name__ if boundary_fun else "unknown",
        "device": str(jax.devices()[0]),
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": np.asarray(v),
                "a": np.asarray(a),
                "z": np.asarray(z),
                "t": np.asarray(t_ndt),
                "deadline": np.asarray(deadline),
                "s": np.asarray(s),
                "boundary": boundary_np,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


@partial(jit, static_argnums=(6, 7, 8, 9, 10))
def _ddm_flexbound_batch(
    v: jnp.ndarray,
    z: jnp.ndarray,
    t_ndt: jnp.ndarray,
    deadline: jnp.ndarray,
    s: jnp.ndarray,
    boundary: jnp.ndarray,  # (n_trials, num_steps)
    delta_t: float,
    max_t: float,
    num_steps: int,
    n_samples: int,
    n_trials: int,
    key: jrandom.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batch simulation with flexible boundary."""
    sqrt_dt = jnp.sqrt(delta_t)

    # Generate all noise
    noise = jrandom.normal(
        key, shape=(n_samples, n_trials, num_steps), dtype=jnp.float32
    )

    # Broadcast parameters
    v_bc = v[None, :, None]
    z_bc = z[None, :, None]
    s_bc = s[None, :, None]
    t_ndt_bc = t_ndt[None, :]
    deadline_bc = deadline[None, :]

    # Starting position using initial boundary
    b0 = boundary[:, 0][None, :, None]  # (1, n_trials, 1)
    y0 = (-1) * b0 + (z_bc * 2 * b0)
    y0 = y0.squeeze(-1)  # (n_samples, n_trials)

    # Increments
    increments = v_bc * delta_t + sqrt_dt * s_bc * noise

    # Cumulative sum for trajectories
    trajectories = y0[:, :, None] + jnp.cumsum(increments, axis=-1)

    # Check boundary crossings (boundary varies with time)
    # boundary shape: (n_trials, num_steps) -> (1, n_trials, num_steps)
    boundary_bc = boundary[None, :, :]
    crossed_lower = trajectories <= (-1) * boundary_bc
    crossed_upper = trajectories >= boundary_bc
    crossed = crossed_lower | crossed_upper

    # Find first crossing
    has_crossed = crossed.any(axis=-1)
    first_cross_idx = jnp.argmax(crossed, axis=-1)
    first_cross_idx = jnp.where(has_crossed, first_cross_idx, num_steps - 1)

    # Compute RT and choice
    t_particle = (first_cross_idx + 1) * delta_t
    rt = t_particle + t_ndt_bc

    sample_idx = jnp.arange(n_samples)[:, None]
    trial_idx = jnp.arange(n_trials)[None, :]
    y_at_cross = trajectories[sample_idx, trial_idx, first_cross_idx]
    choice = jnp.where(y_at_cross > 0, 1, -1)

    rt = jnp.where(rt >= deadline_bc, -999.0, rt)
    rt = jnp.where(deadline_bc <= 0, -999.0, rt)

    return rt.astype(jnp.float32), choice.astype(jnp.int32)


# ============================================================================
# Full DDM with Inter-trial Variability
# ============================================================================


def full_ddm_jax(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    sz: np.ndarray,
    sv: np.ndarray,
    st: np.ndarray,
    deadline: np.ndarray | None = None,
    s: np.ndarray | None = None,
    delta_t: float = 0.001,
    max_t: float = 20.0,
    n_samples: int = 20000,
    n_trials: int | None = None,
    boundary_fun=None,
    boundary_params: dict | None = None,
    random_state: int | None = None,
    return_option: str = "full",
    **kwargs,
) -> dict:
    """
    Full DDM simulator with inter-trial variability using JAX.
    """
    if boundary_params is None:
        boundary_params = {}

    # Convert to JAX arrays
    v = jnp.asarray(v, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    t_ndt = jnp.asarray(t, dtype=jnp.float32)
    sz = jnp.asarray(sz, dtype=jnp.float32)
    sv = jnp.asarray(sv, dtype=jnp.float32)
    st = jnp.asarray(st, dtype=jnp.float32)

    if n_trials is None:
        n_trials = len(v)

    if deadline is None:
        deadline = jnp.full(n_trials, 999.0, dtype=jnp.float32)
    else:
        deadline = jnp.asarray(deadline, dtype=jnp.float32)

    if s is None:
        s = jnp.ones(n_trials, dtype=jnp.float32)
    else:
        s = jnp.asarray(s, dtype=jnp.float32)

    if random_state is None:
        random_state = np.random.default_rng().integers(0, 2**31)
    key = jrandom.PRNGKey(random_state)

    num_steps = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t, dtype=np.float32)

    # Precompute boundary
    boundary_np = np.zeros((n_trials, num_steps), dtype=np.float32)
    for k in range(n_trials):
        boundary_params_tmp = {
            key_: boundary_params[key_][k] for key_ in boundary_params.keys()
        }
        boundary_np[k, : len(t_s)] = boundary_fun(t=t_s, **boundary_params_tmp).astype(
            np.float32
        )[:num_steps]

    boundary = jnp.asarray(boundary_np)

    # Run simulation
    rts_jax, choices_jax = _full_ddm_batch(
        v,
        z,
        t_ndt,
        sz,
        sv,
        st,
        deadline,
        s,
        boundary,
        delta_t,
        max_t,
        num_steps,
        n_samples,
        n_trials,
        key,
    )

    rts = np.asarray(rts_jax).reshape(n_samples, n_trials, 1).astype(np.float32)
    choices = np.asarray(choices_jax).reshape(n_samples, n_trials, 1).astype(np.int32)

    metadata = {
        "simulator": "full_ddm_jax",
        "possible_choices": [-1, 1],
        "n_samples": n_samples,
        "n_trials": n_trials,
        "boundary_fun_type": boundary_fun.__name__ if boundary_fun else "unknown",
        "device": str(jax.devices()[0]),
    }

    if return_option == "full":
        metadata.update(
            {
                "delta_t": delta_t,
                "max_t": max_t,
                "v": np.asarray(v),
                "a": np.asarray(a),
                "z": np.asarray(z),
                "t": np.asarray(t_ndt),
                "sz": np.asarray(sz),
                "sv": np.asarray(sv),
                "st": np.asarray(st),
                "deadline": np.asarray(deadline),
                "s": np.asarray(s),
                "boundary": boundary_np,
            }
        )

    return {"rts": rts, "choices": choices, "metadata": metadata}


@partial(jit, static_argnums=(9, 10, 11, 12, 13))
def _full_ddm_batch(
    v: jnp.ndarray,
    z: jnp.ndarray,
    t_ndt: jnp.ndarray,
    sz: jnp.ndarray,
    sv: jnp.ndarray,
    st: jnp.ndarray,
    deadline: jnp.ndarray,
    s: jnp.ndarray,
    boundary: jnp.ndarray,
    delta_t: float,
    max_t: float,
    num_steps: int,
    n_samples: int,
    n_trials: int,
    key: jrandom.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batch simulation for full DDM with inter-trial variability."""
    sqrt_dt = jnp.sqrt(delta_t)

    # Split key for different random components
    key, key_noise, key_z, key_v, key_t = jrandom.split(key, 5)

    # Generate noise for diffusion
    noise = jrandom.normal(
        key_noise, shape=(n_samples, n_trials, num_steps), dtype=jnp.float32
    )

    # Generate variability
    z_var = jrandom.uniform(
        key_z, shape=(n_samples, n_trials), minval=-0.5, maxval=0.5, dtype=jnp.float32
    )
    v_var = jrandom.normal(key_v, shape=(n_samples, n_trials), dtype=jnp.float32)
    t_var = jrandom.uniform(
        key_t, shape=(n_samples, n_trials), minval=-0.5, maxval=0.5, dtype=jnp.float32
    )

    # Apply variability
    # sz, sv, st shape: (n_trials,) -> (1, n_trials)
    z_effective = z[None, :] + sz[None, :] * z_var  # (n_samples, n_trials)
    v_effective = v[None, :] + sv[None, :] * v_var  # (n_samples, n_trials)
    t_effective = t_ndt[None, :] + st[None, :] * t_var  # (n_samples, n_trials)

    s_bc = s[None, :, None]  # (1, n_trials, 1)

    # Starting position
    b0 = boundary[:, 0][None, :]  # (1, n_trials)
    y0 = (-1) * b0 + (z_effective * 2 * b0)  # (n_samples, n_trials)

    # Drift increment (constant within trial but varies across samples due to sv)
    drift_increment = v_effective[:, :, None] * delta_t  # (n_samples, n_trials, 1)

    # Total increments
    increments = drift_increment + sqrt_dt * s_bc * noise

    # Trajectories
    trajectories = y0[:, :, None] + jnp.cumsum(increments, axis=-1)

    # Check crossings
    boundary_bc = boundary[None, :, :]
    crossed_lower = trajectories <= (-1) * boundary_bc
    crossed_upper = trajectories >= boundary_bc
    crossed = crossed_lower | crossed_upper

    has_crossed = crossed.any(axis=-1)
    first_cross_idx = jnp.argmax(crossed, axis=-1)
    first_cross_idx = jnp.where(has_crossed, first_cross_idx, num_steps - 1)

    t_particle = (first_cross_idx + 1) * delta_t
    rt = t_particle + t_effective

    sample_idx = jnp.arange(n_samples)[:, None]
    trial_idx = jnp.arange(n_trials)[None, :]
    y_at_cross = trajectories[sample_idx, trial_idx, first_cross_idx]
    choice = jnp.where(y_at_cross > 0, 1, -1)

    deadline_bc = deadline[None, :]
    rt = jnp.where(rt >= deadline_bc, -999.0, rt)
    rt = jnp.where(deadline_bc <= 0, -999.0, rt)

    return rt.astype(jnp.float32), choice.astype(jnp.int32)
