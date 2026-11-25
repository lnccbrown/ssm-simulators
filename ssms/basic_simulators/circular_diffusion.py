"""First exit of a centered planar diffusion from a fixed circle (Smith, 2016).

With radius a, isotropic noise s and constant drift v, the exit angle and
decision time are independent (https://doi.org/10.1037/rev0000023). The angle
is von Mises with concentration k = a |v| / s². In units of a²/s², the time
has Laplace transform I0(k)/I0(sqrt(k²+2u)). The Bessel product
(https://dlmf.nist.gov/10.21.E15) identifies it as a sum of independent
exponentials with means 2/(j_n²+k²), where j_n are the positive zeros of J0.

We sample the first 128 terms and replace the tail by its exact mean. This
preserves mean decision time; the omitted tail's standard deviation is less
than 0.000082 a²/s² even at zero drift (j_n > (n-1/4) pi). No Euler steps,
endpoint correction or angle binning are needed.
"""

import numpy as np
from scipy.special import i0e, i1e, jn_zeros

from .observation_results import OBSERVATION_SCHEMA_VERSION

_J0_SQUARED = jn_zeros(0, 128) ** 2


def circular_diffusion_observation_metadata() -> dict:
    """Describe the ordered RT and angle (radians); each caller owns its mappings."""
    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema": (
            {"name": "rt", "kind": "continuous", "lower": 0, "lower_inclusive": False},
            {"name": "response", "kind": "circular", "lower": -np.pi, "upper": np.pi},
        ),
    }


def _draw_exits(theta, sigma, n_replicas, random_state):
    """Draw uncensored [RT, angle] in (trial, replica, coordinate) order."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    if theta.ndim != 2 or theta.shape[1] != 4 or not np.isfinite(theta).all():
        raise ValueError("theta must contain finite [vx, vy, a, t] rows")
    vx, vy, a, t = theta.T
    if (a <= 0).any() or (t < 0).any():
        raise ValueError("radius a must be positive and nondecision time t nonnegative")
    if (
        isinstance(n_replicas, (bool, np.bool_))
        or not isinstance(n_replicas, (int, np.integer))
        or n_replicas < 1
    ):
        raise ValueError("n_replicas must be a positive integer")
    noise = np.broadcast_to(np.asarray(sigma, dtype=float), a.shape)
    if not np.isfinite(noise).all() or (noise < 0).any():
        raise ValueError("noise must be finite and nonnegative")

    if isinstance(random_state, (int, np.integer)) and random_state < 0:
        random_state = int(random_state) % 2**32  # ssms accepts signed C seeds
    rng = np.random.default_rng(random_state)
    speed, axis = np.hypot(vx, vy), np.arctan2(vy, vx)
    shape = (len(theta), n_replicas)
    times = np.empty(shape)
    angles = np.empty(shape)
    noisy = noise > 0
    if noisy.any():
        scale = (a[noisy] / noise[noisy]) ** 2
        k = a[noisy] * speed[noisy] / noise[noisy] ** 2
        mean = np.divide(i1e(k), k * i0e(k), out=np.full_like(k, 0.5), where=k != 0)
        size = (len(k), n_replicas)
        elapsed = np.zeros(size)
        for root_sq in _J0_SQUARED:
            weight = 2 / (root_sq + k**2)
            elapsed += weight[:, None] * rng.standard_exponential(size)
            mean -= weight
        times[noisy] = (elapsed + mean[:, None]) * scale[:, None]
        angles[noisy] = rng.vonmises(axis[noisy, None], k[:, None], size=size)
    if (~noisy).any():
        v = speed[~noisy]
        times[~noisy] = np.divide(
            a[~noisy], v, out=np.full_like(v, np.inf), where=v > 0
        )[:, None]
        angles[~noisy] = axis[~noisy, None]
    return np.stack(
        (times + t[:, None], (angles + np.pi) % (2 * np.pi) - np.pi), axis=-1
    )


def sample_circular_diffusion(theta, random_state, n_replicas):
    """Sample uncensored unit-noise RT/angle pairs, replicas contiguous per theta row.

    theta is one [vx, vy, a, t] vector or a matrix with one vector per trial.
    random_state is an integer, NumPy Generator, or None. Returns a float array
    of shape (n_trials * n_replicas, 2); radians are always in [-pi, pi).
    """
    return _draw_exits(theta, 1.0, n_replicas, random_state).reshape(-1, 2)


def circular_diffusion(
    *, vx, vy, a, t, s, n_samples, max_t, random_state, deadline, **kwargs
):
    """ssms registry interface; max_t caps decision time, deadline caps total RT."""
    theta = np.column_stack((vx, vy, a, t))
    obs = _draw_exits(theta, s, n_samples, random_state).swapaxes(0, 1)
    rts, angle = obs[..., 0], obs[..., 1]
    omitted = (~np.isfinite(rts)) | (rts - t > max_t) | (rts >= deadline)
    # The legacy slots use -999, while native omitted observations are all NaN.
    legacy_rts = np.where(omitted, -999.0, rts)[..., None]
    legacy_angles = np.where(omitted, -999.0, angle)[..., None]
    obs[omitted] = np.nan
    return {
        "observations": obs,
        "omission_mask": omitted,
        "rts": legacy_rts,
        "choices": legacy_angles,
        "metadata": {
            "simulator": "circular_diffusion",
            "possible_choices": [],
            "max_t": max_t,
            "n_samples": n_samples,
            "n_trials": len(theta),
            **circular_diffusion_observation_metadata(),
        },
    }
