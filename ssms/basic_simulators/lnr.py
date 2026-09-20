"""Lognormal race (LNR) simulator.

The LNR is a linear *deterministic* race: each of ``N`` accumulators travels a
distance ``D`` at a constant rate ``V``, so its boundary-crossing time is
``T = D / V`` (Eq. 1). When distance and rate are (independent) Lognormal
variables their ratio is Lognormal as well (Eq. 5),

    T_i ~ exp(N(mu_i, sigma_i**2)),   mu = mu_d - mu_v,  sigma^2 = sigma_d^2 + sigma_v^2

so the simulator works directly on the finishing times ``T``: the decomposition
into distance and rate is not identifiable from behaviour (the paper says so
explicitly, p. 6). The first accumulator to finish wins the race and its
response is made; observed RT is ``t + min_i T_i`` where ``t`` is the shift
(the paper's ``t0``/``theta``, response production plus dead time).

For two accumulators the paper also allows the log finishing times to be
correlated (Eqs. 8-10); ``rho`` is the correlation of ``log T_0`` and
``log T_1``. ``rho = 0`` recovers the independent race that the paper actually
fit to data.

There is no within-trial noise in this model class, so ``delta_t`` and
``sigma_noise`` play no role: finishing times are drawn exactly, in continuous
time, rather than integrated on a grid.

References
----------
.. [1] Heathcote, A., & Love, J. (2012). Linear deterministic accumulator
       models of simple choice. *Frontiers in Psychology, 3*, 292.
       https://doi.org/10.3389/fpsyg.2012.00292
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Value marking "no response within max_t / deadline". Mirrors
# ``ssms.basic_simulators.simulator.OMISSION_SENTINEL``, which cannot be
# imported here because ``ssms.config`` pulls this module in while
# ``ssms.basic_simulators.simulator`` is still being initialized.
OMISSION_SENTINEL: float = -999.0

# Highest number of accumulators any registered LNR config uses (lnr4).
_MAX_ACCUMULATORS = 4


def _broadcast_param(value: Any, n_trials: int, name: str) -> np.ndarray:
    """Return ``value`` as a trial-length float array."""
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        return np.full(n_trials, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n_trials:
        return arr.astype(np.float64)
    raise ValueError(
        f"Parameter {name!r} must be scalar or length n_trials={n_trials}. "
        f"Got shape {arr.shape}."
    )


def _collect_accumulators(named: dict[str, Any], prefix: str) -> list[Any]:
    """Collect ``prefix``0, ``prefix``1, ... up to the first missing one."""
    out = []
    for i in range(_MAX_ACCUMULATORS):
        value = named.get(f"{prefix}{i}", None)
        if value is None:
            break
        out.append(value)
    return out


def lognormal_race(
    *,
    mu0,
    sigma0,
    mu1=None,
    sigma1=None,
    mu2=None,
    sigma2=None,
    mu3=None,
    sigma3=None,
    t=0.0,
    rho=None,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    delta_t: float = 0.001,
    random_state: int | None = None,
    deadline=None,
    **kwargs,
) -> dict:
    """Simulate a race between Lognormal accumulators.

    Parameters
    ----------
    mu0, mu1, ... : float or array of length ``n_trials``
        Mean of the log finishing time of each accumulator.
    sigma0, sigma1, ... : float or array of length ``n_trials``
        Standard deviation of the log finishing time of each accumulator.
        (Figure 2 of the paper quotes ``sigma**2``; pass its square root here.)
    t : float or array, default 0.0
        Shift / non-decision time in seconds (the paper's ``t0``).
    rho : float or array or None, default None
        Correlation between the log finishing times. Only supported for the
        two-accumulator case (Eq. 8). ``None`` or ``0`` is the independent race.
    n_samples, n_trials, max_t, random_state
        Standard ``ssms`` simulator arguments. ``delta_t`` is accepted and
        ignored: the LNR has no within-trial noise, so finishing times are
        sampled exactly rather than integrated on a time grid.
    deadline : float or array or None
        Per-trial deadline; responses later than it are returned as omissions.

    Returns
    -------
    dict
        ``{'rts': (n_samples, n_trials, 1), 'choices': (n_samples, n_trials, 1),
        'metadata': {...}}`` following the ``ssms`` simulator contract. Omitted
        responses carry ``OMISSION_SENTINEL`` in both ``rts`` and ``choices``.

    Notes
    -----
    Algorithm:

    1. Draw one standard normal deviate per accumulator, trial and sample.
    2. For two accumulators with ``rho != 0``, rotate the deviates with the
       Cholesky factor of ``[[1, rho], [rho, 1]]`` (Eq. 8).
    3. Map to finishing times ``T_i = exp(mu_i + sigma_i * eps_i)`` (Eq. 5).
    4. The winner is ``argmin_i T_i``; the RT is ``t + min_i T_i``.
    5. RTs beyond ``max_t`` (or ``deadline``) become omissions.

    Step 1 draws the deviates before any correlation is applied, so ``rho = 0``
    reproduces the independent race sample for sample at a given seed.

    References
    ----------
    .. [1] Heathcote, A., & Love, J. (2012). Linear deterministic accumulator
           models of simple choice. *Frontiers in Psychology, 3*, 292.

    Examples
    --------
    >>> out = lognormal_race(
    ...     mu0=-1.2, mu1=-0.5, sigma0=0.447, sigma1=0.949, t=0.4,
    ...     n_samples=5, random_state=0,
    ... )
    >>> out["rts"].shape
    (5, 1, 1)
    """
    del kwargs, delta_t

    named = {
        "mu0": mu0,
        "mu1": mu1,
        "mu2": mu2,
        "mu3": mu3,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
    }
    mus_raw = _collect_accumulators(named, "mu")
    sigmas_raw = _collect_accumulators(named, "sigma")
    if len(mus_raw) != len(sigmas_raw):
        raise ValueError(
            f"Got {len(mus_raw)} mu parameters but {len(sigmas_raw)} sigma "
            "parameters; one mu/sigma pair is needed per accumulator."
        )
    n_acc = len(mus_raw)
    if n_acc < 1:
        raise ValueError("At least one accumulator (mu0, sigma0) is required.")

    mus = np.column_stack(
        [_broadcast_param(m, n_trials, f"mu{i}") for i, m in enumerate(mus_raw)]
    )
    sigmas = np.column_stack(
        [_broadcast_param(s, n_trials, f"sigma{i}") for i, s in enumerate(sigmas_raw)]
    )
    if np.any(sigmas <= 0):
        raise ValueError("All sigma parameters must be strictly positive.")

    t_arr = _broadcast_param(t if t is not None else 0.0, n_trials, "t")

    rng = np.random.default_rng(random_state)
    # (n_samples, n_trials, n_acc) standard normal deviates, drawn identically
    # whether or not a correlation is applied, so that rho = 0 reproduces the
    # independent race sample for sample.
    eps = rng.standard_normal((n_samples, n_trials, n_acc))

    if rho is not None:
        rho_arr = _broadcast_param(rho, n_trials, "rho")
        if np.any(np.abs(rho_arr) >= 1.0):
            raise ValueError("rho must lie strictly between -1 and 1.")
        if np.any(rho_arr != 0.0):
            if n_acc != 2:
                raise ValueError(
                    "Correlated log finishing times are only implemented for the "
                    "two-accumulator case (Heathcote & Love, 2012, Eq. 8)."
                )
            # Cholesky factor of [[1, rho], [rho, 1]], applied per trial.
            eps = np.stack(
                [
                    eps[..., 0],
                    rho_arr[None, :] * eps[..., 0]
                    + np.sqrt(1.0 - rho_arr[None, :] ** 2) * eps[..., 1],
                ],
                axis=-1,
            )

    finishing_times = np.exp(mus[None, :, :] + sigmas[None, :, :] * eps)

    winners = np.argmin(finishing_times, axis=-1)
    decision_times = np.min(finishing_times, axis=-1)
    rts = t_arr[None, :] + decision_times

    # Omissions: past max_t, or past a supplied deadline.
    censor = np.full(n_trials, float(max_t), dtype=np.float64)
    if deadline is not None:
        deadline_arr = _broadcast_param(deadline, n_trials, "deadline")
        censor = np.minimum(censor, deadline_arr)
    omitted = rts > censor[None, :]

    rts_out = np.where(omitted, OMISSION_SENTINEL, rts).astype(np.float32)
    choices_out = np.where(omitted, OMISSION_SENTINEL, winners).astype(np.int64)

    metadata: dict[str, Any] = {
        "simulator": "lognormal_race",
        "possible_choices": list(range(n_acc)),
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_accumulators": n_acc,
        "max_t": max_t,
        "t": t_arr.astype(np.float32),
        "deadline": None if deadline is None else np.asarray(deadline),
    }
    for i in range(n_acc):
        metadata[f"mu_{i}"] = mus[:, i].astype(np.float32)
        metadata[f"sigma_{i}"] = sigmas[:, i].astype(np.float32)
    if rho is not None:
        metadata["rho"] = np.asarray(rho, dtype=np.float32)

    return {
        "rts": rts_out[:, :, None],
        "choices": choices_out[:, :, None],
        "metadata": metadata,
    }
