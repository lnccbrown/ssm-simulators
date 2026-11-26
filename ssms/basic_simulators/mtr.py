"""Multiple-Threshold Race (MTR) simulator: an LBA race with don't-know thresholds.

Reynolds, Garton, Kvam, Sauer, Osth & Heathcote, "A dynamic model of deciding
not to choose". Two linear ballistic accumulators race, one per definitive
response. Accumulator ``i`` starts at ``k_i ~ U(0, A)`` and rises at a rate
``r_i ~ N(v_i, sv_i**2)`` truncated to positive values (Heathcote & Love,
2012), so that it reaches its choice threshold ``b_i`` at

    T_i = (b_i - k_i) / r_i.

The first accumulator to reach its choice threshold fixes the response time,
``RT = t + min_i T_i``. Each accumulator also has a lower, "don't-know"
threshold ``d_i`` with ``0 <= d_i <= b_i``. The response rule (paper, p. 4):
the winner ``w`` gives its own definitive response only if the *loser* ``l``
is still below the loser's don't-know threshold at the moment of choice,

    k_l + r_l * T_w < d_l   ->  definitive response w
    otherwise               ->  don't-know response.

Parameterisation follows the paper's estimation scheme (p. 5): ``b_i = A +
B_i`` with ``B_i > 0`` (so ``A < b_i`` and no decision is instantaneous) and
``d_i = D_i * b_i`` with ``0 <= D_i <= 1``. The paper reports ``DK_i = 1 -
D_i``; ``D_i = 1`` switches the don't-know mechanism off for losses of
accumulator ``i`` and ``D_i = 0`` turns every loss of ``i`` into a don't-know.

Choice coding: ``0`` and ``1`` are the definitive responses of accumulators 0
and 1, ``2`` is the don't-know response. The two latent don't-know states
(either accumulator winning) are not distinguished in the output, exactly as
in the paper's manifest data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

# Mirrors ``ssms.basic_simulators.simulator.OMISSION_SENTINEL`` (not imported
# here to keep this module free of the config <-> simulator import cycle).
OMISSION_SENTINEL: float = -999.0
DONT_KNOW: int = 2
POSSIBLE_CHOICES: list[int] = [0, 1, DONT_KNOW]


def _broadcast(value: Any, n_trials: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        return np.full(n_trials, float(arr))
    if arr.ndim == 1 and arr.shape[0] == n_trials:
        return arr.astype(np.float64)
    raise ValueError(
        f"Parameter {name!r} must be scalar or length n_trials={n_trials}; "
        f"got shape {arr.shape}."
    )


def sample_truncated_normal_rates(
    u: np.ndarray, mean: np.ndarray, sd: np.ndarray
) -> np.ndarray:
    """Map uniforms ``u`` to ``N(mean, sd**2)`` truncated to ``(0, inf)``.

    Inverse-CDF sampling, computed from the upper tail for accuracy:
    ``r = mean + sd * z`` where ``P(Z > z) = (1 - u) * P(Z > -mean/sd)``.
    """
    upper_mass = ndtr(mean / sd)  # P(Z > -mean/sd) = P(rate > 0)
    z = -ndtri((1.0 - u) * upper_mass)
    rates = mean + sd * z
    # Guard the round-off case u -> 1 (rate -> 0+) without introducing bias.
    return np.maximum(rates, np.finfo(np.float64).tiny)


def multiple_threshold_race(
    *,
    A,
    B0,
    B1,
    D0,
    D1,
    v0,
    v1,
    sv0,
    sv1,
    t=0.0,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    delta_t: float = 0.001,
    random_state: int | None = None,
    deadline=None,
    return_latent: bool = False,
    **kwargs,
) -> dict:
    """Simulate the two-accumulator Multiple-Threshold Race.

    Parameters
    ----------
    A : start-point range, ``k_i ~ U(0, A)`` (shared by both accumulators).
    B0, B1 : ``b_i - A``, distance from the top of the start range to the
        choice threshold of accumulator ``i`` (``B_i > 0``).
    D0, D1 : ``d_i / b_i`` in ``[0, 1]``; the paper's ``DK_i = 1 - D_i``.
    v0, v1 : mean rate of accumulator ``i`` (before truncation).
    sv0, sv1 : between-trial SD of the rate of accumulator ``i`` (> 0).
    t : non-decision time in seconds (the paper's ``t0``).
    n_samples, n_trials, max_t, random_state : standard ``ssms`` arguments.
        ``max_t`` bounds the decision time (as in ``lognormal_race``): a race
        not won by then is an omission. ``delta_t``, ``s``/``sd`` and the other
        generic simulator keywords are accepted and ignored: finishing times
        are sampled exactly (no time grid, no within-trial noise).
    deadline : optional per-trial deadline on the RT (``t`` included);
        responses at or after it are omissions.
    return_latent : also return the latent race winner (``'winners'``).

    Returns
    -------
    dict with ``'rts'`` and ``'choices'`` of shape ``(n_samples, n_trials, 1)``
    and ``'metadata'``. Omitted trials have ``rts == -999`` and keep the
    response the race would have produced (0, 1 or 2) in ``choices``.
    """
    del kwargs, delta_t
    p = {
        name: _broadcast(val, n_trials, name)
        for name, val in {
            "A": A,
            "B0": B0,
            "B1": B1,
            "D0": D0,
            "D1": D1,
            "v0": v0,
            "v1": v1,
            "sv0": sv0,
            "sv1": sv1,
            "t": 0.0 if t is None else t,
        }.items()
    }
    if np.any(p["A"] < 0):
        raise ValueError("A must be non-negative.")
    if np.any(p["B0"] <= 0) or np.any(p["B1"] <= 0):
        raise ValueError("B0 and B1 must be strictly positive (b_i > A).")
    for name in ("D0", "D1"):
        if np.any((p[name] < 0) | (p[name] > 1)):
            raise ValueError(f"{name} must lie in [0, 1].")
    if np.any(p["sv0"] <= 0) or np.any(p["sv1"] <= 0):
        raise ValueError("sv0 and sv1 must be strictly positive.")

    A_ = p["A"][None, :, None]
    b = np.stack([p["A"] + p["B0"], p["A"] + p["B1"]], axis=-1)[None]  # (1,T,2)
    d = np.stack([p["D0"], p["D1"]], axis=-1)[None] * b
    v = np.stack([p["v0"], p["v1"]], axis=-1)[None]
    sv = np.stack([p["sv0"], p["sv1"]], axis=-1)[None]

    rng = np.random.default_rng(random_state)
    # One start-point and one rate uniform per accumulator, sample and trial.
    u_start = rng.random((n_samples, n_trials, 2))
    u_rate = rng.random((n_samples, n_trials, 2))
    starts = A_ * u_start
    rates = sample_truncated_normal_rates(u_rate, v, sv)

    finish = (b - starts) / rates  # (S, T, 2)
    winner = np.argmin(finish, axis=-1)  # ties: accumulator 0 (measure zero)
    dec_time = np.take_along_axis(finish, winner[..., None], axis=-1)[..., 0]
    loser = 1 - winner
    loser_pos = (
        np.take_along_axis(starts, loser[..., None], axis=-1)[..., 0]
        + np.take_along_axis(rates, loser[..., None], axis=-1)[..., 0] * dec_time
    )
    loser_d = np.take_along_axis(
        np.broadcast_to(d, finish.shape), loser[..., None], axis=-1
    )[..., 0]
    dont_know = loser_pos >= loser_d
    choices = np.where(dont_know, DONT_KNOW, winner).astype(np.int64)

    rts = p["t"][None, :] + dec_time
    omitted = dec_time > max_t
    if deadline is not None:
        omitted |= rts >= _broadcast(deadline, n_trials, "deadline")[None, :]
    # Contract: censor rts only; the latent response stays in choices.
    rts_out = np.where(omitted, OMISSION_SENTINEL, rts).astype(np.float32)

    metadata: dict[str, Any] = {
        "simulator": "multiple_threshold_race",
        "possible_choices": list(POSSIBLE_CHOICES),
        "choice_labels": {0: "accumulator 0", 1: "accumulator 1", 2: "don't know"},
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_accumulators": 2,
        "max_t": max_t,
        "deadline": None if deadline is None else np.asarray(deadline),
    }
    for name, arr in p.items():
        metadata[name] = arr.astype(np.float32)

    out = {
        "rts": rts_out[:, :, None],
        "choices": choices[:, :, None],
        "metadata": metadata,
    }
    if return_latent:
        out["winners"] = winner[:, :, None].astype(np.int64)
    return out
