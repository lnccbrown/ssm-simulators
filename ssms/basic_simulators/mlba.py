"""MLBA simulator: MLBA front end + standard (non-folded) LBA back end.

The back end is the linear ballistic accumulator of Brown & Heathcote (2008)
as used by Trueblood, Brown & Heathcote (2014): accumulator ``i`` starts at
``k_i ~ U(0, A)``, rises at a rate ``v_i ~ N(d_i, sd)`` drawn once per trial,
and reaches threshold ``b`` at ``(b - k_i) / v_i``. A rate ``v_i <= 0`` never
reaches threshold. The first accumulator to finish gives the choice; the RT is
``t + min_i (b - k_i) / v_i``.

This differs from the library's ``cssm.lba_vanilla``, which takes ``|v_i|``
(a folded normal) so that every accumulator eventually finishes. The paper's
model needs the untruncated normal: footnote 3 notes that negative mean drifts
occur and only one positive rate is needed to terminate, and its fitted
parameters (Table 6, similarity: ``I0 = 0.31``) put substantial mass on
negative rates, where folding changes the predicted choice probabilities.

Omission contract (as ``ssms.basic_simulators.lnr``): a trial is an omission
when no accumulator ever finishes (all sampled rates ``<= 0``), when the
decision time exceeds ``max_t``, or when the RT (``t`` included) is at or after
``deadline``. Only ``rts`` carries ``OMISSION_SENTINEL``; ``choices`` keeps the
latent choice: the eventual winner if any accumulator finishes, else the
accumulator that leads (highest ``k_i + v_i * max_t``) at ``max_t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mlba_frontend import mlba_drifts

OMISSION_SENTINEL: float = -999.0
_MAX_OPTIONS = 4


def _vec(value: Any, n_trials: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == n_trials:  # e.g. (n_trials, n_particles)
        arr = arr[:, 0]
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full(n_trials, arr.item())
    if arr.size == n_trials:
        return arr
    raise ValueError(
        f"Parameter {name!r} must be scalar or length n_trials={n_trials}; "
        f"got shape {np.shape(value)}."
    )


def lba_race_from_drifts(
    drifts: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    sd: np.ndarray,
    t: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    max_t: float = 20.0,
    deadline: np.ndarray | None = None,
):
    """Standard LBA race for given mean drifts, shape ``(n_trials, n_acc)``.

    Returns ``(rts, choices)`` of shape ``(n_samples, n_trials)``.
    """
    n_trials, n_acc = drifts.shape
    starts = rng.uniform(size=(n_samples, n_trials, n_acc)) * A[None, :, None]
    rates = drifts[None] + sd[None, :, None] * rng.standard_normal(
        (n_samples, n_trials, n_acc)
    )
    dist = b[None, :, None] - starts
    with np.errstate(divide="ignore", invalid="ignore"):
        finish = np.where(rates > 0, dist / rates, np.inf)
    winners = np.argmin(finish, axis=-1)
    dec_time = np.min(finish, axis=-1)
    never = ~np.isfinite(dec_time)
    if np.any(never):
        leading = np.argmax(starts + rates * max_t, axis=-1)
        winners = np.where(never, leading, winners)
    rts = t[None, :] + np.where(never, 0.0, dec_time)
    omitted = never | (dec_time > max_t)
    if deadline is not None:
        omitted |= rts >= deadline[None, :]
    rts = np.where(omitted, OMISSION_SENTINEL, rts)
    return rts, winners


def mlba_race(
    *,
    m,
    lambda1,
    lambda2,
    I0,
    A,
    b,
    P0,
    Q0,
    P1,
    Q1,
    P2=None,
    Q2=None,
    P3=None,
    Q3=None,
    beta=1.0,
    t=0.0,
    sd=None,
    deadline=None,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    random_state: int | None = None,
    **kwargs,
) -> dict:
    """Simulate the multiattribute linear ballistic accumulator (MLBA).

    Parameters
    ----------
    m, lambda1, lambda2, I0, beta
        Front-end parameters (curvature, attention-weight decay for positive /
        negative differences, baseline input, attribute bias).
    A, b
        Start-point range and threshold (paper's ``A`` and ``chi``); ``b >= A``.
    P0, Q0, P1, Q1, ... : attribute values of options 0, 1, ...
    t
        Non-decision time added to every RT.
    sd
        Between-trial SD of the drift rates (the paper's ``s``). The ``ssms``
        ``simulator`` fills it from ``sigma_noise`` (default 0.1 for LBA-type
        model names), like ``lba3``.
    deadline, n_samples, n_trials, max_t, random_state
        Standard ``ssms`` simulator arguments. ``delta_t`` and friends are
        accepted and ignored: the race is sampled exactly in continuous time.

    Returns
    -------
    dict with ``rts`` and ``choices`` of shape ``(n_samples, n_trials, 1)`` and
    ``metadata`` (``possible_choices`` = option indices, mean ``drifts``).
    """
    del kwargs
    named = {
        "P0": P0,
        "Q0": Q0,
        "P1": P1,
        "Q1": Q1,
        "P2": P2,
        "Q2": Q2,
        "P3": P3,
        "Q3": Q3,
    }
    Ps, Qs = [], []
    for i in range(_MAX_OPTIONS):
        p, q = named[f"P{i}"], named[f"Q{i}"]
        if p is None and q is None:
            break
        if p is None or q is None:
            raise ValueError(f"Option {i} needs both P{i} and Q{i}.")
        Ps.append(_vec(p, n_trials, f"P{i}"))
        Qs.append(_vec(q, n_trials, f"Q{i}"))
    n_opt = len(Ps)
    if n_opt < 2:
        raise ValueError("The MLBA needs at least two options.")
    P = np.column_stack(Ps)
    Q = np.column_stack(Qs)

    A_ = _vec(A, n_trials, "A")
    b_ = _vec(b, n_trials, "b")
    t_ = _vec(t if t is not None else 0.0, n_trials, "t")
    sd_ = _vec(0.1 if sd is None else sd, n_trials, "sd")
    if np.any(A_ < 0):
        raise ValueError("A must be non-negative.")
    if np.any(b_ < A_):
        raise ValueError("The threshold b must not be below the start-point range A.")
    if np.any(sd_ < 0):
        raise ValueError("sd must be non-negative.")
    dl = None if deadline is None else _vec(deadline, n_trials, "deadline")

    drifts = mlba_drifts(
        P,
        Q,
        _vec(m, n_trials, "m"),
        _vec(lambda1, n_trials, "lambda1"),
        _vec(lambda2, n_trials, "lambda2"),
        _vec(I0, n_trials, "I0"),
        _vec(beta if beta is not None else 1.0, n_trials, "beta"),
    )

    rng = np.random.default_rng(random_state)
    rts, choices = lba_race_from_drifts(
        drifts, A_, b_, sd_, t_, n_samples, rng, max_t=max_t, deadline=dl
    )

    metadata: dict[str, Any] = {
        "simulator": "mlba_race",
        "possible_choices": list(range(n_opt)),
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_options": n_opt,
        "max_t": max_t,
        "drifts": drifts.astype(np.float32),
        "A": A_.astype(np.float32),
        "b": b_.astype(np.float32),
        "sd": sd_.astype(np.float32),
        "t": t_.astype(np.float32),
        "deadline": None if dl is None else dl.astype(np.float32),
    }
    return {
        "rts": rts.astype(np.float32)[:, :, None],
        "choices": choices.astype(np.int64)[:, :, None],
        "metadata": metadata,
    }
