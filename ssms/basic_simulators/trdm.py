"""Timed Racing Diffusion Model (TRDM) simulator.

The TRDM (Hawkins & Heathcote, 2021) is a race between ``N`` one-boundary
diffusion processes that accumulate evidence, one per response, and a single
one-boundary diffusion process that accumulates *time* (an internal timer).
The first process to reach its threshold ends the trial:

* an evidence accumulator wins -> its response is made;
* the timer wins -> a stimulus-uninformed guess, uniformly over the ``N``
  responses (the paper's default ``g_n = 1/N``, Eq. 1), or, for two
  responses, with a guess probability ``g`` for response 0 (the paper's
  ``g_{T,1}``; Part 2 guessing variants, Part 3 go/no-go).

Every process is a Wiener process with constant drift that starts at a fixed
point and has no trial-to-trial variability in the paper, so each finishing
time is exactly (shifted) inverse Gaussian / Wald, Eqs. 2-3. The simulator
therefore draws the finishing times directly, in continuous time, instead of
integrating the processes on a ``delta_t`` grid. This is exact for any drift
sign and noise level:

* ``v > 0``  : first-passage time ``~ IG(mean=d/v, shape=d**2/s**2)``;
* ``v = 0``  : Levy, ``d**2 / (s**2 Z**2)`` (the ``v -> 0`` limit of the same
  sampler);
* ``v < 0``  : the threshold is reached with probability ``exp(2 v d / s**2)``,
  and when it is, the passage time has the law of the ``|v|`` case;
  otherwise it is never reached (``inf``);
* ``s = 0``  : deterministic ``d / v`` (``inf`` if ``v <= 0``);
* ``d <= 0`` : the process starts on/above threshold and finishes at once.

Units. The evidence accumulators use the library's noise convention: their
diffusion coefficient is the ``s`` the ``ssms`` simulator passes in
(``sigma_noise``, default 1.0), exactly as for ``racing_diffusion_3``. With
``s = 1`` the paper's evidence parameters map as ``v_i = gamma_i / sigma_E``
and ``b_i = alpha_i / sigma_E`` (a Wiener process divided by ``sigma_E`` is a
unit-noise Wiener process). The timer keeps the paper's own scale: threshold
``alpha_T = 1`` (fixed, as in the paper), drift ``vT = gamma_T``, diffusion
coefficient ``sT = sigma_T`` and onset ``tT = tau_T``. ``vT = sT = 0`` switches
the timer off (it never reaches threshold); with ``sT = 0, vT > 0`` the timer
is a deterministic internal deadline at ``tT + 1/vT``.

Omissions follow the library contract (``tests/test_omission_labels.py``): a
trial whose response falls at or after ``deadline`` (RT, onset included), or
whose winning finishing time exceeds ``t + max_t`` (``max_t`` bounds the
decision time measured from the evidence onset ``t``, as
``compute_deadline_tmp`` does in the Cython kernels), gets
``OMISSION_SENTINEL`` in ``rts`` only. ``choices`` keeps the latent response of
the race (the winning accumulator, or the timer's guess), so every returned
choice is in ``possible_choices`` and ``choice_p`` sums to one.

References
----------
Hawkins, G. E., & Heathcote, A. (2021). Racing against the clock: Evidence-based
versus time-based decisions. *Psychological Review, 128*(2), 222-263.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Mirrors ``ssms.basic_simulators.simulator.OMISSION_SENTINEL``.
OMISSION_SENTINEL: float = -999.0

#: Threshold of the timing process. Fixed at 1 for identification, as in the
#: paper ("we set the value of the response threshold to alpha = 1 in the
#: evidence and timing processes").
TIMER_THRESHOLD: float = 1.0

_MAX_ACCUMULATORS = 8


def _as_trials(value: Any, n_trials: int, name: str) -> np.ndarray:
    """Return ``value`` as a float64 array of length ``n_trials``."""
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        return np.full(n_trials, float(arr))
    if arr.ndim == 1 and arr.shape[0] == n_trials:
        return arr.astype(np.float64)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return np.full(n_trials, float(arr[0]))
    raise ValueError(
        f"Parameter {name!r} must be scalar or of length n_trials={n_trials}; "
        f"got shape {np.shape(value)}."
    )


def first_passage_times(
    v: np.ndarray,
    s: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    u_accept: np.ndarray,
    u_hit: np.ndarray,
) -> np.ndarray:
    """Exact first-passage times of ``x(t) = v t + s W(t)`` to level ``d``.

    All inputs broadcast to a common shape. ``z`` is standard normal,
    ``u_accept`` and ``u_hit`` are Uniform(0, 1); the same three variates are
    used whatever the case, so the draw structure does not depend on the
    parameters (a given seed gives the same stream for every parameter set).

    The positive-drift case is the Michael-Schucany-Haas (1976) transformation
    for the inverse Gaussian ``IG(mu = d/v, lambda = d**2/s**2)``, written in
    terms of the drift so that it is numerically stable for small ``v`` and has
    the Levy law as its ``v = 0`` limit::

        q = s**2 z**2,  c = 2 d |v|
        x1 = 2 d**2 / (c + q + sqrt(q**2 + 2 q c))      # smaller root
        x2 = (c + q + sqrt(q**2 + 2 q c)) / (2 v**2)    # = mu**2 / x1
        T  = x1 if u_accept <= d / (d + |v| x1) else x2

    For ``v < 0`` the passage happens with probability ``exp(2 v d / s**2)``
    and, conditionally on it, ``T`` has the law of the ``|v|`` case.
    """
    v, s, d, z, u_accept, u_hit = np.broadcast_arrays(v, s, d, z, u_accept, u_hit)
    av = np.abs(v)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        q = (s * z) ** 2
        c = 2.0 * d * av
        root = c + q + np.sqrt(q * q + 2.0 * q * c)
        x1 = np.where(root > 0, 2.0 * d * d / root, np.inf)
        x2 = np.where(av > 0, root / (2.0 * av * av), np.inf)
        p_small = np.where(np.isfinite(x1), d / (d + av * x1), 0.0)
        fpt = np.where(u_accept <= p_small, x1, x2)

        # Defective passage for negative drift (s > 0); with s = 0 and v < 0 the
        # level is never reached, which ``root = c > 0`` alone would miss.
        p_hit = np.where(
            v < 0,
            np.where(
                s > 0,
                np.exp(np.minimum(2.0 * v * d / np.maximum(s * s, 1e-300), 0.0)),
                0.0,
            ),
            1.0,
        )
        fpt = np.where(u_hit < p_hit, fpt, np.inf)
    # Starting on or above the threshold: immediate passage.
    fpt = np.where(d <= 0, 0.0, fpt)
    return fpt


def simulate_trdm_latent(
    v: np.ndarray,
    b: np.ndarray,
    t: np.ndarray,
    s: np.ndarray,
    vT: np.ndarray,
    sT: np.ndarray,
    tT: np.ndarray,
    A: np.ndarray | None,
    n_samples: int,
    rng: np.random.Generator,
    g: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the race and return the latent outcome of every sample.

    Parameters
    ----------
    v, b : (n_trials, N) arrays
        Evidence drift rates and thresholds per accumulator.
    t, s, vT, sT, tT : (n_trials,) arrays
        Evidence onset, evidence diffusion coefficient, timer drift, timer
        diffusion coefficient and timer onset.
    A : (n_trials,) array or None
        Upper bound of a Uniform(0, A) evidence start point (``None``: start
        at 0, the paper's TRDM).
    g : (n_trials,) array or None
        Two accumulators only: probability that a timer-terminated trial is
        answered with response 0 (the paper's ``g_{T,1}``, Eq. 1). ``None``
        is the paper's default unbiased guess ``1/N``. The guess is
        ``0 if u < g else 1`` with the same uniform ``u`` the unbiased guess
        uses (``floor(u N)``), so ``g = 0.5`` reproduces ``g = None`` sample
        for sample.

    Returns
    -------
    dict with ``rt`` (n_samples, n_trials; ``inf`` when nothing ever
    finishes), ``choice`` (latent response index), ``timer_won`` (bool),
    ``evidence_rt`` (finishing time incl. onset of the winning evidence
    accumulator, ``inf`` if none).

    Draw order (fixed): for each evidence accumulator in index order, then for
    the timer: ``z``, ``u_accept``, ``u_hit`` (each ``(n_samples, n_trials)``);
    then the start points (if ``A`` is given, one uniform per accumulator);
    then one uniform for the timer's guess.
    """
    n_trials, n_acc = v.shape
    shape = (n_samples, n_trials)

    draws = []
    for _ in range(n_acc + 1):
        draws.append(
            (
                rng.standard_normal(shape),
                rng.random(shape),
                rng.random(shape),
            )
        )
    if A is not None:
        start = rng.random((n_acc,) + shape) * A[None, None, :]
    else:
        start = np.zeros((n_acc,) + shape)
    u_guess = rng.random(shape)

    finish = np.empty((n_acc + 1,) + shape)
    for i in range(n_acc):
        z, ua, uh = draws[i]
        d = b[None, :, i] - start[i]
        fpt = first_passage_times(v[None, :, i], s[None, :], d, z, ua, uh)
        finish[i] = t[None, :] + fpt
    z, ua, uh = draws[n_acc]
    fpt_timer = first_passage_times(
        vT[None, :], sT[None, :], np.full(shape, TIMER_THRESHOLD), z, ua, uh
    )
    finish[n_acc] = tT[None, :] + fpt_timer

    winner = np.argmin(finish, axis=0)  # ties -> lowest index (evidence first)
    rt = np.min(finish, axis=0)
    if g is None:
        guess = np.minimum((u_guess * n_acc).astype(np.int64), n_acc - 1)
    else:
        if n_acc != 2:
            raise ValueError(
                "A guess probability g is only defined for two accumulators."
            )
        guess = np.where(u_guess < g[None, :], 0, 1).astype(np.int64)
    timer_won = winner == n_acc
    nothing_finished = ~np.isfinite(rt)
    choice = np.where(timer_won | nothing_finished, guess, winner).astype(np.int64)
    evidence_rt = np.min(finish[:n_acc], axis=0)
    return {
        "rt": rt,
        "choice": choice,
        "timer_won": timer_won & ~nothing_finished,
        "evidence_rt": evidence_rt,
    }


def _collect(named: dict[str, Any], prefix: str) -> list[Any]:
    out = []
    for i in range(_MAX_ACCUMULATORS):
        value = named.get(f"{prefix}{i}")
        if value is None:
            break
        out.append(value)
    return out


def timed_racing_diffusion(
    *,
    t,
    vT,
    sT,
    tT,
    b=None,
    A=None,
    g=None,
    s=1.0,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    delta_t: float = 0.001,
    random_state: int | None = None,
    deadline=None,
    return_option: str = "full",
    **kwargs,
) -> dict:
    """``ssms`` simulator for the Timed Racing Diffusion Model.

    Accumulator parameters are passed as numbered keywords: drifts
    ``v0, v1, ...`` and either per-accumulator thresholds ``b0, b1, ...``
    (``trdm_2``) or one shared threshold ``b`` (``trdm_3``). ``A`` (optional)
    is the bound of a Uniform(0, A) start point shared by the evidence
    accumulators (``trdm_3``; the paper's model has ``A = 0``). ``g``
    (optional, two accumulators, ``trdm_2_guess``) is the probability that a
    timer-terminated trial gives response 0; omitted, the guess is unbiased.

    ``s`` is the evidence diffusion coefficient injected by
    ``ssms.basic_simulators.simulator`` (``sigma_noise``, default 1.0).
    ``delta_t`` and ``smooth_unif`` are accepted and ignored: finishing times
    are drawn exactly, in continuous time. ``random_state`` seeds a NumPy
    ``default_rng``; it is not comparable with the Cython-backed models'
    streams.

    Returns ``{'rts', 'choices', 'metadata'}`` with arrays of shape
    ``(n_samples, n_trials, 1)``; see the module docstring for omissions.
    """
    named = {
        k: kwargs.get(k) for k in kwargs if k[:1] in ("v", "b") and k[1:].isdigit()
    }
    vs = _collect(named, "v")
    n_acc = len(vs)
    if n_acc < 2:
        raise ValueError("The TRDM needs at least two evidence accumulators (v0, v1).")
    bs = _collect(named, "b")
    if bs:
        if len(bs) != n_acc:
            raise ValueError(f"Got {n_acc} drifts but {len(bs)} thresholds.")
    elif b is not None:
        bs = [b] * n_acc
    else:
        raise ValueError("Pass either a shared threshold b or b0, b1, ...")

    v_arr = np.column_stack(
        [_as_trials(x, n_trials, f"v{i}") for i, x in enumerate(vs)]
    )
    b_arr = np.column_stack(
        [_as_trials(x, n_trials, f"b{i}") for i, x in enumerate(bs)]
    )
    t_arr = _as_trials(t, n_trials, "t")
    # ``simulator()`` passes the noise as an (n_trials, n_particles) tile of
    # one value per trial (``make_noise_vec``); one column carries it.
    s_in = np.asarray(s, dtype=np.float64)
    if s_in.ndim == 2:
        if not np.allclose(s_in, s_in[:, :1]):
            raise ValueError("s must be shared by the evidence accumulators.")
        s_in = s_in[:, 0]
    s_arr = _as_trials(s_in, n_trials, "s")
    vT_arr = _as_trials(vT, n_trials, "vT")
    sT_arr = _as_trials(sT, n_trials, "sT")
    tT_arr = _as_trials(tT, n_trials, "tT")
    A_arr = None if A is None else _as_trials(A, n_trials, "A")
    g_arr = None if g is None else _as_trials(g, n_trials, "g")
    if g_arr is not None and np.any((g_arr < 0) | (g_arr > 1)):
        raise ValueError("g must lie in [0, 1].")

    for name, arr in (("s", s_arr), ("sT", sT_arr), ("t", t_arr), ("tT", tT_arr)):
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite and non-negative.")
    if np.any(vT_arr < 0):
        raise ValueError("vT (timer drift) must be non-negative.")
    if A_arr is not None and np.any(A_arr < 0):
        raise ValueError("A must be non-negative.")

    rng = np.random.default_rng(random_state)
    latent = simulate_trdm_latent(
        v_arr,
        b_arr,
        t_arr,
        s_arr,
        vT_arr,
        sT_arr,
        tT_arr,
        A_arr,
        n_samples,
        rng,
        g=g_arr,
    )
    rt = latent["rt"]

    omitted = ~np.isfinite(rt) | (rt - t_arr[None, :] > max_t)
    if deadline is not None:
        deadline_arr = _as_trials(deadline, n_trials, "deadline")
        omitted |= rt >= deadline_arr[None, :]
    rts_out = np.where(omitted, OMISSION_SENTINEL, rt).astype(np.float32)
    # Only rts carry the sentinel; the latent response stays in choices.
    choices_out = latent["choice"].astype(np.int64)

    metadata: dict[str, Any] = {
        "simulator": "timed_racing_diffusion",
        "possible_choices": list(range(n_acc)),
        "n_samples": n_samples,
        "n_trials": n_trials,
        "max_t": max_t,
        "delta_t": delta_t,
    }
    if return_option == "full":
        metadata.update(
            {
                "v": v_arr.astype(np.float32),
                "b": b_arr.astype(np.float32),
                "t": t_arr.astype(np.float32),
                "s": s_arr.astype(np.float32),
                "vT": vT_arr.astype(np.float32),
                "sT": sT_arr.astype(np.float32),
                "tT": tT_arr.astype(np.float32),
                "timer_threshold": TIMER_THRESHOLD,
                "A": None if A_arr is None else A_arr.astype(np.float32),
                "g": None if g_arr is None else g_arr.astype(np.float32),
                "deadline": None if deadline is None else np.asarray(deadline),
            }
        )
    elif return_option != "minimal":
        raise ValueError('return_option must be either "full" or "minimal"')

    return {
        "rts": rts_out[:, :, None],
        "choices": choices_out[:, :, None],
        "metadata": metadata,
    }
