"""Multiple-threshold (correlated) Lognormal race: choice + confidence.

Reynolds, Kvam, Osth & Heathcote (2020) extend the correlated two-accumulator
Lognormal race (the library's ``lnr2_corr``) with *confidence thresholds*
("multiple-threshold log-normal race", MTLNR). The choice is made exactly as in
``lnr2_corr``: the first accumulator to reach its (choice) threshold wins and
RT = ``t + min(T_0, T_1)``. Confidence is read off the *losing* accumulator at
that moment: because accumulation is deterministic, the loser has covered the
fraction ``dt / T_loser`` of its distance, so it lies below a lower threshold
placed at the fraction ``d`` of its choice threshold iff ``T_loser > dt / d``
(paper p. 7, Eqs. 6-7). Each accumulator ``i`` has ``K`` such thresholds
``d1acc{i} < d2acc{i} < ... < 1``; they are used when ``i`` loses.

The confidence level is the number of the loser's thresholds that its evidence
total is below::

    conf = sum_k [ d_k(loser) * T_loser > dt ]        in {0, ..., K}

so ``conf = K`` is the highest confidence (loser below its lowest threshold,
``d1``) and ``conf = 0`` the lowest (loser above every confidence threshold).

Output coding
-------------
``choices = winner + 2 * conf`` with ``winner`` in {0, 1}. Hence
``choices % 2`` is the accumulator that won (the ``lnr2``/``lnr2_corr`` choice)
and ``choices // 2`` is the confidence level. With every ``d`` set to 0 no
loser is ever below a threshold, ``conf = 0`` always, and the output is
sample-for-sample that of ``lnr2_corr``: the normal deviates are drawn and
rotated with exactly the same calls as ``ssms.basic_simulators.lnr``.

Omissions follow ``lognormal_race``: ``max_t`` bounds the decision time,
``deadline`` bounds the RT (shift included); censored trials get
``OMISSION_SENTINEL`` in ``rts`` only and keep their latent (winner,
confidence) label in ``choices``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

OMISSION_SENTINEL: float = -999.0

#: Largest number of confidence thresholds per accumulator the kernel collects.
MAX_CONF_THRESHOLDS = 4


def _broadcast_param(value: Any, n_trials: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        return np.full(n_trials, float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n_trials:
        return arr.astype(np.float64)
    raise ValueError(
        f"Parameter {name!r} must be scalar or length n_trials={n_trials}. "
        f"Got shape {arr.shape}."
    )


def _collect_thresholds(named: dict[str, Any], acc: int) -> list[Any]:
    out = []
    for k in range(1, MAX_CONF_THRESHOLDS + 1):
        value = named.get(f"d{k}acc{acc}", None)
        if value is None:
            break
        out.append(value)
    return out


def decode_choices(choices) -> tuple[np.ndarray, np.ndarray]:
    """Split MTLNR labels into (winning accumulator, confidence level)."""
    c = np.asarray(choices)
    return c % 2, c // 2


def multiple_threshold_lognormal_race(
    *,
    mu0,
    mu1,
    sigma0,
    sigma1,
    rho=0.0,
    t=0.0,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    delta_t: float = 0.001,
    random_state: int | None = None,
    deadline=None,
    **kwargs,
) -> dict:
    """Simulate the two-accumulator multiple-threshold Lognormal race.

    Parameters
    ----------
    mu0, mu1, sigma0, sigma1 : float or array (n_trials,)
        Mean / SD of ``log T_i`` (as in ``lnr2_corr``).
    rho : float or array
        Correlation of ``log T_0`` and ``log T_1``; ``|rho| < 1``.
    t : float or array
        Non-decision time (s).
    **kwargs
        Confidence thresholds ``d1acc0, d2acc0, ..., d1acc1, d2acc1, ...``
        (float or array (n_trials,), keyword-only). Confidence thresholds of accumulator 0 / 1 as fractions of its choice
        threshold, in [0, 1]. Both accumulators must have the same number
        ``K`` of thresholds (``K = 0`` gives ``lnr2_corr``). The count is
        order-invariant, so the thresholds need not be sorted; sorted
        (``d1 < d2``) is the identifiable labelling.
    n_samples, n_trials, max_t, random_state, deadline
        Standard ``ssms`` arguments; ``delta_t`` and noise arguments are
        ignored (no within-trial noise, exact sampling).

    Returns
    -------
    dict with ``rts`` and ``choices`` of shape (n_samples, n_trials, 1) and
    ``metadata``; ``possible_choices = range(2 * (K + 1))``.
    """
    named = {k: v for k, v in kwargs.items() if k.startswith("d")}
    del delta_t

    th_raw = [_collect_thresholds(named, 0), _collect_thresholds(named, 1)]
    if len(th_raw[0]) != len(th_raw[1]):
        raise ValueError(
            "Both accumulators need the same number of confidence thresholds; "
            f"got {len(th_raw[0])} and {len(th_raw[1])}."
        )
    n_thr = len(th_raw[0])
    n_levels = n_thr + 1

    mus = np.column_stack(
        [_broadcast_param(mu0, n_trials, "mu0"), _broadcast_param(mu1, n_trials, "mu1")]
    )
    sigmas = np.column_stack(
        [
            _broadcast_param(sigma0, n_trials, "sigma0"),
            _broadcast_param(sigma1, n_trials, "sigma1"),
        ]
    )
    if np.any(sigmas <= 0):
        raise ValueError("All sigma parameters must be strictly positive.")
    t_arr = _broadcast_param(t if t is not None else 0.0, n_trials, "t")

    # thresholds: (n_trials, 2 accumulators, K)
    if n_thr:
        thr = np.stack(
            [
                np.column_stack(
                    [
                        _broadcast_param(v, n_trials, f"d{k + 1}acc{i}")
                        for k, v in enumerate(th_raw[i])
                    ]
                )
                for i in range(2)
            ],
            axis=1,
        )
        if np.any((thr < 0.0) | (thr > 1.0)):
            raise ValueError("Confidence thresholds must lie in [0, 1].")
    else:
        thr = np.zeros((n_trials, 2, 0))

    # --- identical draw to ssms.basic_simulators.lnr.lognormal_race --------
    rng = np.random.default_rng(random_state)
    eps = rng.standard_normal((n_samples, n_trials, 2))
    rho_arr = _broadcast_param(rho if rho is not None else 0.0, n_trials, "rho")
    if np.any(np.abs(rho_arr) >= 1.0):
        raise ValueError("rho must lie strictly between -1 and 1.")
    if np.any(rho_arr != 0.0):
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
    # ----------------------------------------------------------------------

    loser_times = np.max(finishing_times, axis=-1)
    losers = 1 - winners
    if n_thr:
        # loser's thresholds, broadcast to (n_samples, n_trials, K)
        loser_thr = np.where(
            (losers == 0)[..., None], thr[None, :, 0, :], thr[None, :, 1, :]
        )
        conf = np.sum(
            loser_thr * loser_times[..., None] > decision_times[..., None], axis=-1
        )
    else:
        conf = np.zeros_like(winners)

    omitted = decision_times > max_t
    if deadline is not None:
        deadline_arr = _broadcast_param(deadline, n_trials, "deadline")
        omitted |= rts >= deadline_arr[None, :]

    rts_out = np.where(omitted, OMISSION_SENTINEL, rts).astype(np.float32)
    # Censoring touches rts only: the latent (winner, confidence) label stays.
    choices_out = (winners + 2 * conf).astype(np.int64)

    metadata: dict[str, Any] = {
        "simulator": "multiple_threshold_lognormal_race",
        "possible_choices": list(range(2 * n_levels)),
        "n_confidence_levels": n_levels,
        "choice_coding": "winner + 2 * confidence (0 = lowest)",
        "n_samples": n_samples,
        "n_trials": n_trials,
        "n_accumulators": 2,
        "max_t": max_t,
        "t": t_arr.astype(np.float32),
        "rho": rho_arr.astype(np.float32),
        "deadline": None if deadline is None else np.asarray(deadline),
    }
    for i in range(2):
        metadata[f"mu_{i}"] = mus[:, i].astype(np.float32)
        metadata[f"sigma_{i}"] = sigmas[:, i].astype(np.float32)
        for k in range(n_thr):
            metadata[f"d{k + 1}acc{i}"] = thr[:, i, k].astype(np.float32)

    return {
        "rts": rts_out[:, :, None],
        "choices": choices_out[:, :, None],
        "metadata": metadata,
    }
