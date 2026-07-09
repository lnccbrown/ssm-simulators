"""Choice-only inverse-temperature softmax simulators."""

from __future__ import annotations

from typing import Any

import numpy as np


PLACEHOLDER_RT = -1.0


def _broadcast_param(value: Any, n_trials: int, name: str) -> np.ndarray:
    """Return ``value`` as a trial-length float array."""
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        return np.full(n_trials, float(arr), dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == n_trials:
        return arr.astype(np.float32)
    raise ValueError(
        f"Parameter {name!r} must be scalar or length n_trials={n_trials}. "
        f"Got shape {arr.shape}."
    )


def inv_temp_softmax(
    *,
    beta,
    q0,
    q1,
    q2=None,
    q3=None,
    n_samples: int = 1000,
    n_trials: int = 1,
    max_t: float = 20.0,
    random_state: int | None = None,
    **kwargs,
) -> dict:
    """Sample choices from softmax(beta * Q) with placeholder RTs."""
    del kwargs

    q_values = [
        _broadcast_param(q0, n_trials, "q0"),
        _broadcast_param(q1, n_trials, "q1"),
    ]
    if q2 is not None:
        q_values.append(_broadcast_param(q2, n_trials, "q2"))
    if q3 is not None:
        q_values.append(_broadcast_param(q3, n_trials, "q3"))

    beta_arr = _broadcast_param(beta, n_trials, "beta")
    q_matrix = np.column_stack(q_values)
    logits = beta_arr[:, None] * q_matrix
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(random_state)
    n_choices = q_matrix.shape[1]
    labels = np.arange(n_choices, dtype=np.int64)
    choices = np.empty((n_samples, n_trials, 1), dtype=np.int64)
    for trial_idx in range(n_trials):
        choices[:, trial_idx, 0] = rng.choice(
            labels,
            size=n_samples,
            p=probs[trial_idx],
        )

    rts = np.full((n_samples, n_trials, 1), PLACEHOLDER_RT, dtype=np.float32)
    return {
        "rts": rts,
        "choices": choices,
        "metadata": {
            "possible_choices": labels.tolist(),
            "n_choices": n_choices,
            "n_samples": n_samples,
            "max_t": max_t,
            "placeholder_rt": PLACEHOLDER_RT,
        },
    }
