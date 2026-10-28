"""Simulation-time parameter transform implementing the MLBA front end.

``ssms`` already owns the MLBA's back end: it is the linear ballistic
accumulator race in ``cssm.lba_vanilla`` (registered as ``lba2``/``lba3``).
What the paper adds is a deterministic map from the attribute values of the
options onto the mean drift rates of that race.  In ``ssms`` terms that is
exactly a *simulation transform*: it consumes registered parameters and emits
the parameters the low-level simulator expects, which is why this lives in the
``parameter_transforms["simulation"]`` pipeline of the model config rather than
in a new Cython simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ssms.transforms.base import ParameterTransform
from ssms.transforms.simulation.mlba_front_end import drift_rates

__all__ = ["MLBAFrontEnd"]


class MLBAFrontEnd(ParameterTransform):
    """Turn MLBA attribute parameters into LBA drift rates.

    Consumes (and removes) ``m``, ``lambda1``, ``lambda2``, ``beta``, ``I0``
    and the per-option attribute values ``P0, Q0, ..., P{n-1}, Q{n-1}``.
    Produces ``v0 ... v{n-1}`` (the mean drift rates ``d_i`` of Equations
    1a-1c) and sets ``nact``.

    It also rewrites ``sd`` -- the per-trial, per-accumulator drift standard
    deviation that ``cssm.lba_vanilla`` expects -- from the registered
    parameter ``s`` (Table 2 of the paper, fixed to 1 there).  ``ssms`` already
    routes a theta entry named ``s`` into its noise slot, but
    ``make_noise_vec`` broadcasts only the first trial's value; rewriting
    ``sd`` here keeps ``s`` genuinely per-trial.

    Parameters
    ----------
    n_options : int
        Number of options in the choice set (number of accumulators).

    Examples
    --------
    >>> transform = MLBAFrontEnd(n_options=2)
    >>> theta = {
    ...     "m": np.array([1.0]), "lambda1": np.array([0.0]),
    ...     "lambda2": np.array([0.0]), "beta": np.array([1.0]),
    ...     "I0": np.array([0.5]), "s": np.array([0.1]),
    ...     "P0": np.array([1.0]), "Q0": np.array([1.0]),
    ...     "P1": np.array([1.0]), "Q1": np.array([1.0]),
    ... }
    >>> theta = transform.apply(theta, {}, n_trials=1)
    >>> float(theta["v0"][0]), float(theta["v1"][0])
    (0.5, 0.5)
    """

    def __init__(self, n_options: int):
        self.n_options = int(n_options)

    @property
    def attribute_params(self) -> list[str]:
        """Names of the attribute parameters this transform consumes."""
        return [f"{a}{i}" for i in range(self.n_options) for a in ("P", "Q")]

    @property
    def front_end_params(self) -> list[str]:
        """Names of the front-end parameters this transform consumes."""
        return ["m", "lambda1", "lambda2", "beta", "I0"]

    def apply(
        self,
        theta: dict[str, Any],
        model_config: dict[str, Any] | None = None,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        """Compute drift rates from attribute values; see class docstring."""
        n = self.n_options
        missing = [
            p for p in self.front_end_params + self.attribute_params if p not in theta
        ]
        if missing:
            raise KeyError(f"MLBAFrontEnd is missing parameters: {missing}")

        P = np.column_stack(
            [np.asarray(theta[f"P{i}"], dtype=np.float64) for i in range(n)]
        )
        Q = np.column_stack(
            [np.asarray(theta[f"Q{i}"], dtype=np.float64) for i in range(n)]
        )

        d = drift_rates(
            P,
            Q,
            m=np.asarray(theta["m"], dtype=np.float64),
            lambda1=np.asarray(theta["lambda1"], dtype=np.float64),
            lambda2=np.asarray(theta["lambda2"], dtype=np.float64),
            I0=np.asarray(theta["I0"], dtype=np.float64),
            beta=np.asarray(theta["beta"], dtype=np.float64),
        )

        for i in range(n):
            theta[f"v{i}"] = d[:, i].astype(np.float32)

        # Per-trial drift standard deviation, shape (n_trials, n_options).
        if "s" in theta:
            s = np.asarray(theta["s"], dtype=np.float32).reshape(-1, 1)
            theta["sd"] = np.tile(s, (1, n)).astype(np.float32)

        theta["nact"] = n

        for key in self.front_end_params + self.attribute_params + ["s"]:
            theta.pop(key, None)

        return theta
