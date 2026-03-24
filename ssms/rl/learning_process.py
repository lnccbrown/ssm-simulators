"""LearningProcess protocol and built-in implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LearningProcess(Protocol):
    """Protocol for RLSSM learning processes.

    A learning process maintains internal state (e.g., Q-values) and computes
    SSM parameters (e.g., drift rate) from that state on each trial. After each
    trial's decision and reward, the state is updated.

    The ``computed_params`` property is the formal HANDSHAKE between the learning
    process and the decision process — it declares which SSM parameters the
    learning process produces. The simulator validates that these, together with
    fixed SSM params provided by the user, cover all parameters required by the
    decision process model.
    """

    @property
    def computed_params(self) -> list[str]:
        """SSM parameter names this process computes (e.g., ['v']).

        This is the handshake — declares what SSM parameters are
        informed by the learning process."""
        ...

    @property
    def free_params(self) -> list[str]:
        """RL parameter names this process requires from theta
        (e.g., ['rl_alpha', 'scaler'])."""
        ...

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Bounds for each free param. Used by config validation
        and to_hssm_config_dict()."""
        ...

    @property
    def default_params(self) -> dict[str, float]:
        """Default values for each free param."""
        ...

    def reset(self, **kwargs) -> None:
        """Reset internal state for a new participant.
        Called at the start of each participant's trial sequence."""
        ...

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        """Compute SSM parameters from current learning state.
        Called BEFORE the SSM runs on each trial.
        ``trial_params`` contains the RL free params for this trial.
        Returns e.g. {'v': 0.35}."""
        ...

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Update learning state given the choice outcome.
        Called AFTER the SSM runs and reward is generated.
        ``action`` is in task-action space (e.g., 0 or 1)."""
        ...


class RescorlaWagnerDeltaRule:
    """Rescorla-Wagner delta learning rule for 2-armed bandit tasks.

    Computes drift rate as scaled Q-value difference: v = (Q[1] - Q[0]) * scaler.
    Updates Q-values via: Q[action] += alpha * (reward - Q[action]).

    Numerically equivalent to HSSM's compute_v_trial_wise() in
    src/hssm/rl/likelihoods/two_armed_bandit.py.

    Parameters
    ----------
    n_choices : int
        Number of choice alternatives. Default 2.
    initial_q : float
        Initial Q-value for all alternatives. Default 0.5.
        Matches HSSM's ``jnp.ones(2) * 0.5``.
    """

    def __init__(self, n_choices: int = 2, initial_q: float = 0.5):
        self._n_choices = n_choices
        self._initial_q = initial_q
        self._q_values: np.ndarray | None = None

    @property
    def computed_params(self) -> list[str]:
        return ["v"]

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "scaler"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"rl_alpha": (0.0, 1.0), "scaler": (0.001, 10.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2, "scaler": 2.0}

    @property
    def q_values(self) -> np.ndarray | None:
        """Current Q-values. None if reset() has not been called."""
        return self._q_values.copy() if self._q_values is not None else None

    def reset(self, **kwargs) -> None:
        self._q_values = np.full(self._n_choices, self._initial_q, dtype=np.float64)

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        """Drift = (Q[1] - Q[0]) * scaler.

        NOTE: drift is computed BEFORE the Q-value update for this trial,
        matching HSSM's scan order where computed_v precedes delta_RL update.
        """
        scaler = trial_params["scaler"]
        v = float((self._q_values[1] - self._q_values[0]) * scaler)
        return {"v": v}

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Q[action] += alpha * (reward - Q[action])."""
        alpha = trial_params["rl_alpha"]
        delta = reward - self._q_values[action]
        self._q_values[action] += alpha * delta
