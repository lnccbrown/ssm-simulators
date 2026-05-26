"""LearningProcess protocol and built-in implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


LearningState = dict[str, Any]


def _import_jax_numpy():
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "The JAX learning backend requires installing ssm-simulators with "
            "the 'jax' extra."
        ) from exc
    return jnp


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

    @property
    def available_backends(self) -> tuple[str, ...]:
        """Learning backends implemented by this process."""
        ...

    @property
    def supports_gradient(self) -> bool:
        """Whether the differentiable backend supports gradient-based inference."""
        ...

    def init_state(self) -> LearningState:
        """Return an explicit initial learning state for one participant."""
        ...

    def compute_python(
        self, state: LearningState, trial_params: dict[str, float]
    ) -> dict[str, float]:
        """Compute SSM parameters from explicit Python/NumPy state."""
        ...

    def update_python(
        self,
        state: LearningState,
        action: int,
        reward: float,
        trial_params: dict[str, float],
    ) -> LearningState:
        """Return the next explicit Python/NumPy state."""
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
        ``action`` is the zero-based learning action index."""
        ...


class RescorlaWagnerDeltaRule:
    """Rescorla-Wagner delta learning rule for 2-armed bandit tasks.

    Computes drift rate as scaled Q-value difference: v = (Q[1] - Q[0]) * scaler.
    Updates Q-values via: Q[action] += alpha * (reward - Q[action]).

    Numerically equivalent to HSSM's compute_v_trial_wise() in
    src/hssm/rl/likelihoods/two_armed_bandit.py.

    Parameters
    ----------
    n_actions : int
        Number of choice alternatives. Default 2.
    initial_q : float
        Initial Q-value for all alternatives. Default 0.5.
        Matches HSSM's ``jnp.ones(2) * 0.5``.
    """

    def __init__(self, n_actions: int = 2, initial_q: float = 0.5):
        if n_actions != 2:
            raise ValueError(
                "n_actions must be 2; RescorlaWagnerDeltaRule supports "
                "two-action tasks only"
            )
        self._n_actions = n_actions
        self._initial_q = initial_q
        self._state: LearningState | None = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

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
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def q_values(self) -> np.ndarray | None:
        """Current Q-values. None if reset() has not been called."""
        if self._state is None:
            return None
        return np.asarray(self._state["q_values"], dtype=np.float64).copy()

    def init_state(self) -> LearningState:
        return {"q_values": np.full(self._n_actions, self._initial_q, dtype=np.float64)}

    def init_jax_state(self) -> LearningState:
        jnp = _import_jax_numpy()
        return {"q_values": jnp.full((self._n_actions,), self._initial_q)}

    def compute_python(
        self, state: LearningState, trial_params: dict[str, float]
    ) -> dict[str, float]:
        scaler = trial_params["scaler"]
        q_values = state["q_values"]
        v = float((q_values[1] - q_values[0]) * scaler)
        return {"v": v}

    def compute_jax(self, state: LearningState, trial_params: dict[str, float]):
        scaler = trial_params["scaler"]
        q_values = state["q_values"]
        return {"v": (q_values[1] - q_values[0]) * scaler}

    def update_python(
        self,
        state: LearningState,
        action: int,
        reward: float,
        trial_params: dict[str, float],
    ) -> LearningState:
        alpha = trial_params["rl_alpha"]
        q_values = np.asarray(state["q_values"], dtype=np.float64).copy()
        delta = reward - q_values[action]
        q_values[action] += alpha * delta
        return {"q_values": q_values}

    def update_jax(
        self,
        state: LearningState,
        action: int,
        reward: float,
        trial_params: dict[str, float],
    ) -> LearningState:
        alpha = trial_params["rl_alpha"]
        q_values = state["q_values"]
        delta = reward - q_values[action]
        return {"q_values": q_values.at[action].add(alpha * delta)}

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        """Drift = (Q[1] - Q[0]) * scaler.

        NOTE: drift is computed BEFORE the Q-value update for this trial,
        matching HSSM's scan order where computed_v precedes delta_RL update.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params)

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Q[action] += alpha * (reward - Q[action])."""
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(self._state, action, reward, trial_params)


class RescorlaWagnerDualAlphaRule(RescorlaWagnerDeltaRule):
    """Rescorla-Wagner delta learning rule with separate learning rates.

    Positive prediction errors use ``rl_alpha`` and negative prediction errors
    use ``rl_alpha_neg``. Drift computation and Q-value initialization match
    ``RescorlaWagnerDeltaRule``.
    """

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "rl_alpha_neg", "scaler"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
            "scaler": (0.001, 10.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2, "rl_alpha_neg": 0.2, "scaler": 2.0}

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Update Q[action] with sign-dependent learning rates."""
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(self._state, action, reward, trial_params)

    def update_python(
        self,
        state: LearningState,
        action: int,
        reward: float,
        trial_params: dict[str, float],
    ) -> LearningState:
        """Update Q[action] with sign-dependent learning rates."""
        q_values = np.asarray(state["q_values"], dtype=np.float64).copy()
        delta = reward - q_values[action]
        alpha = trial_params["rl_alpha_neg"] if delta < 0 else trial_params["rl_alpha"]
        q_values[action] += alpha * delta
        return {"q_values": q_values}

    def update_jax(
        self,
        state: LearningState,
        action: int,
        reward: float,
        trial_params: dict[str, float],
    ) -> LearningState:
        jnp = _import_jax_numpy()
        q_values = state["q_values"]
        delta = reward - q_values[action]
        alpha = jnp.where(
            delta < 0,
            trial_params["rl_alpha_neg"],
            trial_params["rl_alpha"],
        )
        return {"q_values": q_values.at[action].add(alpha * delta)}
