"""LearningProcess protocol and built-in implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


LearningState = dict[str, Any]


def _validated_choice_index(choice: int, n_choices: int) -> int:
    if not isinstance(choice, (int, np.integer)) or isinstance(choice, bool):
        raise TypeError(f"choice must be an int, got {type(choice).__name__}")
    choice = int(choice)
    if choice < 0 or choice >= n_choices:
        raise ValueError(f"choice {choice} out of range for n_choices={n_choices}")
    return choice


def _import_jax_numpy():
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "The JAX learning backend requires installing ssm-simulators with "
            "the 'jax' extra."
        ) from exc
    return jnp


def _init_q_values(n_actions: int, initial_q: float) -> np.ndarray:
    return np.full(n_actions, initial_q, dtype=np.float64)


def _init_jax_q_values(n_actions: int, initial_q: float):
    jnp = _import_jax_numpy()
    return jnp.full((n_actions,), initial_q)


def _rw_delta_update_python(
    q_values,
    *,
    choice: int,
    feedback: float,
    alpha: float,
    n_actions: int,
) -> np.ndarray:
    choice = _validated_choice_index(choice, n_actions)
    updated = np.asarray(q_values, dtype=np.float64).copy()
    delta = feedback - updated[choice]
    updated[choice] += alpha * delta
    return updated


def _rw_delta_update_jax(q_values, *, choice, feedback, alpha):
    delta = feedback - q_values[choice]
    return q_values.at[choice].add(alpha * delta)


def _validate_n_actions(n_actions: int) -> int:
    if not isinstance(n_actions, int) or isinstance(n_actions, bool):
        raise TypeError(f"n_actions must be an int, got {type(n_actions).__name__}")
    if n_actions < 2:
        raise ValueError("n_actions must be at least 2")
    return n_actions


@runtime_checkable
class LearningProcess(Protocol):
    """Protocol for RLSSM learning processes.

    A learning process maintains internal state (e.g., Q-values) and computes
    SSM parameters (e.g., drift rate) from that state on each trial. After each
    trial's decision and reward, the state is updated.

    The ``computed_params`` property is the formal handshake between the learning
    process and the decision process: it declares which SSM parameters the
    learning process produces. The simulator validates that these, together with
    fixed SSM params provided by the user, cover all parameters required by the
    decision process model.
    """

    @property
    def computed_params(self) -> list[str]:
        """SSM parameter names this process computes (e.g., ['v'])."""
        ...

    @property
    def free_params(self) -> list[str]:
        """RL parameter names this process requires from theta."""
        ...

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Bounds for each free param."""
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

    @property
    def required_context_fields(self) -> list[str]:
        """Context keys this process needs for compute/update."""
        ...

    def init_state(self) -> LearningState:
        """Return an explicit initial learning state for one participant."""
        ...

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Compute SSM parameters from explicit Python/NumPy state."""
        ...

    def update_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> LearningState:
        """Return the next explicit Python/NumPy state."""
        ...

    def reset(self, **kwargs) -> None:
        """Reset internal state for a new participant."""
        ...

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        """Compute SSM parameters from current learning state."""
        ...

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Update learning state given the choice outcome."""
        ...


class RescorlaWagnerDeltaRule:
    """Rescorla-Wagner delta learning core.

    Updates Q-values via ``Q[action] += alpha * (reward - Q[action])``. This
    class owns Q-value state and replay/update behavior but emits no SSM
    parameters by itself. Use ``RescorlaWagnerDrift`` for two-action drift
    models and ``RescorlaWagnerSoftmax`` for inverse-temperature softmax models.
    """

    def __init__(
        self,
        n_actions: int = 2,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
    ):
        self._n_actions = _validate_n_actions(n_actions)
        self._initial_q = float(initial_q)
        self._feedback_field = feedback_field
        self._state: LearningState | None = None

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return []

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"rl_alpha": (0.0, 1.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2}

    @property
    def available_backends(self) -> tuple[str, ...]:
        return ("python", "jax")

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    @property
    def q_values(self) -> np.ndarray | None:
        """Current Q-values. None if reset() has not been called."""
        if self._state is None:
            return None
        return np.asarray(self._state["q_values"], dtype=np.float64).copy()

    def init_state(self) -> LearningState:
        return {"q_values": _init_q_values(self._n_actions, self._initial_q)}

    def init_jax_state(self) -> LearningState:
        return {"q_values": _init_jax_q_values(self._n_actions, self._initial_q)}

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        return {}

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        return {}

    def update_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> LearningState:
        feedback = float(context[self._feedback_field])
        q_values = _rw_delta_update_python(
            state["q_values"],
            choice=context["choice"],
            feedback=feedback,
            alpha=params["rl_alpha"],
            n_actions=self._n_actions,
        )
        return {"q_values": q_values}

    def update_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> LearningState:
        q_values = _rw_delta_update_jax(
            state["q_values"],
            choice=context["choice"],
            feedback=context[self._feedback_field],
            alpha=params["rl_alpha"],
        )
        return {"q_values": q_values}

    def reset(self, **kwargs) -> None:
        self._state = self.init_state()

    def compute_ssm_params(self, trial_params: dict[str, float]) -> dict[str, float]:
        """Compute pre-update SSM parameters from current learning state."""
        if self._state is None:
            raise RuntimeError("Call reset() before compute_ssm_params()")
        return self.compute_python(self._state, trial_params, context={})

    def update(
        self, action: int, reward: float, trial_params: dict[str, float]
    ) -> None:
        """Update Q[action] from the observed outcome."""
        if self._state is None:
            raise RuntimeError("Call reset() before update()")
        self._state = self.update_python(
            self._state,
            trial_params,
            context={"choice": action, self._feedback_field: reward},
        )


class RescorlaWagnerDrift(RescorlaWagnerDeltaRule):
    """Rescorla-Wagner learner emitting two-action drift ``v``.

    Computes drift rate as scaled Q-value difference:
    ``v = (Q[1] - Q[0]) * scaler``.
    """

    def __init__(
        self,
        n_actions: int = 2,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
    ):
        super().__init__(
            n_actions=n_actions,
            initial_q=initial_q,
            feedback_field=feedback_field,
        )
        if self._n_actions != 2:
            raise ValueError(
                "n_actions must be 2; RescorlaWagnerDrift supports "
                "two-action tasks only"
            )

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

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        scaler = params["scaler"]
        q_values = state["q_values"]
        v = float((q_values[1] - q_values[0]) * scaler)
        return {"v": v}

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        scaler = params["scaler"]
        q_values = state["q_values"]
        return {"v": (q_values[1] - q_values[0]) * scaler}


class RescorlaWagnerSoftmax(RescorlaWagnerDeltaRule):
    """Rescorla-Wagner learner emitting pre-update Q-values ``q0..qN``."""

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"rl_alpha": (0.0, 1.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2}

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        q_values = np.asarray(state["q_values"], dtype=np.float64)
        return {name: float(q_values[i]) for i, name in enumerate(self.computed_params)}

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        q_values = state["q_values"]
        return {name: q_values[i] for i, name in enumerate(self.computed_params)}


class RescorlaWagnerRaceDrifts(RescorlaWagnerDeltaRule):
    """Rescorla-Wagner learner emitting scaled race drifts ``v0..vN``.

    The scaling contract is explicit: on each trial, before the RW update,
    ``v_i = scaler * q_i`` for every action ``i``.
    """

    @property
    def computed_params(self) -> list[str]:
        return [f"v{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "scaler"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"rl_alpha": (0.0, 1.0), "scaler": (0.001, 10.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2, "scaler": 2.0}

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        scaler = params["scaler"]
        q_values = np.asarray(state["q_values"], dtype=np.float64)
        return {
            name: float(scaler * q_values[i])
            for i, name in enumerate(self.computed_params)
        }

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        scaler = params["scaler"]
        q_values = state["q_values"]
        return {
            name: scaler * q_values[i] for i, name in enumerate(self.computed_params)
        }


class RescorlaWagnerDualAlphaRule(RescorlaWagnerDeltaRule):
    """Rescorla-Wagner learning core with separate learning rates.

    Positive prediction errors use ``rl_alpha`` and negative prediction errors
    use ``rl_alpha_neg``.
    """

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "rl_alpha_neg"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2, "rl_alpha_neg": 0.2}

    def update_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> LearningState:
        """Update Q[action] with sign-dependent learning rates."""
        choice = _validated_choice_index(context["choice"], self._n_actions)
        feedback = float(context[self._feedback_field])
        q_values = np.asarray(state["q_values"], dtype=np.float64).copy()
        delta = feedback - q_values[choice]
        alpha = params["rl_alpha_neg"] if delta < 0 else params["rl_alpha"]
        q_values[choice] += alpha * delta
        return {"q_values": q_values}

    def update_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> LearningState:
        jnp = _import_jax_numpy()
        choice = context["choice"]
        feedback = context[self._feedback_field]
        q_values = state["q_values"]
        delta = feedback - q_values[choice]
        alpha = jnp.where(
            delta < 0,
            params["rl_alpha_neg"],
            params["rl_alpha"],
        )
        return {"q_values": q_values.at[choice].add(alpha * delta)}


class RescorlaWagnerDualAlphaDrift(RescorlaWagnerDualAlphaRule):
    """Dual-alpha Rescorla-Wagner learner emitting two-action drift ``v``."""

    def __init__(
        self,
        n_actions: int = 2,
        initial_q: float = 0.5,
        feedback_field: str = "feedback",
    ):
        super().__init__(
            n_actions=n_actions,
            initial_q=initial_q,
            feedback_field=feedback_field,
        )
        if self._n_actions != 2:
            raise ValueError(
                "n_actions must be 2; RescorlaWagnerDualAlphaDrift supports "
                "two-action tasks only"
            )

    @property
    def computed_params(self) -> list[str]:
        return ["v"]

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

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        scaler = params["scaler"]
        q_values = state["q_values"]
        v = float((q_values[1] - q_values[0]) * scaler)
        return {"v": v}

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        scaler = params["scaler"]
        q_values = state["q_values"]
        return {"v": (q_values[1] - q_values[0]) * scaler}


class RescorlaWagnerDualAlphaSoftmax(RescorlaWagnerDualAlphaRule):
    """Dual-alpha Rescorla-Wagner learner emitting Q-values ``q0..qN``."""

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["rl_alpha", "rl_alpha_neg"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "rl_alpha": (0.0, 1.0),
            "rl_alpha_neg": (0.0, 1.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {"rl_alpha": 0.2, "rl_alpha_neg": 0.2}

    def compute_python(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        q_values = np.asarray(state["q_values"], dtype=np.float64)
        return {name: float(q_values[i]) for i, name in enumerate(self.computed_params)}

    def compute_jax(
        self,
        state: LearningState,
        params: dict[str, float],
        context: dict[str, Any],
    ):
        q_values = state["q_values"]
        return {name: q_values[i] for i, name in enumerate(self.computed_params)}
