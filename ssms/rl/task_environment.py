"""TaskEnvironment protocol and built-in implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass


@runtime_checkable
class TaskEnvironment(Protocol):
    """Protocol for RLSSM task environments.

    A task environment generates rewards and optional per-trial context data.
    It is stateful (has an RNG) and must be reset before each participant.
    """

    @property
    def n_choices(self) -> int:
        """Number of available actions (e.g., 2 for two-armed bandit)."""
        ...

    @property
    def choices(self) -> list[int]:
        """Valid action values in task space (e.g., [0, 1])."""
        ...

    @property
    def extra_fields(self) -> list[str]:
        """Names of additional per-trial data columns this environment provides
        (beyond rt, response, feedback). E.g., ['stimulus_id', 'set_size']."""
        ...

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset environment state for a new participant.
        ``rng`` is the NumPy Generator for reproducible reward generation."""
        ...

    def generate_reward(self, action: int, trial_idx: int) -> float:
        """Generate reward for the given action on the given trial.
        Returns reward value (e.g., 0.0 or 1.0 for Bernoulli)."""
        ...

    def get_extra_data(self, trial_idx: int) -> dict[str, float]:
        """Return additional per-trial data columns.
        Returns empty dict if no extra fields."""
        ...


class TwoArmedBandit:
    """Two-armed Bernoulli bandit task environment.

    Generates binary rewards (0.0 or 1.0) with configurable per-arm probabilities.

    Parameters
    ----------
    reward_probabilities : list[float]
        P(reward=1) for each arm. Length must equal len(choices).
        Default [0.7, 0.3].
    choices : list[int]
        Action values in task space. Default [0, 1].
    """

    def __init__(
        self,
        reward_probabilities: list[float] | None = None,
        choices: list[int] | None = None,
    ):
        self._reward_probabilities = reward_probabilities or [0.7, 0.3]
        self._choices = choices or [0, 1]
        if len(self._reward_probabilities) != len(self._choices):
            raise ValueError(
                f"reward_probabilities length ({len(self._reward_probabilities)}) "
                f"must match choices length ({len(self._choices)})"
            )
        for p in self._reward_probabilities:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Reward probability {p} not in [0, 1]")
        self._rng: np.random.Generator | None = None

    @property
    def n_choices(self) -> int:
        return len(self._choices)

    @property
    def choices(self) -> list[int]:
        return list(self._choices)

    @property
    def extra_fields(self) -> list[str]:
        return []

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def generate_reward(self, action: int, trial_idx: int) -> float:
        if self._rng is None:
            raise RuntimeError("Call reset() before generate_reward()")
        idx = self._choices.index(action)
        return float(self._rng.random() < self._reward_probabilities[idx])

    def get_extra_data(self, trial_idx: int) -> dict[str, float]:
        return {}


@dataclass
class TaskConfig:
    """Convenience configuration for common task paradigms.

    Use ``build_environment()`` to construct the appropriate ``TaskEnvironment``.
    Accepted by ``RLSSMModelConfig`` in place of a ``TaskEnvironment`` instance —
    auto-converted in ``__post_init__``.

    Parameters
    ----------
    n_arms : int
        Number of arms/choices. Default 2.
    reward_type : str
        "bernoulli" (v1) or "gaussian" (v2).
    reward_probs : list[float] | None
        Per-arm reward probabilities for Bernoulli tasks.
    choices : list[int] | None
        Action values. Default: [0, 1, ..., n_arms-1].
    """

    n_arms: int = 2
    reward_type: str = "bernoulli"
    reward_probs: list[float] | None = None
    choices: list[int] | None = None

    def build_environment(self) -> TaskEnvironment:
        choices = self.choices or list(range(self.n_arms))
        if self.reward_type == "bernoulli":
            probs = self.reward_probs or [0.7, 0.3]
            return TwoArmedBandit(reward_probabilities=probs, choices=choices)
        else:
            raise ValueError(
                f"Unsupported reward_type '{self.reward_type}'. "
                f"v1 supports: 'bernoulli'."
            )
