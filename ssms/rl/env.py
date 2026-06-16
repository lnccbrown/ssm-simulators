"""TaskEnvironment protocol and built-in implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TaskEnvironment(Protocol):
    """Protocol for RLSSM task environments.

    A task environment provides per-trial context and optional post-decision
    signals. It is stateful and must be reset before each participant.

    Discrete response/choice mapping requires :class:`DiscreteChoiceEnvironment`.
    """

    @property
    def context_fields(self) -> list[str]:
        """Names of per-trial context columns this environment provides."""
        ...

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset environment state for a new participant."""
        ...

    def get_trial_context(self, trial_idx: int) -> dict[str, float]:
        """Return pre-decision per-trial context columns."""
        ...

    def sample_context(self, context: dict, trial_idx: int) -> dict[str, float]:
        """Return post-decision context columns for learning/output."""
        ...


@runtime_checkable
class DiscreteChoiceEnvironment(TaskEnvironment, Protocol):
    """Task environment with discrete SSM response labels and learning choices."""

    @property
    def n_choices(self) -> int:
        """Number of available zero-based learning choices."""
        ...

    @property
    def response_labels(self) -> list[int]:
        """SSM response labels corresponding to choices in order."""
        ...


class _RewardDistribution(Protocol):
    @property
    def n_arms(self) -> int: ...

    def sample(self, choice: int, rng: np.random.Generator) -> float: ...


class _BernoulliRewards:
    def __init__(self, probabilities: list[float]):
        if len(probabilities) < 2:
            raise ValueError("Bandit environments require at least 2 arms")
        for probability in probabilities:
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Reward probability {probability} not in [0, 1]")
        self._probabilities = list(probabilities)

    @property
    def n_arms(self) -> int:
        return len(self._probabilities)

    def sample(self, choice: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self._probabilities[choice])


class _GaussianRewards:
    def __init__(self, means: list[float], sds: list[float]):
        if len(means) < 2:
            raise ValueError("Bandit environments require at least 2 arms")
        if len(means) != len(sds):
            raise ValueError(
                f"means length ({len(means)}) must match sds length ({len(sds)})"
            )
        for sd in sds:
            if sd <= 0.0:
                raise ValueError(f"Reward standard deviation {sd} must be positive")
        self._means = list(means)
        self._sds = list(sds)

    @property
    def n_arms(self) -> int:
        return len(self._means)

    def sample(self, choice: int, rng: np.random.Generator) -> float:
        return float(rng.normal(self._means[choice], self._sds[choice]))


class Bandit:
    """Generic bandit task environment.

    Public constructors are ``Bandit.bernoulli(...)`` and
    ``Bandit.gaussian(...)``. Rewards are sampled by zero-based choice index;
    ``response_labels`` define the SSM labels mapped onto those choices.
    """

    def __init__(
        self,
        rewards: _RewardDistribution,
        response_labels: list[int] | None = None,
    ):
        self._rewards = rewards
        self._response_labels = self._validate_response_labels(
            response_labels, rewards.n_arms
        )
        self._rng: np.random.Generator | None = None

    @classmethod
    def bernoulli(
        cls,
        probabilities: list[float] | None = None,
        response_labels: list[int] | None = None,
    ) -> Bandit:
        """Build a Bernoulli-reward bandit."""
        if probabilities is None:
            probabilities = [0.7, 0.3]
        return cls(
            rewards=_BernoulliRewards(probabilities),
            response_labels=response_labels,
        )

    @classmethod
    def gaussian(
        cls,
        means: list[float] | None = None,
        sds: list[float] | None = None,
        response_labels: list[int] | None = None,
    ) -> Bandit:
        """Build a Gaussian-reward bandit."""
        if means is None:
            means = [1.0, 0.0]
        if sds is None:
            sds = [1.0] * len(means)
        return cls(
            rewards=_GaussianRewards(means, sds), response_labels=response_labels
        )

    @staticmethod
    def _validate_response_labels(
        response_labels: list[int] | None, n_arms: int
    ) -> list[int]:
        if n_arms < 2:
            raise ValueError("Bandit environments require at least 2 arms")
        labels = (
            list(range(n_arms)) if response_labels is None else list(response_labels)
        )
        if len(labels) != n_arms:
            raise ValueError(
                f"response_labels length ({len(labels)}) must match n_arms ({n_arms})"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("response_labels must be unique")
        return labels

    @property
    def n_choices(self) -> int:
        return self._rewards.n_arms

    @property
    def n_arms(self) -> int:
        """Alias for :attr:`n_choices` (bandit terminology)."""
        return self.n_choices

    @property
    def response_labels(self) -> list[int]:
        return list(self._response_labels)

    @property
    def context_fields(self) -> list[str]:
        return ["feedback"]

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def get_trial_context(self, trial_idx: int) -> dict[str, float]:
        return {}

    def sample_context(self, context: dict, trial_idx: int) -> dict[str, float]:
        if self._rng is None:
            raise RuntimeError("Call reset() before sample_context()")
        choice = int(context["choice"])
        if choice < 0 or choice >= self.n_choices:
            raise ValueError(
                f"Choice {choice} is out of range for bandit with {self.n_choices} arms"
            )
        return {"feedback": self._rewards.sample(choice, self._rng)}

    def sample_reward(self, action: int, trial_idx: int) -> float:
        """Compatibility wrapper around ``sample_context``."""
        return self.sample_context({"choice": action}, trial_idx)["feedback"]

    def get_extra_data(self, trial_idx: int) -> dict[str, float]:
        """Compatibility wrapper around ``get_trial_context``."""
        return self.get_trial_context(trial_idx)


TaskEnvironmentBuilder = Callable[[str | None, dict], TaskEnvironment]
_TASK_REGISTRY: dict[str, TaskEnvironmentBuilder] = {}


def register_task(
    task: str, builder: TaskEnvironmentBuilder, *, overwrite: bool = False
) -> None:
    """Register a task environment builder for ``TaskConfig``."""
    if not overwrite and task in _TASK_REGISTRY:
        raise ValueError(
            f"Task {task!r} is already registered. Pass overwrite=True to replace it."
        )
    _TASK_REGISTRY[task] = builder


def registered_tasks() -> list[str]:
    """List task names available through ``TaskConfig``."""
    return sorted(_TASK_REGISTRY)


class TaskConfig:
    """Convenience configuration for registered task environments.

    ``TaskConfig`` is a shorthand that delegates task-specific options to a
    registry builder. Built in support currently includes ``task="bandit"``
    with ``reward="bernoulli"`` or ``reward="gaussian"``.
    """

    def __init__(self, task: str = "bandit", reward: str | None = None, **options):
        self.task = task
        self.reward = reward
        self.options = dict(options)

    def build_environment(self) -> TaskEnvironment:
        if self.task not in _TASK_REGISTRY:
            available = registered_tasks()
            raise ValueError(
                f"Unknown task '{self.task}'. Registered tasks: {available}."
            )
        return _TASK_REGISTRY[self.task](self.reward, dict(self.options))


def _build_bandit(reward: str | None, options: dict) -> TaskEnvironment:
    if reward is None:
        reward = "bernoulli"
    if reward == "bernoulli":
        allowed = {"probabilities", "response_labels"}
        _validate_options("bandit", reward, options, allowed)
        return Bandit.bernoulli(
            probabilities=options.get("probabilities"),
            response_labels=options.get("response_labels"),
        )
    if reward == "gaussian":
        allowed = {"means", "sds", "response_labels"}
        _validate_options("bandit", reward, options, allowed)
        return Bandit.gaussian(
            means=options.get("means"),
            sds=options.get("sds"),
            response_labels=options.get("response_labels"),
        )
    raise ValueError(
        f"Unknown bandit reward '{reward}'. Supported rewards: 'bernoulli', 'gaussian'."
    )


def _validate_options(task: str, reward: str, options: dict, allowed: set[str]) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported options for task='{task}', reward='{reward}': {unknown}"
        )


register_task("bandit", _build_bandit)
