"""Registry for RLSSM presets."""

from __future__ import annotations

from typing import Callable

from .config import ModelConfig

_PRESETS: dict[str, Callable[[], ModelConfig]] = {}


def register(name: str, factory: Callable[[], ModelConfig]) -> None:
    """Register a named RLSSM preset."""
    _PRESETS[name] = factory


def get(name: str) -> ModelConfig:
    """Get a named RLSSM preset config. Returns a fresh instance."""
    if name not in _PRESETS:
        available = sorted(_PRESETS.keys())
        raise KeyError(f"Unknown RLSSM preset '{name}'. Available: {available}")
    return _PRESETS[name]()


def list() -> list[str]:
    """List available RLSSM preset names."""
    return sorted(_PRESETS.keys())


# --- v1 Built-in Presets ---


def _make_rlssm1() -> ModelConfig:
    from .env import TwoArmedBandit
    from .learning import RescorlaWagnerDeltaRule

    return ModelConfig(
        model_name="rlssm1",
        description=(
            "RLSSM: Rescorla-Wagner delta rule + angle SSM + "
            "two-armed Bernoulli bandit. Matches HSSM's rlssm1 preset."
        ),
        decision_process="angle",
        learning_process=RescorlaWagnerDeltaRule(n_choices=2, initial_q=0.5),
        task_environment=TwoArmedBandit(
            reward_probabilities=[0.7, 0.3], choices=[-1, 1]
        ),
    )


register("rlssm1", _make_rlssm1)
