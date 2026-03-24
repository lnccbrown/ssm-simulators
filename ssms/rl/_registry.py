"""Registry for RLSSM presets."""

from __future__ import annotations

from typing import Callable

from .rl_config import RLSSMModelConfig

_RLSSM_PRESETS: dict[str, Callable[[], RLSSMModelConfig]] = {}


def register_rlssm_preset(
    name: str, factory: Callable[[], RLSSMModelConfig]
) -> None:
    """Register a named RLSSM preset."""
    _RLSSM_PRESETS[name] = factory


def get_rlssm_preset(name: str) -> RLSSMModelConfig:
    """Get a named RLSSM preset config. Returns a fresh instance."""
    if name not in _RLSSM_PRESETS:
        available = sorted(_RLSSM_PRESETS.keys())
        raise KeyError(f"Unknown RLSSM preset '{name}'. Available: {available}")
    return _RLSSM_PRESETS[name]()


def list_rlssm_presets() -> list[str]:
    """List available RLSSM preset names."""
    return sorted(_RLSSM_PRESETS.keys())


# --- v1 Built-in Presets ---


def _make_rlssm1() -> RLSSMModelConfig:
    from .learning_process import RescorlaWagnerDeltaRule
    from .task_environment import TwoArmedBandit

    return RLSSMModelConfig(
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


register_rlssm_preset("rlssm1", _make_rlssm1)
