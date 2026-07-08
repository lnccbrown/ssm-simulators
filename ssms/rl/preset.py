"""Registry for RLSSM presets."""

from __future__ import annotations

import builtins
from typing import Any, Callable

from .config import ModelConfig

_PRESETS: dict[str, Callable[[], ModelConfig]] = {}
_PRESET_METADATA: dict[str, dict[str, Any]] = {}


class PresetInfo(dict):
    """Dictionary-like preset metadata with a readable string representation."""

    def __str__(self) -> str:
        default_params = self.get("default_parameters", {})
        default_summary = ", ".join(
            f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in default_params.items()
        )
        lines = [
            f"Preset: {self.get('name')}",
            f"Description: {self.get('description')}",
            f"Task: {self.get('task')}",
            f"Learning process: {self.get('learning_process')}",
            f"Decision process: {self.get('decision_process')}",
            f"Required parameters: {', '.join(self.get('required_parameters', []))}",
            f"Default parameters: {default_summary}",
            f"Response labels: {self.get('response_labels')}",
            f"Response to choice: {self.get('response_to_choice')}",
            f"Context fields: {self.get('context_fields')}",
            f"Learning backend: {self.get('learning_backend')}",
            f"Gradient support: {self.get('gradient')}",
        ]
        hssm = self.get("hssm_compatibility", {})
        if hssm:
            compatible = "yes" if hssm.get("participant_contract") else "no"
            lines.append(f"HSSM participant contract: {compatible}")
        return "\n".join(lines)

    __repr__ = __str__


def register(
    name: str,
    factory: Callable[[], ModelConfig],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a named RLSSM preset."""
    _PRESETS[name] = factory
    if metadata is None:
        _PRESET_METADATA.pop(name, None)
    else:
        _PRESET_METADATA[name] = dict(metadata)


def get(name: str) -> ModelConfig:
    """Get a named RLSSM preset config. Returns a fresh instance."""
    if name not in _PRESETS:
        available = sorted(_PRESETS.keys())
        raise KeyError(f"Unknown RLSSM preset '{name}'. Available: {available}")
    return _PRESETS[name]()


def list() -> builtins.list[str]:
    """List available RLSSM preset names."""
    return sorted(_PRESETS.keys())


def info(name: str) -> PresetInfo:
    """Return readable metadata for a named RLSSM preset."""
    config = get(name)
    config.validate()
    metadata = dict(_PRESET_METADATA.get(name, {}))
    defaults = dict(zip(config.list_params, config.params_default))
    contract = config.participant_contract()
    hssm_metadata = {
        "participant_contract": True,
        "participant_input_fields": builtins.list(contract.input_fields),
        "learning_process_kind": (
            "approx_differentiable"
            if config.resolved_gradient == "available"
            else "blackbox"
        ),
    }
    hssm_metadata.update(metadata.pop("hssm_compatibility", {}))
    return PresetInfo(
        {
            "name": name,
            "model_name": config.model_name,
            "description": config.description,
            "task": metadata.get("task", type(config.task_environment).__name__),
            "learning_process": type(config.learning_process).__name__,
            "decision_process": config.decision_process,
            "required_parameters": builtins.list(config.required_params),
            "default_parameters": defaults,
            "bounds": dict(config.bounds),
            "response_labels": tuple(config.choices),
            "response_to_choice": dict(config.resolved_response_to_choice),
            "context_fields": (
                builtins.list(config.context_fields) if config.context_fields else []
            ),
            "computed_parameters": builtins.list(config._computed_ssm_params),
            "fixed_ssm_parameters": builtins.list(config._fixed_ssm_params),
            "hssm_compatibility": hssm_metadata,
            "learning_backend": config.resolved_learning_backend,
            "gradient": config.resolved_gradient,
            **metadata,
        }
    )


# --- v1 Built-in Presets ---


def _make_two_arm_rw_angle() -> ModelConfig:
    from .env import Bandit
    from .learning import RescorlaWagnerDrift

    return ModelConfig(
        model_name="2AB_RW_Angle",
        description=(
            "Two-armed bandit with a Rescorla-Wagner delta-rule learner "
            "and an angle decision process."
        ),
        decision_process="angle",
        learning_process=RescorlaWagnerDrift(n_actions=2, initial_q=0.5),
        task_environment=Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
    )


def _make_two_arm_rw_inv_temp_softmax() -> ModelConfig:
    from .env import Bandit
    from .learning import RescorlaWagnerSoftmax

    return ModelConfig(
        model_name="2AB_RW_InvTempSoftmax",
        description=(
            "Two-armed bandit with a Rescorla-Wagner delta-rule learner "
            "and a choice-only inverse-temperature softmax decision process."
        ),
        decision_process="inv_temp_softmax_2",
        learning_process=RescorlaWagnerSoftmax(n_actions=2, initial_q=0.5),
        task_environment=Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[0, 1]
        ),
        response=["response"],
    )


def _make_three_arm_rw_inv_temp_softmax() -> ModelConfig:
    from .env import Bandit
    from .learning import RescorlaWagnerSoftmax

    return ModelConfig(
        model_name="3AB_RW_InvTempSoftmax",
        description=(
            "Three-armed bandit with a Rescorla-Wagner delta-rule learner "
            "and a choice-only inverse-temperature softmax decision process."
        ),
        decision_process="inv_temp_softmax_3",
        learning_process=RescorlaWagnerSoftmax(n_actions=3, initial_q=0.5),
        task_environment=Bandit.bernoulli(
            probabilities=[0.7, 0.2, 0.1], response_labels=[0, 1, 2]
        ),
        response=["response"],
    )


register(
    "2AB_RW_Angle",
    _make_two_arm_rw_angle,
    metadata={
        "task": "two-armed Bernoulli bandit",
        "hssm_compatibility": {
            "participant_contract": True,
            "notes": (
                "Exports participant_contract, response-to-choice mapping, "
                "context fields, backend policy, and gradient policy for "
                "HSSM-side RLSSM wiring."
            ),
        },
    },
)

register(
    "2AB_RW_InvTempSoftmax",
    _make_two_arm_rw_inv_temp_softmax,
    metadata={
        "task": "two-armed Bernoulli bandit",
        "hssm_compatibility": {
            "participant_contract": True,
            "notes": (
                "Choice-only RL preset. Uses response-only data, beta as inverse "
                "temperature, q0/q1 as learning-computed SSM parameters, and "
                "rt=-1.0 as a simulator placeholder."
            ),
        },
    },
)

register(
    "3AB_RW_InvTempSoftmax",
    _make_three_arm_rw_inv_temp_softmax,
    metadata={
        "task": "three-armed Bernoulli bandit",
        "hssm_compatibility": {
            "participant_contract": True,
            "notes": (
                "Choice-only RL preset. Uses response-only data, beta as inverse "
                "temperature, q0/q1/q2 as learning-computed SSM parameters, and "
                "rt=-1.0 as a simulator placeholder."
            ),
        },
    },
)
