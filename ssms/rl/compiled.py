"""Compiled RLSSM model interfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import preset
from .config import ModelConfig, _jax_available


CompiledFunctionOutput = Literal["array", "dict"]
LearningBackend = Literal["auto", "python", "jax"]


def resolve_model(model: str | ModelConfig) -> ModelConfig:
    """Resolve a preset name or validate an existing RLSSM model config."""
    if isinstance(model, str):
        config = preset.get(model)
        config.validate()
        return config
    if isinstance(model, ModelConfig):
        model.validate()
        return model
    raise TypeError(
        f"model must be a str or ModelConfig. Got {type(model).__name__!r} instead."
    )


@dataclass(frozen=True)
class CompiledModel:
    """Validated executable form of an RLSSM ``ModelConfig``.

    The compiled model exposes package-neutral metadata and pure Python/JAX
    computed-parameter functions that downstream packages can wrap without
    importing HSSM or PyTensor in ``ssm-simulators``.
    """

    config: ModelConfig
    learning_backend: Literal["python", "jax"]
    gradient: Literal["available", "unavailable"]
    model_name: str
    decision_process: str
    list_params: list[str]
    bounds: dict[str, tuple[float, float]]
    params_default: list[float]
    response: list[str]
    choices: tuple[int, ...]
    extra_fields: list[str]
    computed_params: list[str]
    response_to_action: dict[int, int]

    @classmethod
    def from_config(
        cls, config: ModelConfig, backend: LearningBackend = "auto"
    ) -> CompiledModel:
        """Build a compiled model from a structural model config."""
        config.validate()
        resolved_backend = _resolve_backend(config, backend)
        gradient = _resolve_gradient(config, resolved_backend)
        return cls(
            config=config,
            learning_backend=resolved_backend,
            gradient=gradient,
            model_name=config.model_name,
            decision_process=config.decision_process,
            list_params=list(config.list_params),
            bounds=dict(config.bounds),
            params_default=list(config.params_default),
            response=list(config.response),
            choices=tuple(config.choices),
            extra_fields=list(config.extra_fields) if config.extra_fields else [],
            computed_params=list(config._computed_ssm_params),
            response_to_action=dict(config.response_to_action),
        )

    def make_subject_wise_function(
        self,
        input_fields: Sequence[str],
        *,
        action_field: str | None = "action",
        response_field: str | None = None,
        reward_field: str = "feedback",
        output: CompiledFunctionOutput = "array",
    ) -> Callable[[Any], Any]:
        """Generate a subject-wise computed-parameter function.

        The returned function accepts a ``(n_trials, n_fields)`` array whose
        columns match ``input_fields``. It computes SSM parameters before each
        learning update, then updates learning state from either a zero-based
        ``action_field`` or a response label mapped through ``response_field``.
        """
        if output not in {"array", "dict"}:
            raise ValueError("output must be 'array' or 'dict'")

        field_to_idx = {name: idx for idx, name in enumerate(input_fields)}
        if len(field_to_idx) != len(input_fields):
            raise ValueError("input_fields must be unique")

        missing_params = [
            p for p in self.config.learning_process.free_params if p not in field_to_idx
        ]
        if missing_params:
            raise ValueError(
                f"input_fields is missing learning parameters: {missing_params}"
            )
        if reward_field not in field_to_idx:
            raise ValueError(f"input_fields is missing reward_field={reward_field!r}")
        if action_field is None and response_field is None:
            raise ValueError("Either action_field or response_field must be provided")
        resolved_action_field = action_field
        if action_field is not None and action_field not in field_to_idx:
            if response_field is not None:
                resolved_action_field = None
            else:
                raise ValueError(
                    f"input_fields is missing action_field={action_field!r}"
                )
        if resolved_action_field is None and response_field is None:
            raise ValueError(f"input_fields is missing action_field={action_field!r}")
        if response_field is not None and response_field not in field_to_idx:
            raise ValueError(
                f"input_fields is missing response_field={response_field!r}"
            )

        if self.learning_backend == "jax":
            return self._make_jax_subject_wise_function(
                field_to_idx=field_to_idx,
                action_field=resolved_action_field,
                response_field=response_field,
                reward_field=reward_field,
                output=output,
            )
        return self._make_python_subject_wise_function(
            field_to_idx=field_to_idx,
            action_field=resolved_action_field,
            response_field=response_field,
            reward_field=reward_field,
            output=output,
        )

    def _make_python_subject_wise_function(
        self,
        *,
        field_to_idx: dict[str, int],
        action_field: str | None,
        response_field: str | None,
        reward_field: str,
        output: CompiledFunctionOutput,
    ) -> Callable[[Any], Any]:
        lp = self.config.learning_process
        free_params = list(lp.free_params)

        def compute(subject_trials):
            trials = np.asarray(subject_trials)
            state = lp.init_state()
            collected: dict[str, list[float]] = {
                name: [] for name in self.computed_params
            }
            for row in trials:
                trial_params = {
                    name: float(row[field_to_idx[name]]) for name in free_params
                }
                computed = self._map_computed_params(
                    lp.compute_python(state, trial_params)
                )
                for name in self.computed_params:
                    collected[name].append(float(computed[name]))
                action = self._extract_python_action(
                    row=row,
                    field_to_idx=field_to_idx,
                    action_field=action_field,
                    response_field=response_field,
                )
                reward = float(row[field_to_idx[reward_field]])
                state = lp.update_python(state, action, reward, trial_params)
            return self._format_python_output(collected, output)

        return compute

    def _make_jax_subject_wise_function(
        self,
        *,
        field_to_idx: dict[str, int],
        action_field: str | None,
        response_field: str | None,
        reward_field: str,
        output: CompiledFunctionOutput,
    ) -> Callable[[Any], Any]:
        import jax.numpy as jnp
        from jax.lax import scan

        lp = self.config.learning_process
        free_params = list(lp.free_params)
        response_labels = jnp.asarray(list(self.response_to_action.keys()))
        response_actions = jnp.asarray(list(self.response_to_action.values()))

        def compute(subject_trials):
            def step(state, row):
                trial_params = {name: row[field_to_idx[name]] for name in free_params}
                computed = self._map_computed_params(
                    lp.compute_jax(state, trial_params)
                )
                action = self._extract_jax_action(
                    row=row,
                    field_to_idx=field_to_idx,
                    action_field=action_field,
                    response_field=response_field,
                    response_labels=response_labels,
                    response_actions=response_actions,
                )
                reward = row[field_to_idx[reward_field]]
                state = lp.update_jax(state, action, reward, trial_params)
                return state, {name: computed[name] for name in self.computed_params}

            _, values = scan(step, lp.init_jax_state(), subject_trials)
            return self._format_jax_output(values, output)

        return compute

    def _map_computed_params(self, computed_raw: dict[str, Any]) -> dict[str, Any]:
        mapping = self.config.computed_param_mapping or {}
        mapped = {}
        for output_name, value in computed_raw.items():
            mapped[mapping.get(output_name, output_name)] = value
        missing = [name for name in self.computed_params if name not in mapped]
        if missing:
            raise ValueError(f"Learning process did not compute params: {missing}")
        return mapped

    def _extract_python_action(
        self,
        *,
        row,
        field_to_idx: dict[str, int],
        action_field: str | None,
        response_field: str | None,
    ) -> int:
        if action_field is not None:
            return int(row[field_to_idx[action_field]])
        response = int(row[field_to_idx[response_field]])
        return int(self.response_to_action[response])

    def _extract_jax_action(
        self,
        *,
        row,
        field_to_idx: dict[str, int],
        action_field: str | None,
        response_field: str | None,
        response_labels,
        response_actions,
    ):
        import jax.numpy as jnp

        if action_field is not None:
            return jnp.asarray(row[field_to_idx[action_field]], dtype=jnp.int32)
        response = row[field_to_idx[response_field]]
        matches = response_labels == response.astype(response_labels.dtype)
        return jnp.asarray(
            jnp.sum(jnp.where(matches, response_actions, 0)), dtype=jnp.int32
        )

    def _format_python_output(
        self, collected: dict[str, list[float]], output: CompiledFunctionOutput
    ):
        arrays = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in collected.items()
        }
        if output == "dict":
            return arrays
        if len(self.computed_params) == 1:
            return arrays[self.computed_params[0]]
        return np.column_stack([arrays[name] for name in self.computed_params])

    def _format_jax_output(
        self, values: dict[str, Any], output: CompiledFunctionOutput
    ):
        import jax.numpy as jnp

        if output == "dict":
            return values
        if len(self.computed_params) == 1:
            return values[self.computed_params[0]]
        return jnp.column_stack([values[name] for name in self.computed_params])


def _resolve_backend(
    config: ModelConfig, backend: LearningBackend
) -> Literal["python", "jax"]:
    if backend not in {"auto", "python", "jax"}:
        raise ValueError("backend must be one of 'auto', 'python', or 'jax'")
    if backend == "auto":
        return config.resolved_learning_backend

    available = config._available_learning_backends()
    if backend not in available:
        raise ValueError(
            f"Learning process does not implement the {backend!r} backend. "
            f"Available backends: {available}."
        )
    if backend == "jax" and not _jax_available():
        raise ValueError(
            "JAX backend requested, but JAX is not installed. Install "
            "ssm-simulators with the 'jax' extra or use backend='python'."
        )
    if backend == "python":
        _require_methods(
            config.learning_process, ("init_state", "compute_python", "update_python")
        )
    else:
        _require_methods(
            config.learning_process,
            ("init_jax_state", "compute_jax", "update_jax"),
        )
    return backend


def _resolve_gradient(
    config: ModelConfig, backend: Literal["python", "jax"]
) -> Literal["available", "unavailable"]:
    supports_gradient = bool(
        getattr(config.learning_process, "supports_gradient", False)
    )
    if config.gradient == "available" and (backend != "jax" or not supports_gradient):
        raise ValueError(
            "gradient='available' requires a JAX learning backend with declared "
            "gradient support."
        )
    if backend == "jax" and supports_gradient:
        return "available"
    return "unavailable"


def _require_methods(obj: Any, method_names: tuple[str, ...]) -> None:
    missing = [name for name in method_names if not callable(getattr(obj, name, None))]
    if missing:
        raise ValueError(f"Learning process is missing required methods: {missing}")
