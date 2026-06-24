"""Assembled RLSSM model interfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from . import preset
from .config import (
    DEFAULT_RESPONSE_FIELD,
    ModelConfig,
    _jax_available,
    derive_participant_contract,
)


class LearningBackendRequest(StrEnum):
    """Requested learning backend policy for assembly."""

    AUTO = "auto"
    PYTHON = "python"
    JAX = "jax"


class ResolvedLearningBackend(StrEnum):
    """Concrete learning backend selected after resolving policy."""

    PYTHON = "python"
    JAX = "jax"


class AssembledFunctionOutput(StrEnum):
    """Return shape for participant-wise computed-parameter functions."""

    ARRAY = "array"
    DICT = "dict"


def _is_jax_tracer(value: Any) -> bool:
    """Return True when ``value`` is a JAX tracer (a symbolic trace input).

    Used to skip eager, concrete-only validation when an assembled JAX function
    is being traced for gradients/jit (e.g. during HSSM inference), where the
    inputs are symbolic and cannot be converted to NumPy.
    """
    if not _jax_available():
        return False
    import jax

    return isinstance(value, jax.core.Tracer)


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


def _coerce_assembled_output(
    output: AssembledFunctionOutput | str,
) -> AssembledFunctionOutput:
    """Normalize participant-function output mode to ``AssembledFunctionOutput``."""
    if isinstance(output, AssembledFunctionOutput):
        return output
    try:
        return AssembledFunctionOutput(output)
    except ValueError as exc:
        raise ValueError(
            f"output must be one of {[member.value for member in AssembledFunctionOutput]}. "
            f"Got {output!r}."
        ) from exc


@dataclass(frozen=True)
class AssembledModel:
    """Validated executable form of an RLSSM ``ModelConfig``.

    The assembled model exposes package-neutral metadata and pure Python/JAX
    computed-parameter functions that downstream packages can wrap without
    importing HSSM or PyTensor in ``ssm-simulators``.
    """

    config: ModelConfig
    learning_backend: ResolvedLearningBackend
    gradient: Literal["available", "unavailable"]
    model_name: str
    decision_process: str
    list_params: list[str]
    bounds: dict[str, tuple[float, float]]
    params_default: list[float]
    response: list[str]
    choices: tuple[int, ...]
    context_fields: list[str]
    computed_params: list[str]
    response_to_choice: dict[int, int]

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        backend: LearningBackendRequest | str = LearningBackendRequest.AUTO,
    ) -> AssembledModel:
        """Build an assembled model from a structural model config."""
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
            context_fields=list(config.context_fields) if config.context_fields else [],
            computed_params=list(config._computed_ssm_params),
            response_to_choice=dict(config.response_to_choice),
        )

    def get_participant_input_fields(
        self,
        *,
        response_field: str = DEFAULT_RESPONSE_FIELD,
    ) -> list[str]:
        """Return the default participant input columns derived from the config."""
        contract = derive_participant_contract(
            self.config,
            response_field=response_field,
        )
        return list(contract.input_fields)

    def participant_input_fields(
        self,
        *,
        response_field: str = DEFAULT_RESPONSE_FIELD,
    ) -> list[str]:
        """Backward-compatible alias for :meth:`get_participant_input_fields`."""
        return self.get_participant_input_fields(response_field=response_field)

    def assemble_participant_fn(
        self,
        input_fields: Sequence[str] | None = None,
        *,
        response_field: str = DEFAULT_RESPONSE_FIELD,
        output: AssembledFunctionOutput | str = AssembledFunctionOutput.ARRAY,
    ) -> Callable[[Any], Any]:
        """Assemble a participant-wise computed-parameter function.

        By default, ``input_fields`` are derived from the model config. Pass
        explicit values only for non-standard layouts.

        The returned function accepts a ``(n_trials, n_fields)`` array whose
        columns match ``input_fields``. It computes SSM parameters before each
        learning update, maps response labels to zero-based action indices, and
        updates learning state from the response and optional outcome.
        """
        output_mode = _coerce_assembled_output(output)

        if input_fields is None:
            input_fields = self.get_participant_input_fields(
                response_field=response_field,
            )
        else:
            input_fields = list(input_fields)

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
        missing_context = [
            field_name
            for field_name in self.context_fields
            if field_name not in field_to_idx
        ]
        if missing_context:
            raise ValueError(
                f"input_fields is missing context fields: {missing_context}"
            )
        if response_field not in field_to_idx:
            raise ValueError(
                f"input_fields is missing response_field={response_field!r}"
            )

        if self.learning_backend is ResolvedLearningBackend.JAX:
            return self._make_jax_subject_wise_function(
                field_to_idx=field_to_idx,
                response_field=response_field,
                output=output_mode,
            )
        return self._make_python_subject_wise_function(
            field_to_idx=field_to_idx,
            response_field=response_field,
            output=output_mode,
        )

    def _make_python_subject_wise_function(
        self,
        *,
        field_to_idx: dict[str, int],
        response_field: str,
        output: AssembledFunctionOutput,
    ) -> Callable[[Any], Any]:
        """Build a NumPy loop that replays learning and collects computed params."""
        lp = self.config.learning_process
        free_params = list(lp.free_params)
        context_fields = list(self.context_fields)

        def compute(subject_trials):
            self._validate_trial_response_labels(
                subject_trials, field_to_idx, response_field
            )
            trials = np.asarray(subject_trials)
            state = lp.init_state()
            collected: dict[str, list[float]] = {
                name: [] for name in self.computed_params
            }
            for row in trials:
                trial_params = {
                    name: float(row[field_to_idx[name]]) for name in free_params
                }
                context = {
                    name: float(row[field_to_idx[name]]) for name in context_fields
                }
                computed = self._map_computed_params(
                    lp.compute_python(state, trial_params, context)
                )
                for name in self.computed_params:
                    collected[name].append(float(computed[name]))
                choice = self._extract_choice(
                    row=row,
                    field_to_idx=field_to_idx,
                    response_field=response_field,
                    backend=ResolvedLearningBackend.PYTHON,
                )
                context.update(
                    {
                        "response": int(row[field_to_idx[response_field]]),
                        "choice": choice,
                    }
                )
                state = lp.update_python(state, trial_params, context)
            return self._format_python_output(collected, output)

        return compute

    def _make_jax_subject_wise_function(
        self,
        *,
        field_to_idx: dict[str, int],
        response_field: str,
        output: AssembledFunctionOutput,
    ) -> Callable[[Any], Any]:
        """Build a ``jax.lax.scan`` replay of learning and computed params."""
        import jax.numpy as jnp
        from jax.lax import scan

        lp = self.config.learning_process
        free_params = list(lp.free_params)
        context_fields = list(self.context_fields)
        response_labels = jnp.asarray(list(self.response_to_choice.keys()))
        response_choices = jnp.asarray(list(self.response_to_choice.values()))

        def compute(subject_trials):
            self._validate_trial_response_labels(
                subject_trials, field_to_idx, response_field
            )

            def step(state, row):
                trial_params = {name: row[field_to_idx[name]] for name in free_params}
                context = {name: row[field_to_idx[name]] for name in context_fields}
                computed = self._map_computed_params(
                    lp.compute_jax(state, trial_params, context)
                )
                choice = self._extract_choice(
                    row=row,
                    field_to_idx=field_to_idx,
                    response_field=response_field,
                    backend=ResolvedLearningBackend.JAX,
                    response_labels=response_labels,
                    response_choices=response_choices,
                )
                context.update(
                    {
                        "response": row[field_to_idx[response_field]],
                        "choice": choice,
                    }
                )
                state = lp.update_jax(state, trial_params, context)
                return state, {name: computed[name] for name in self.computed_params}

            _, values = scan(step, lp.init_jax_state(), subject_trials)
            return self._format_jax_output(values, output)

        return compute

    def _validate_trial_response_labels(
        self,
        subject_trials,
        field_to_idx: dict[str, int],
        response_field: str,
    ) -> None:
        """Raise when trial responses are absent from ``response_to_choice``.

        Response labels can only be read from a concrete array. During a JAX
        trace (gradient/jit for inference) ``subject_trials`` is a symbolic
        tracer, so this eager check is skipped; the panel is validated at the
        data boundary (``ModelConfig.validate_data`` and the concrete Python /
        eager paths) instead.
        """
        if _is_jax_tracer(subject_trials):
            return
        trials = np.asarray(subject_trials)
        if trials.ndim == 1:
            trials = trials.reshape(1, -1)
        responses = trials[:, field_to_idx[response_field]]
        mapping_keys = set(self.response_to_choice.keys())
        unmapped = sorted(
            {
                int(value)
                for value in np.unique(responses)
                if int(value) not in mapping_keys
            }
        )
        if unmapped:
            raise ValueError(
                f"Trial responses {unmapped} are not in response_to_choice. "
                f"Expected one of: {sorted(mapping_keys)}."
            )

    def _map_computed_params(self, computed_raw: dict[str, Any]) -> dict[str, Any]:
        """Apply ``computed_param_mapping`` and verify all outputs are present."""
        mapping = self.config.computed_param_mapping or {}
        mapped = {
            mapping.get(output_name, output_name): value
            for output_name, value in computed_raw.items()
        }
        missing = [name for name in self.computed_params if name not in mapped]
        if missing:
            raise ValueError(f"Learning process did not compute params: {missing}")
        return mapped

    def _extract_choice(
        self,
        *,
        row,
        field_to_idx: dict[str, int],
        response_field: str,
        backend: ResolvedLearningBackend,
        response_labels=None,
        response_choices=None,
    ):
        """Map a trial response label to a zero-based learning choice index."""
        if backend is ResolvedLearningBackend.PYTHON:
            response = int(row[field_to_idx[response_field]])
            return int(self.response_to_choice[response])

        import jax.numpy as jnp

        response = row[field_to_idx[response_field]]
        matches = response_labels == response.astype(response_labels.dtype)
        return jnp.asarray(
            jnp.sum(jnp.where(matches, response_choices, 0)), dtype=jnp.int32
        )

    def _format_python_output(
        self, collected: dict[str, list[float]], output: AssembledFunctionOutput
    ):
        """Format collected trial-wise values as an array or parameter dict."""
        arrays = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in collected.items()
        }
        if output is AssembledFunctionOutput.DICT:
            return arrays
        if len(self.computed_params) == 1:
            return arrays[self.computed_params[0]]
        return np.column_stack([arrays[name] for name in self.computed_params])

    def _format_jax_output(
        self, values: dict[str, Any], output: AssembledFunctionOutput
    ):
        """Format scan outputs as an array or parameter dict."""
        import jax.numpy as jnp

        if output is AssembledFunctionOutput.DICT:
            return values
        if len(self.computed_params) == 1:
            return values[self.computed_params[0]]
        return jnp.column_stack([values[name] for name in self.computed_params])


def _resolve_backend(
    config: ModelConfig,
    backend: LearningBackendRequest | str,
) -> ResolvedLearningBackend:
    """Resolve requested backend policy to a concrete python or jax backend."""
    if isinstance(backend, str):
        try:
            backend = LearningBackendRequest(backend)
        except ValueError as exc:
            raise ValueError(
                f"backend must be one of {[member.value for member in LearningBackendRequest]}. "
                f"Got {backend!r}."
            ) from exc

    if backend is LearningBackendRequest.AUTO:
        resolved = config.resolved_learning_backend
        return ResolvedLearningBackend(resolved)

    available = config._available_learning_backends()
    backend_value = backend.value
    if backend_value not in available:
        raise ValueError(
            f"Learning process does not implement the {backend_value!r} backend. "
            f"Available backends: {available}."
        )
    if backend is LearningBackendRequest.JAX and not _jax_available():
        raise ValueError(
            "JAX backend requested, but JAX is not installed. Install "
            "ssm-simulators with the 'jax' extra or use backend='python'."
        )
    if backend is LearningBackendRequest.PYTHON:
        _require_methods(
            config.learning_process, ("init_state", "compute_python", "update_python")
        )
    else:
        _require_methods(
            config.learning_process,
            ("init_jax_state", "compute_jax", "update_jax"),
        )
    return ResolvedLearningBackend(backend_value)


def _resolve_gradient(
    config: ModelConfig, backend: ResolvedLearningBackend
) -> Literal["available", "unavailable"]:
    """Resolve whether gradient inference is available for this assembly."""
    supports_gradient = bool(
        getattr(config.learning_process, "supports_gradient", False)
    )
    if config.gradient == "available" and (
        backend is not ResolvedLearningBackend.JAX or not supports_gradient
    ):
        raise ValueError(
            "gradient='available' requires a JAX learning backend with declared "
            "gradient support."
        )
    if backend is ResolvedLearningBackend.JAX and supports_gradient:
        return "available"
    return "unavailable"


def _require_methods(obj: Any, method_names: tuple[str, ...]) -> None:
    """Raise when a learning process is missing required backend methods."""
    missing = [name for name in method_names if not callable(getattr(obj, name, None))]
    if missing:
        raise ValueError(f"Learning process is missing required methods: {missing}")
