"""ModelConfig — structural model specification for RLSSM simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Literal, cast

from ssms.config.model_config_builder import ModelConfigBuilder

from .env import TaskConfig, TaskEnvironment
from .learning import LearningProcess


# Fields that to_hssm_config_dict() must emit — kept as a constant
# so we can write contract tests against it.
_HSSM_SHARED_FIELDS = (
    "model_name",
    "description",
    "list_params",
    "bounds",
    "params_default",
    "choices",
    "decision_process",
    "response",
    "response_mapping",
    "learning_backend",
    "gradient",
    "extra_fields",
)


def _jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


@dataclass
class ModelConfig:
    """RLSSM model configuration for ssm-simulators.

    Describes the *structural specification* of an RLSSM model:
    which learning process, which decision process (SSM), and which task
    environment. Concrete parameter values are NOT stored here — they
    are passed as ``theta`` to ``Simulator.simulate()``.

    Parameters
    ----------
    model_name : str
        Unique identifier for this RLSSM model (e.g., "rlssm_angle_rw").
    description : str
        Human-readable model description.
    decision_process : str
        SSM model name in ssm-simulators registry (e.g., "angle", "ddm").
        Must be resolvable via ``ModelConfigBuilder.from_model()``.
    learning_process : LearningProcess
        Instance of a class satisfying the ``LearningProcess`` protocol.
    task_environment : TaskEnvironment | TaskConfig
        Task environment instance or a ``TaskConfig`` to auto-build one.
        If ``TaskConfig``, ``build_environment()`` is called in ``__post_init__``.
    list_params : list[str] | None
        All free parameter names (RL + fixed SSM), in order.
        If None, auto-derived: ``learning_process.free_params`` + fixed SSM params.
    bounds : dict[str, tuple[float, float]] | None
        Parameter bounds. If None, auto-derived from learning_process.param_bounds
        + SSM model config param_bounds.
    params_default : list[float] | None
        Default values in same order as list_params. If None, auto-derived.
    choices : tuple[int, ...] | None
        SSM response labels (e.g., (-1, 1)). If None, taken from task_environment.
    response : list[str]
        Response column names. Default ["rt", "response"].
    response_mapping : Literal["auto"] | dict[int, int]
        Mapping from SSM response labels to zero-based learning actions.
        ``"auto"`` maps labels by ``task_environment.response_labels`` order.
    learning_backend : Literal["auto", "python", "jax"]
        Learning-process backend used for simulation and exported HSSM metadata.
        ``"auto"`` selects JAX when the process implements it and JAX is installed;
        otherwise it selects Python.
    gradient : Literal["auto", "available", "unavailable"]
        Gradient-support policy for HSSM integration metadata.
    include_action : bool
        Whether simulator output includes the derived zero-based ``action`` column.
        Default False.
    extra_fields : list[str] | None
        Extra data columns beyond response. Default: ["feedback"] + task_environment.extra_fields.
    computed_param_mapping : dict[str, str] | None
        Optional override for non-name-matching handshakes.
        Maps learning process output name -> SSM param name.
        E.g., {"drift": "v"} if learning process outputs "drift" but SSM expects "v".
        Default: None (same-name linking).
    ssm_kwargs : dict
        Default kwargs for the underlying SSM simulator call.
        Default: {"delta_t": 0.001, "max_t": 20.0}.
    """

    model_name: str
    description: str
    decision_process: str
    learning_process: LearningProcess
    task_environment: TaskEnvironment | TaskConfig

    # Auto-derivable fields (None = derive from components)
    list_params: list[str] | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    params_default: list[float] | None = None
    choices: tuple[int, ...] | None = None
    response: list[str] = field(default_factory=lambda: ["rt", "response"])
    response_mapping: Literal["auto"] | dict[int, int] = "auto"
    learning_backend: Literal["auto", "python", "jax"] = "auto"
    gradient: Literal["auto", "available", "unavailable"] = "auto"
    include_action: bool = False
    extra_fields: list[str] | None = None

    # Optional handshake override
    computed_param_mapping: dict[str, str] | None = None

    # SSM simulator defaults
    ssm_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"delta_t": 0.001, "max_t": 20.0}
    )

    def __post_init__(self):
        """Auto-build task environment and derive missing fields."""
        # Convert TaskConfig -> TaskEnvironment
        if isinstance(self.task_environment, TaskConfig):
            self.task_environment = self.task_environment.build_environment()

        # Load SSM model config for validation and auto-derivation
        self._ssm_config = ModelConfigBuilder.from_model(self.decision_process)

        # Auto-derive fields if not provided
        if self.choices is None:
            self.choices = tuple(self.task_environment.response_labels)

        self.response_to_action = self._normalize_response_mapping()
        self._validate_ssm_choices()
        self.resolved_learning_backend = self._resolve_learning_backend()
        self.resolved_gradient = self._resolve_gradient()

        if self.extra_fields is None:
            base_extra = ["feedback"]
            env_extra = self.task_environment.extra_fields or []
            self.extra_fields = base_extra + [
                f for f in env_extra if f not in base_extra
            ]

        # Resolve the handshake: which SSM params are computed vs fixed
        self._resolve_handshake()

        # Auto-derive list_params, bounds, params_default
        if self.list_params is None:
            self.list_params = self._derive_list_params()
        if self.bounds is None:
            self.bounds = self._derive_bounds()
        if self.params_default is None:
            self.params_default = self._derive_params_default()

    def _normalize_response_mapping(self) -> dict[int, int]:
        """Normalize response labels to zero-based action indices."""
        task_environment = cast(TaskEnvironment, self.task_environment)
        response_labels = list(task_environment.response_labels)
        n_arms = task_environment.n_arms
        if tuple(response_labels) != tuple(self.choices):
            raise ValueError(
                "choices must match task_environment.response_labels. "
                f"Got choices={self.choices} and response_labels={response_labels}."
            )

        if self.response_mapping == "auto":
            return {label: action for action, label in enumerate(response_labels)}

        mapping = {
            int(response): int(action)
            for response, action in self.response_mapping.items()
        }
        label_set = set(response_labels)
        mapping_labels = set(mapping)
        if mapping_labels != label_set:
            missing = sorted(label_set - mapping_labels)
            extra = sorted(mapping_labels - label_set)
            raise ValueError(
                "response_mapping must cover response labels exactly. "
                f"Missing: {missing}; extra: {extra}."
            )

        values = list(mapping.values())
        expected_actions = set(range(n_arms))
        action_set = set(values)
        if len(action_set) != len(values):
            raise ValueError("response_mapping action values must be unique")
        if action_set != expected_actions:
            raise ValueError(
                "response_mapping action values must be exactly "
                f"{sorted(expected_actions)}. Got {sorted(action_set)}."
            )
        return mapping

    def _available_learning_backends(self) -> tuple[str, ...]:
        """Return the learning process backends declared or implied by methods."""
        declared = getattr(self.learning_process, "available_backends", None)
        if declared is not None:
            return tuple(declared)

        backends: list[str] = []
        if hasattr(self.learning_process, "compute_python") and hasattr(
            self.learning_process, "update_python"
        ):
            backends.append("python")
        if hasattr(self.learning_process, "compute_jax") and hasattr(
            self.learning_process, "update_jax"
        ):
            backends.append("jax")
        if not backends:
            backends.append("python")
        return tuple(backends)

    def _resolve_learning_backend(self) -> Literal["python", "jax"]:
        """Resolve the requested learning backend to a concrete backend."""
        if self.learning_backend not in {"auto", "python", "jax"}:
            raise ValueError(
                "learning_backend must be one of 'auto', 'python', or 'jax'. "
                f"Got {self.learning_backend!r}."
            )

        available = self._available_learning_backends()
        if self.learning_backend == "auto":
            if "jax" in available and _jax_available():
                return "jax"
            return "python"

        if self.learning_backend not in available:
            raise ValueError(
                f"Learning process does not implement the {self.learning_backend!r} "
                f"backend. Available backends: {available}."
            )
        if self.learning_backend == "jax" and not _jax_available():
            raise ValueError(
                "JAX backend requested, but JAX is not installed. Install "
                "ssm-simulators with the 'jax' extra or use learning_backend='python'."
            )
        return self.learning_backend

    def _resolve_gradient(self) -> Literal["available", "unavailable"]:
        """Resolve gradient policy from backend and learning-process declaration."""
        if self.gradient not in {"auto", "available", "unavailable"}:
            raise ValueError(
                "gradient must be one of 'auto', 'available', or 'unavailable'. "
                f"Got {self.gradient!r}."
            )

        supports_gradient = bool(
            getattr(self.learning_process, "supports_gradient", False)
        )
        if self.gradient == "auto":
            if self.resolved_learning_backend == "jax" and supports_gradient:
                return "available"
            return "unavailable"

        if self.gradient == "available":
            if self.resolved_learning_backend != "jax" or not supports_gradient:
                raise ValueError(
                    "gradient='available' requires a JAX learning backend with "
                    "declared gradient support."
                )
            return "available"

        return "unavailable"

    def _validate_ssm_choices(self) -> None:
        """Ensure task response labels match the decision simulator labels."""
        ssm_choices = self._ssm_config.get("choices")
        if ssm_choices is None:
            return
        ssm_choices_tuple = tuple(int(choice) for choice in ssm_choices)
        if tuple(self.choices) != ssm_choices_tuple:
            raise ValueError(
                "choices and task_environment.response_labels must match SSM choices "
                f"for decision_process='{self.decision_process}'. "
                f"Got choices={self.choices}; SSM choices={ssm_choices_tuple}."
            )

    def _resolve_handshake(self):
        """Resolve which SSM params are computed by learning vs fixed by user.

        Populates:
        - self._computed_ssm_params: SSM param names filled by learning process
        - self._fixed_ssm_params: SSM param names user must provide in theta
        """
        ssm_params: list[str] = list(self._ssm_config["params"])
        learning_outputs = self.learning_process.computed_params

        # Apply computed_param_mapping if provided
        mapping = self.computed_param_mapping or {}
        computed_ssm_params: list[str] = []
        for output_name in learning_outputs:
            ssm_name = mapping.get(output_name, output_name)
            computed_ssm_params.append(ssm_name)

        if len(set(computed_ssm_params)) != len(computed_ssm_params):
            raise ValueError(
                "computed_param_mapping must map learning outputs to unique "
                f"SSM parameter names. Got mapped params: {computed_ssm_params}."
            )

        self._computed_ssm_params = computed_ssm_params
        self._fixed_ssm_params: list[str] = [
            p for p in ssm_params if p not in computed_ssm_params
        ]

    @property
    def required_params(self) -> list[str]:
        """Parameters that simulation requires from ``theta``."""
        return list(self.learning_process.free_params) + list(self._fixed_ssm_params)

    def _derive_list_params(self) -> list[str]:
        """RL free params + fixed SSM params (in that order)."""
        return list(self.learning_process.free_params) + self._fixed_ssm_params

    def _derive_bounds(self) -> dict[str, tuple[float, float]]:
        """Merge RL param bounds + SSM param bounds for fixed params."""
        bounds = dict(self.learning_process.param_bounds)
        ssm_bounds_dict = self._ssm_config.get("param_bounds_dict", {})
        if not ssm_bounds_dict:
            # Fallback: build from parallel arrays
            ssm_param_names = self._ssm_config["params"]
            ssm_lower = self._ssm_config["param_bounds"][0]
            ssm_upper = self._ssm_config["param_bounds"][1]
            ssm_bounds_dict = {
                name: (float(lo), float(hi))
                for name, lo, hi in zip(ssm_param_names, ssm_lower, ssm_upper)
            }
        for p in self._fixed_ssm_params:
            bounds[p] = ssm_bounds_dict[p]
        return bounds

    def _derive_params_default(self) -> list[float]:
        """Default values in list_params order."""
        rl_defaults = self.learning_process.default_params
        ssm_defaults_list = self._ssm_config["default_params"]
        ssm_param_names = self._ssm_config["params"]
        ssm_defaults = dict(zip(ssm_param_names, ssm_defaults_list))
        defaults = []
        for p in self.list_params:
            if p in rl_defaults:
                defaults.append(float(rl_defaults[p]))
            elif p in ssm_defaults:
                defaults.append(float(ssm_defaults[p]))
            else:
                raise ValueError(f"No default value for param '{p}'")
        return defaults

    def validate(self) -> None:
        """Validate config consistency. Called by Simulator.__init__().

        Checks:
        1. decision_process exists in ssm-simulators registry
        2. Handshake: computed + fixed params cover all SSM params exactly once
        3. No param is both computed and fixed
        4. list_params length matches params_default length
        5. All list_params have bounds
        """
        ssm_params = set(self._ssm_config["params"])

        # Handshake coverage
        computed = set(self._computed_ssm_params)
        fixed = set(self._fixed_ssm_params)
        covered = computed | fixed
        missing = ssm_params - covered
        if missing:
            raise ValueError(
                f"SSM model '{self.decision_process}' requires params "
                f"{sorted(ssm_params)}, but the following are neither computed "
                f"by the learning process nor available as fixed params: "
                f"{sorted(missing)}. Learning process computes: {sorted(computed)}. "
                f"Fixed SSM params: {sorted(fixed)}."
            )

        # No overlap
        overlap = computed & fixed
        if overlap:
            raise ValueError(
                f"Params {sorted(overlap)} are both computed by the learning "
                f"process and listed as fixed SSM params. Each param must have "
                f"exactly one source."
            )

        # Unknown computed params
        unknown_computed = computed - ssm_params
        if unknown_computed:
            raise ValueError(
                f"Learning process computes {sorted(unknown_computed)} which "
                f"are not params of the '{self.decision_process}' SSM model."
            )

        # list_params / params_default consistency
        if self.list_params and self.params_default:
            if len(self.list_params) != len(self.params_default):
                raise ValueError(
                    f"list_params length ({len(self.list_params)}) != "
                    f"params_default length ({len(self.params_default)})"
                )

        # All list_params have bounds
        if self.list_params and self.bounds:
            missing_bounds = [p for p in self.list_params if p not in self.bounds]
            if missing_bounds:
                raise ValueError(f"Missing bounds for params: {missing_bounds}")

        required_params = self.required_params
        if self.list_params:
            missing_required = sorted(set(required_params) - set(self.list_params))
            extra_params = sorted(set(self.list_params) - set(required_params))
            has_duplicates = len(set(self.list_params)) != len(self.list_params)
            if missing_required or extra_params or has_duplicates:
                raise ValueError(
                    "list_params must match required params from the learning "
                    "process and fixed SSM parameters. "
                    f"Missing: {missing_required}; extra: {extra_params}; "
                    f"required: {required_params}."
                )

        task_environment = cast(TaskEnvironment, self.task_environment)
        learning_n_actions = getattr(self.learning_process, "n_actions", None)
        if (
            learning_n_actions is not None
            and learning_n_actions != task_environment.n_arms
        ):
            raise ValueError(
                "learning_process.n_actions must match task_environment.n_arms. "
                f"Got n_actions={learning_n_actions}, n_arms={task_environment.n_arms}."
            )

    def to_hssm_config_dict(self) -> dict[str, Any]:
        """Produce a dict compatible with HSSM's RLSSMConfig.from_rlssm_dict().

        The output contains all fields from _HSSM_SHARED_FIELDS plus placeholder
        values for inference-only fields that the user must fill in on the HSSM side.

        Returns
        -------
        dict[str, Any]
            Dict ready for ``RLSSMConfig.from_rlssm_dict(result)`` after user
            fills in inference-only fields.
        """
        return {
            # Shared structural fields
            "model_name": self.model_name,
            "description": self.description,
            "decision_process": self.decision_process,
            "list_params": list(self.list_params),
            "bounds": dict(self.bounds),
            "params_default": list(self.params_default),
            "choices": tuple(self.choices),
            "response": list(self.response),
            "response_mapping": dict(self.response_to_action),
            "learning_backend": self.resolved_learning_backend,
            "gradient": self.resolved_gradient,
            "extra_fields": list(self.extra_fields) if self.extra_fields else [],
            # Inference-only placeholders (user fills on HSSM side)
            "ssm_logp_func": None,
            "learning_process": {},
            "decision_process_loglik_kind": "approx_differentiable",
            "learning_process_kind": (
                "approx_differentiable"
                if self.resolved_gradient == "available"
                else "blackbox"
            ),
        }

    def compile(self, backend: Literal["auto", "python", "jax"] = "auto"):
        """Return a validated executable compiled model."""
        from .compiled import CompiledModel

        return CompiledModel.from_config(self, backend=backend)
