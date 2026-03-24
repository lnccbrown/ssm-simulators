"""RLSSMModelConfig — structural model specification for RLSSM simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ssms.config.model_config_builder import ModelConfigBuilder

from .learning_process import LearningProcess
from .task_environment import TaskConfig, TaskEnvironment


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
    "extra_fields",
)


@dataclass
class RLSSMModelConfig:
    """RLSSM model configuration for ssm-simulators.

    Describes the *structural specification* of an RLSSM model:
    which learning process, which decision process (SSM), and which task
    environment. Concrete parameter values are NOT stored here — they
    are passed as ``theta`` to ``RLSSMSimulator.simulate()``.

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
        Task-space choice values (e.g., (0, 1)). If None, taken from task_environment.
    response : list[str]
        Response column names. Default ["rt", "response"].
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
            self.choices = tuple(self.task_environment.choices)

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

    def _resolve_handshake(self):
        """Resolve which SSM params are computed by learning vs fixed by user.

        Populates:
        - self._computed_ssm_params: SSM param names filled by learning process
        - self._fixed_ssm_params: SSM param names user must provide in theta
        """
        ssm_params = self._ssm_config["params"]
        learning_outputs = self.learning_process.computed_params

        # Apply computed_param_mapping if provided
        mapping = self.computed_param_mapping or {}
        computed_ssm_params = set()
        for output_name in learning_outputs:
            ssm_name = mapping.get(output_name, output_name)
            computed_ssm_params.add(ssm_name)

        self._computed_ssm_params = list(computed_ssm_params)
        self._fixed_ssm_params = [
            p for p in ssm_params if p not in computed_ssm_params
        ]

    def _derive_list_params(self) -> list[str]:
        """RL free params + fixed SSM params (in that order)."""
        return self.learning_process.free_params + self._fixed_ssm_params

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
        """Validate config consistency. Called by RLSSMSimulator.__init__().

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
            "extra_fields": list(self.extra_fields) if self.extra_fields else [],
            # Inference-only placeholders (user fills on HSSM side)
            "ssm_logp_func": None,
            "learning_process": {},
            "decision_process_loglik_kind": "approx_differentiable",
            "learning_process_loglik_kind": "blackbox",
        }
