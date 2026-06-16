"""Tests for compiled RLSSM model interfaces."""

import numpy as np
import pytest

import ssms.rl as rl


def _make_default_config(**overrides):
    defaults = dict(
        model_name="test_rlssm",
        description="Test RLSSM config",
        decision_process="angle",
        learning_process=rl.learning.RescorlaWagnerDeltaRule(
            n_actions=2, initial_q=0.5
        ),
        task_environment=rl.env.Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
    )
    defaults.update(overrides)
    return rl.ModelConfig(**defaults)


class _NoFeedbackEnvironment:
    n_arms = 2
    context_fields = []

    @property
    def response_labels(self):
        return [-1, 1]

    def reset(self, rng=None):
        pass

    def get_trial_context(self, trial_idx):
        return {}

    def sample_context(self, context, trial_idx):
        return {}


class _RewardEnvironment:
    n_arms = 2
    context_fields = ["reward"]

    @property
    def response_labels(self):
        return [-1, 1]

    def reset(self, rng=None):
        pass

    def get_trial_context(self, trial_idx):
        return {}

    def sample_context(self, context, trial_idx):
        return {"reward": 1.0}


class TestResolveModel:
    def test_resolves_preset_name_to_fresh_model_config(self):
        config = rl.resolve_model("2AB_RW_Angle")

        assert isinstance(config, rl.ModelConfig)
        assert config.model_name == "2AB_RW_Angle"

    def test_validates_and_passes_through_model_config(self):
        config = _make_default_config()

        resolved = rl.resolve_model(config)

        assert resolved is config

    def test_rejects_unsupported_model_reference(self):
        with pytest.raises(TypeError, match="str or ModelConfig"):
            rl.resolve_model(object())


class TestCompiledModel:
    def test_compile_returns_validated_metadata_snapshot(self):
        config = _make_default_config(learning_backend="python")

        compiled = config.compile(backend="python")

        assert isinstance(compiled, rl.CompiledModel)
        assert compiled.model_name == config.model_name
        assert compiled.decision_process == "angle"
        assert compiled.list_params == ["rl_alpha", "scaler", "a", "z", "t", "theta"]
        assert compiled.bounds == config.bounds
        assert compiled.params_default == config.params_default
        assert compiled.response == ["rt", "response"]
        assert compiled.choices == (-1, 1)
        assert compiled.context_fields == ["feedback"]
        assert compiled.computed_params == ["v"]
        assert compiled.response_to_choice == {-1: 0, 1: 1}
        assert compiled.learning_backend == "python"
        assert compiled.gradient == "unavailable"

    def test_resolve_model_can_compile_preset_name(self):
        compiled = rl.resolve_model("2AB_RW_Angle").compile(backend="python")

        assert isinstance(compiled, rl.CompiledModel)
        assert compiled.model_name == "2AB_RW_Angle"
        assert compiled.computed_params == ["v"]

    def test_python_subject_wise_function_matches_manual_rw_trajectory(self):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )
        compute = compiled.compile_participant_fn()
        trials = np.asarray(
            [
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 1.0, 0.0],
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        values = compute(trials)

        np.testing.assert_allclose(values, [0.0, -0.5, -1.0, -1.25])

    def test_python_subject_wise_function_can_return_dict(self):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )
        compute = compiled.compile_participant_fn(output="dict")
        trials = np.asarray([[0.5, 2.0, -1.0, 1.0]], dtype=np.float64)

        values = compute(trials)

        assert set(values) == {"v"}
        np.testing.assert_allclose(values["v"], [0.0])

    def test_subject_wise_function_uses_response_to_choice(self):
        compiled = _make_default_config(
            learning_backend="python", response_to_choice={-1: 1, 1: 0}
        ).compile(backend="python")
        compute = compiled.compile_participant_fn()
        trials = np.asarray(
            [
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        values = compute(trials)

        np.testing.assert_allclose(values, [0.0, 0.5])

    def test_jax_subject_wise_function_matches_manual_rw_trajectory(self):
        jnp = pytest.importorskip("jax.numpy")

        compiled = _make_default_config(learning_backend="jax").compile(backend="jax")
        compute = compiled.compile_participant_fn()
        trials = jnp.asarray(
            [
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 1.0, 0.0],
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 1.0, 1.0],
            ]
        )

        values = compute(trials)

        np.testing.assert_allclose(np.asarray(values), [0.0, -0.5, -1.0, -1.25])

    def test_jax_subject_wise_function_rejects_unmapped_response_labels(self):
        pytest.importorskip("jax.numpy")

        compiled = _make_default_config(learning_backend="jax").compile(backend="jax")
        compute = compiled.compile_participant_fn()
        trials = np.asarray(
            [
                [0.5, 2.0, -1.0, 1.0],
                [0.5, 2.0, 999.0, 0.0],
            ]
        )

        with pytest.raises(ValueError, match="not in response_to_choice"):
            compute(trials)

    def test_compile_jax_rejects_learning_without_jax_methods(self):
        class PythonOnlyLearning:
            computed_params = ["v"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}

            def init_state(self):
                return {"value": 0.0}

            def compute_python(self, state, trial_params):
                return {"v": 0.0}

            def update_python(self, state, action, reward, trial_params):
                return state

        config = _make_default_config(learning_process=PythonOnlyLearning())

        with pytest.raises(ValueError, match="does not implement the 'jax' backend"):
            config.compile(backend="jax")

    def test_participant_input_fields_derived_from_config(self):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )

        assert compiled.participant_input_fields() == [
            "rl_alpha",
            "scaler",
            "response",
            "feedback",
        ]
        assert (
            compiled.get_participant_input_fields()
            == compiled.participant_input_fields()
        )

    def test_participant_input_fields_can_use_custom_response_field(self):
        compiled = _make_default_config(
            learning_backend="python",
            response=["rt", "response", "choice_response"],
        ).compile(backend="python")

        assert compiled.participant_input_fields(response_field="choice_response") == [
            "rl_alpha",
            "scaler",
            "choice_response",
            "feedback",
        ]

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"output": "frame"}, "output must be"),
            (
                {"input_fields": ["rl_alpha", "rl_alpha", "response", "feedback"]},
                "input_fields must be unique",
            ),
            (
                {"input_fields": ["rl_alpha", "response", "feedback"]},
                "missing learning parameters",
            ),
            (
                {"input_fields": ["rl_alpha", "scaler", "response"]},
                "missing context fields",
            ),
            (
                {"input_fields": ["rl_alpha", "scaler", "feedback"]},
                "missing response_field",
            ),
        ],
    )
    def test_compile_participant_fn_validates_layout(self, kwargs, match):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )

        with pytest.raises(ValueError, match=match):
            compiled.compile_participant_fn(**kwargs)

    def test_zero_config_compile_participant_fn(self):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )

        compute = compiled.compile_participant_fn()
        trials = np.asarray([[0.5, 2.0, -1.0, 1.0]], dtype=np.float64)

        values = compute(trials)

        np.testing.assert_allclose(values, [0.0])

    def test_empty_context_fields_supports_choice_only_learning(self):
        class ChoiceOnlyLearning:
            computed_params = ["v"]
            free_params = ["bias"]
            param_bounds = {"bias": (-5.0, 5.0)}
            default_params = {"bias": 0.0}
            available_backends = ("python",)
            supports_gradient = False
            n_actions = 2
            required_context_fields = ["choice"]

            def init_state(self):
                return {"count": 0}

            def compute_python(self, state, trial_params, context):
                return {"v": float(trial_params["bias"] + state["count"])}

            def update_python(self, state, trial_params, context):
                return {"count": state["count"] + int(context["choice"])}

        compiled = _make_default_config(
            learning_backend="python",
            learning_process=ChoiceOnlyLearning(),
            task_environment=_NoFeedbackEnvironment(),
            context_fields=[],
        ).compile(backend="python")
        compute = compiled.compile_participant_fn()
        trials = np.asarray(
            [
                [0.5, -1.0],
                [0.5, 1.0],
                [0.5, 1.0],
            ],
            dtype=np.float64,
        )

        values = compute(trials)

        np.testing.assert_allclose(values, [0.5, 0.5, 1.5])

    def test_custom_context_field_name(self):
        compiled = _make_default_config(
            learning_backend="python",
            learning_process=rl.learning.RescorlaWagnerDeltaRule(
                n_actions=2, initial_q=0.5, feedback_field="reward"
            ),
            task_environment=_RewardEnvironment(),
            context_fields=["reward"],
        ).compile(backend="python")

        assert compiled.participant_input_fields() == [
            "rl_alpha",
            "scaler",
            "response",
            "reward",
        ]
        compute = compiled.compile_participant_fn()
        trials = np.asarray([[0.5, 2.0, -1.0, 1.0]], dtype=np.float64)

        values = compute(trials)

        np.testing.assert_allclose(values, [0.0])

    def test_multi_output_learning_with_mapping(self):
        class DualOutputLearning:
            computed_params = ["drift", "urgency"]
            free_params = ["gain"]
            param_bounds = {"gain": (0.0, 1.0)}
            default_params = {"gain": 0.5}
            available_backends = ("python",)
            supports_gradient = False
            required_context_fields = ["choice"]

            def init_state(self):
                return {"step": 0}

            def compute_python(self, state, trial_params, context):
                step = state["step"]
                gain = trial_params["gain"]
                return {
                    "drift": float(gain * step),
                    "urgency": float(0.1 * step),
                }

            def update_python(self, state, trial_params, context):
                return {"step": state["step"] + 1}

        config = _make_default_config(
            learning_backend="python",
            learning_process=DualOutputLearning(),
            task_environment=_NoFeedbackEnvironment(),
            computed_param_mapping={"drift": "v", "urgency": "theta"},
            context_fields=[],
        )
        compiled = config.compile(backend="python")
        compute = compiled.compile_participant_fn(output="dict")
        trials = np.asarray(
            [
                [0.5, -1.0],
                [0.5, 1.0],
                [0.5, 1.0],
            ],
            dtype=np.float64,
        )

        values = compute(trials)

        assert set(values) == {"v", "theta"}
        np.testing.assert_allclose(values["v"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(values["theta"], [0.0, 0.1, 0.2])

    def test_multi_output_learning_can_return_array(self):
        class DualOutputLearning:
            computed_params = ["drift", "urgency"]
            free_params = ["gain"]
            param_bounds = {"gain": (0.0, 1.0)}
            default_params = {"gain": 0.5}
            available_backends = ("python",)
            supports_gradient = False
            required_context_fields = ["choice"]

            def init_state(self):
                return {"step": 0}

            def compute_python(self, state, trial_params, context):
                step = state["step"]
                gain = trial_params["gain"]
                return {"drift": gain * step, "urgency": 0.1 * step}

            def update_python(self, state, trial_params, context):
                return {"step": state["step"] + 1}

        config = _make_default_config(
            learning_backend="python",
            learning_process=DualOutputLearning(),
            task_environment=_NoFeedbackEnvironment(),
            computed_param_mapping={"drift": "v", "urgency": "theta"},
            context_fields=[],
        )
        compute = config.compile(backend="python").compile_participant_fn()
        trials = np.asarray([[0.5, -1.0], [0.5, 1.0]], dtype=np.float64)

        values = compute(trials)

        np.testing.assert_allclose(values, [[0.0, 0.0], [0.5, 0.1]])

    def test_missing_computed_param_mapping_output_raises(self):
        class IncompleteLearning:
            computed_params = ["drift", "urgency"]
            free_params = ["gain"]
            param_bounds = {"gain": (0.0, 1.0)}
            default_params = {"gain": 0.5}
            available_backends = ("python",)
            supports_gradient = False
            required_context_fields = ["choice"]

            def init_state(self):
                return {}

            def compute_python(self, state, trial_params, context):
                return {"drift": trial_params["gain"]}

            def update_python(self, state, trial_params, context):
                return state

        config = _make_default_config(
            learning_backend="python",
            learning_process=IncompleteLearning(),
            task_environment=_NoFeedbackEnvironment(),
            computed_param_mapping={"drift": "v", "urgency": "theta"},
            context_fields=[],
        )
        compute = config.compile(backend="python").compile_participant_fn()
        trials = np.asarray([[0.5, -1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Learning process did not compute"):
            compute(trials)
