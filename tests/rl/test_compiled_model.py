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


class TestResolveModel:
    def test_resolves_preset_name_to_fresh_model_config(self):
        config = rl.resolve_model("rlssm1")

        assert isinstance(config, rl.ModelConfig)
        assert config.model_name == "rlssm1"

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
        assert compiled.extra_fields == ["feedback"]
        assert compiled.computed_params == ["v"]
        assert compiled.response_to_action == {-1: 0, 1: 1}
        assert compiled.learning_backend == "python"
        assert compiled.gradient == "unavailable"

    def test_resolve_model_can_compile_preset_name(self):
        compiled = rl.resolve_model("rlssm1").compile(backend="python")

        assert isinstance(compiled, rl.CompiledModel)
        assert compiled.model_name == "rlssm1"
        assert compiled.computed_params == ["v"]

    def test_python_subject_wise_function_matches_manual_rw_trajectory(self):
        compiled = _make_default_config(learning_backend="python").compile(
            backend="python"
        )
        compute = compiled.make_subject_wise_function(
            input_fields=["rl_alpha", "scaler", "action", "feedback"],
            action_field="action",
            reward_field="feedback",
        )
        trials = np.asarray(
            [
                [0.5, 2.0, 0.0, 1.0],
                [0.5, 2.0, 1.0, 0.0],
                [0.5, 2.0, 0.0, 1.0],
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
        compute = compiled.make_subject_wise_function(
            input_fields=["rl_alpha", "scaler", "action", "feedback"],
            action_field="action",
            reward_field="feedback",
            output="dict",
        )
        trials = np.asarray([[0.5, 2.0, 0.0, 1.0]], dtype=np.float64)

        values = compute(trials)

        assert set(values) == {"v"}
        np.testing.assert_allclose(values["v"], [0.0])

    def test_subject_wise_function_uses_response_mapping_when_action_is_absent(self):
        compiled = _make_default_config(
            learning_backend="python", response_mapping={-1: 1, 1: 0}
        ).compile(backend="python")
        compute = compiled.make_subject_wise_function(
            input_fields=["rl_alpha", "scaler", "response", "feedback"],
            response_field="response",
            reward_field="feedback",
        )
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
        compute = compiled.make_subject_wise_function(
            input_fields=["rl_alpha", "scaler", "action", "feedback"],
            action_field="action",
            reward_field="feedback",
        )
        trials = jnp.asarray(
            [
                [0.5, 2.0, 0.0, 1.0],
                [0.5, 2.0, 1.0, 0.0],
                [0.5, 2.0, 0.0, 1.0],
                [0.5, 2.0, 1.0, 1.0],
            ]
        )

        values = compute(trials)

        np.testing.assert_allclose(np.asarray(values), [0.0, -0.5, -1.0, -1.25])

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
