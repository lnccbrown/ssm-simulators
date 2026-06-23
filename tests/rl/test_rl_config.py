"""Tests for rl.ModelConfig."""

import pytest

import ssms.rl as rl
from ssms.rl import config as rl_config
from ssms.rl.config import _HSSM_SHARED_FIELDS


def _make_default_config(**overrides):
    """Helper to create a default angle + RW config with optional overrides."""
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


class TestAutoDerivation:
    def test_auto_derive_list_params(self):
        config = _make_default_config()
        # RL free params + fixed SSM params (v is computed, so excluded)
        assert config.list_params == ["rl_alpha", "scaler", "a", "z", "t", "theta"]

    def test_auto_derive_bounds(self):
        config = _make_default_config()
        bounds = config.bounds
        # RL bounds
        assert bounds["rl_alpha"] == (0.0, 1.0)
        assert bounds["scaler"] == (0.001, 10.0)
        # SSM bounds for fixed params
        assert bounds["a"] == (0.3, 3.0)
        assert bounds["z"] == (0.1, 0.9)
        assert bounds["t"] == (0.001, 2.0)
        assert bounds["theta"] == (-0.1, 1.3)
        # v should NOT be in bounds (it's computed)
        assert "v" not in bounds

    def test_auto_derive_params_default(self):
        config = _make_default_config()
        # Order matches list_params
        expected_names = ["rl_alpha", "scaler", "a", "z", "t", "theta"]
        assert config.list_params == expected_names
        assert len(config.params_default) == len(expected_names)
        # RL defaults
        assert config.params_default[0] == 0.2  # rl_alpha
        assert config.params_default[1] == 2.0  # scaler
        # SSM defaults for angle model
        assert config.params_default[2] == 1.0  # a
        assert config.params_default[3] == 0.5  # z

    def test_auto_derive_choices(self):
        config = _make_default_config()
        assert config.choices == (-1, 1)

    def test_auto_response_to_choice(self):
        config = _make_default_config(
            task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1])
        )
        assert config.response_to_choice == {-1: 0, 1: 1}

    def test_auto_derive_context_fields(self):
        config = _make_default_config()
        assert config.context_fields == ["feedback"]

    def test_participant_contract_derived_from_config(self):
        config = _make_default_config()

        contract = config.participant_contract()

        assert contract.trial_params == ("rl_alpha", "scaler")
        assert contract.computed_outputs == ("v",)
        assert contract.response_field == "response"
        assert contract.context_fields == ("feedback",)
        assert contract.input_fields == (
            "rl_alpha",
            "scaler",
            "response",
            "feedback",
        )

    def test_participant_contract_requires_configured_response_field(self):
        config = _make_default_config()

        with pytest.raises(ValueError, match="must appear in config.response"):
            rl_config.derive_participant_contract(config, response_field="choice")


class TestDualAlphaAutoDerivation:
    def test_auto_derive_list_params(self):
        config = _make_default_config(
            learning_process=rl.learning.RescorlaWagnerDualAlphaRule()
        )
        assert config.list_params == [
            "rl_alpha",
            "rl_alpha_neg",
            "scaler",
            "a",
            "z",
            "t",
            "theta",
        ]

    def test_auto_derive_bounds(self):
        config = _make_default_config(
            learning_process=rl.learning.RescorlaWagnerDualAlphaRule()
        )
        assert config.bounds["rl_alpha"] == (0.0, 1.0)
        assert config.bounds["rl_alpha_neg"] == (0.0, 1.0)
        assert config.bounds["scaler"] == (0.001, 10.0)
        assert "v" not in config.bounds

    def test_auto_derive_params_default(self):
        config = _make_default_config(
            learning_process=rl.learning.RescorlaWagnerDualAlphaRule()
        )
        assert config.params_default[:3] == [0.2, 0.2, 2.0]


class TestHandshakeValidation:
    def test_valid_config_validates(self):
        config = _make_default_config()
        config.validate()  # Should not raise

    def test_unknown_computed_param(self):
        """Learning computes a param not in the SSM model."""

        class BadLearning:
            computed_params = ["q_diff"]  # Not an SSM param
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}

            def reset(self, **kwargs):
                pass

            def compute_ssm_params(self, trial_params):
                return {"q_diff": 0.0}

            def update(self, action, reward, trial_params):
                pass

        config = _make_default_config(learning_process=BadLearning())
        with pytest.raises(ValueError, match="not params of the 'angle' SSM model"):
            config.validate()

    def test_computed_param_mapping(self):
        """Learning computes 'drift' but SSM expects 'v' — mapping resolves it."""

        class DriftLearning:
            computed_params = ["drift"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}

            def reset(self, **kwargs):
                pass

            def compute_ssm_params(self, trial_params):
                return {"drift": 0.0}

            def update(self, action, reward, trial_params):
                pass

        config = _make_default_config(
            learning_process=DriftLearning(),
            computed_param_mapping={"drift": "v"},
        )
        config.validate()  # Should not raise
        assert "v" not in config.bounds  # v is computed, not fixed

    def test_computed_param_mapping_collision_raises(self):
        """Multiple learning outputs cannot map to one SSM parameter."""

        class CollidingLearning:
            computed_params = ["drift_left", "drift_right"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}

            def reset(self, **kwargs):
                pass

            def compute_ssm_params(self, trial_params):
                return {"drift_left": 0.0, "drift_right": 0.0}

            def update(self, action, reward, trial_params):
                pass

        with pytest.raises(ValueError, match="computed_param_mapping"):
            _make_default_config(
                learning_process=CollidingLearning(),
                computed_param_mapping={"drift_left": "v", "drift_right": "v"},
            )


class TestTaskConfigAutoBuild:
    def test_task_config_auto_build(self):
        config = _make_default_config(
            task_environment=rl.env.TaskConfig(
                task="bandit",
                reward="bernoulli",
                probabilities=[0.6, 0.4],
                response_labels=[-1, 1],
            ),
        )
        assert isinstance(config.task_environment, rl.env.Bandit)
        assert config.choices == (-1, 1)

    def test_gaussian_task_config_auto_build(self):
        config = _make_default_config(
            task_environment=rl.env.TaskConfig(
                task="bandit",
                reward="gaussian",
                means=[1.0, 0.0],
                sds=[0.25, 0.5],
                response_labels=[-1, 1],
            ),
        )
        assert isinstance(config.task_environment, rl.env.Bandit)
        assert config.choices == (-1, 1)


class TestResponseToChoice:
    def test_explicit_reversed_mapping(self):
        config = _make_default_config(
            task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
            response_to_choice={-1: 1, 1: 0},
        )
        assert config.response_to_choice == {-1: 1, 1: 0}

    def test_missing_mapping_label_raises(self):
        with pytest.raises(ValueError, match="cover response labels exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_to_choice={-1: 0},
            )

    def test_extra_mapping_label_raises(self):
        with pytest.raises(ValueError, match="cover response labels exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_to_choice={-1: 0, 1: 1, 0: 0},
            )

    def test_duplicate_mapping_values_raise(self):
        with pytest.raises(ValueError, match="must be unique"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_to_choice={-1: 0, 1: 0},
            )

    def test_out_of_range_mapping_values_raise(self):
        with pytest.raises(ValueError, match="must be exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_to_choice={-1: 0, 1: 2},
            )

    def test_choices_must_match_response_labels(self):
        with pytest.raises(ValueError, match="choices must match"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                choices=(0, 1),
            )

    def test_response_labels_must_match_ssm_choices(self):
        with pytest.raises(ValueError, match="SSM choices"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(
                    probabilities=[0.7, 0.3], response_labels=[0, 1]
                ),
                choices=(0, 1),
            )

    def test_include_choice_defaults_false(self):
        config = _make_default_config()
        assert config.include_choice is False


class TestLearningBackendPolicy:
    def test_invalid_learning_backend_policy_raises(self):
        with pytest.raises(ValueError, match="learning_backend must be"):
            _make_default_config(learning_backend="numpy")

    def test_invalid_gradient_policy_raises(self):
        with pytest.raises(ValueError, match="gradient must be"):
            _make_default_config(gradient="maybe")

    def test_auto_backend_uses_python_for_python_only_learning_process(self):
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

        assert config.resolved_learning_backend == "python"
        assert config.resolved_gradient == "unavailable"

    def test_requested_jax_backend_raises_when_jax_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(rl_config, "_jax_available", lambda: False)

        with pytest.raises(ValueError, match="JAX backend requested"):
            _make_default_config(learning_backend="jax")

    def test_requested_gradient_requires_jax_backend(self):
        with pytest.raises(ValueError, match="gradient='available'"):
            _make_default_config(learning_backend="python", gradient="available")

    def test_auto_backend_raises_for_jax_only_learning_without_jax(self, monkeypatch):
        class JaxOnlyLearning:
            computed_params = ["v"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}
            available_backends = ("jax",)
            supports_gradient = True

            def init_state(self):
                return {"value": 0.0}

            def compute_python(self, state, trial_params):
                return {"v": 0.0}

            def update_python(self, state, action, reward, trial_params):
                return state

        monkeypatch.setattr(rl_config, "_jax_available", lambda: False)

        with pytest.raises(ValueError, match="JAX-only learning process detected"):
            _make_default_config(learning_process=JaxOnlyLearning())


class TestListParamsValidation:
    def test_list_params_without_default_value_raises(self):
        with pytest.raises(ValueError, match="No default value"):
            _make_default_config(
                list_params=["rl_alpha", "scaler", "a", "z", "t", "theta", "missing"]
            )

    def test_list_params_length_must_match_params_default_length(self):
        config = _make_default_config(
            list_params=["rl_alpha", "scaler", "a", "z", "t", "theta"],
            params_default=[0.2],
        )

        with pytest.raises(ValueError, match="params_default length"):
            config.validate()

    def test_list_params_must_have_bounds(self):
        config = _make_default_config(
            list_params=["rl_alpha", "scaler", "a", "z", "t", "theta"],
            bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.001, 10.0)},
        )

        with pytest.raises(ValueError, match="Missing bounds"):
            config.validate()

    def test_list_params_must_include_required_params(self):
        config = _make_default_config(list_params=["rl_alpha", "scaler"])
        with pytest.raises(ValueError, match="list_params must match"):
            config.validate()

    def test_empty_list_params_raises(self):
        config = _make_default_config(list_params=[])
        with pytest.raises(ValueError, match="list_params cannot be empty"):
            config.validate()


class TestModelConfigValidation:
    def test_computed_and_fixed_params_must_not_overlap(self):
        config = _make_default_config()
        config._fixed_ssm_params = ["v", "a", "z", "t", "theta"]

        with pytest.raises(ValueError, match="both computed"):
            config.validate()

    def test_learning_process_action_count_must_match_task(self):
        class ThreeActionLearning:
            computed_params = ["v"]
            free_params = ["alpha"]
            param_bounds = {"alpha": (0.0, 1.0)}
            default_params = {"alpha": 0.2}
            n_actions = 3

            def reset(self, **kwargs):
                pass

            def compute_ssm_params(self, trial_params):
                return {"v": 0.0}

            def update(self, action, reward, trial_params):
                pass

        config = _make_default_config(
            learning_process=ThreeActionLearning(),
        )

        with pytest.raises(ValueError, match="n_actions must match"):
            config.validate()

    def test_non_discrete_task_environment_rejected_at_construction(self):
        class ContextOnlyEnvironment:
            context_fields = ["cue"]

            def reset(self, rng=None):
                pass

            def get_trial_context(self, trial_idx):
                return {"cue": float(trial_idx)}

            def sample_context(self, context, trial_idx):
                return {}

        with pytest.raises(ValueError, match="DiscreteChoiceEnvironment"):
            _make_default_config(task_environment=ContextOnlyEnvironment())

    def test_response_schema_must_include_response_column(self):
        config = _make_default_config(response=["rt"])

        with pytest.raises(ValueError, match="response"):
            config.validate()


class TestToHssmConfigDict:
    def test_all_shared_fields_present(self):
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        for field_name in _HSSM_SHARED_FIELDS:
            assert field_name in d, f"Missing field: {field_name}"
            assert d[field_name] is not None, f"Field {field_name} is None"

    def test_inference_placeholders(self):
        config = _make_default_config(learning_backend="python")
        d = config.to_hssm_config_dict()
        assert d["ssm_logp_func"] is None
        assert d["learning_process"] == {}
        assert d["decision_process_loglik_kind"] == "approx_differentiable"
        assert d["learning_process_kind"] == "blackbox"
        assert d["learning_backend"] == "python"
        assert d["gradient"] == "unavailable"
        assert "learning_process_loglik_kind" not in d

    def test_learning_process_kind_reflects_resolved_gradient(self):
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        expected_kind = (
            "approx_differentiable"
            if config.resolved_gradient == "available"
            else "blackbox"
        )
        assert d["learning_process_kind"] == expected_kind
        assert d["learning_backend"] == config.resolved_learning_backend
        assert d["gradient"] == config.resolved_gradient

    def test_contract_consistency(self):
        """list_params length == params_default length, all have bounds."""
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        assert len(d["list_params"]) == len(d["params_default"])
        for p in d["list_params"]:
            assert p in d["bounds"], f"Missing bounds for {p}"

    def test_values_match_config(self):
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        assert d["model_name"] == config.model_name
        assert d["decision_process"] == config.decision_process
        assert d["choices"] == config.choices
        assert d["response"] == config.response
        assert d["response_to_choice"] == config.response_to_choice
