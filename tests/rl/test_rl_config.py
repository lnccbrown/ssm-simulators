"""Tests for rl.ModelConfig."""

import pytest

import ssms.rl as rl
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
        task_environment=rl.env.Bandit.bernoulli(probabilities=[0.7, 0.3]),
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
        assert config.choices == (0, 1)

    def test_auto_response_mapping(self):
        config = _make_default_config(
            task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1])
        )
        assert config.response_to_action == {-1: 0, 1: 1}

    def test_auto_derive_extra_fields(self):
        config = _make_default_config()
        assert config.extra_fields == ["feedback"]


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


class TestTaskConfigAutoBuild:
    def test_task_config_auto_build(self):
        config = _make_default_config(
            task_environment=rl.env.TaskConfig(
                task="bandit", reward="bernoulli", probabilities=[0.6, 0.4]
            ),
        )
        assert isinstance(config.task_environment, rl.env.Bandit)
        assert config.choices == (0, 1)

    def test_gaussian_task_config_auto_build(self):
        config = _make_default_config(
            task_environment=rl.env.TaskConfig(
                task="bandit",
                reward="gaussian",
                means=[1.0, 0.0],
                sds=[0.25, 0.5],
            ),
        )
        assert isinstance(config.task_environment, rl.env.Bandit)
        assert config.choices == (0, 1)


class TestResponseMapping:
    def test_explicit_reversed_mapping(self):
        config = _make_default_config(
            task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
            response_mapping={-1: 1, 1: 0},
        )
        assert config.response_to_action == {-1: 1, 1: 0}

    def test_missing_mapping_label_raises(self):
        with pytest.raises(ValueError, match="cover response labels exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_mapping={-1: 0},
            )

    def test_extra_mapping_label_raises(self):
        with pytest.raises(ValueError, match="cover response labels exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_mapping={-1: 0, 1: 1, 0: 0},
            )

    def test_duplicate_mapping_values_raise(self):
        with pytest.raises(ValueError, match="must be unique"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_mapping={-1: 0, 1: 0},
            )

    def test_out_of_range_mapping_values_raise(self):
        with pytest.raises(ValueError, match="must be exactly"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                response_mapping={-1: 0, 1: 2},
            )

    def test_choices_must_match_response_labels(self):
        with pytest.raises(ValueError, match="choices must match"):
            _make_default_config(
                task_environment=rl.env.Bandit.bernoulli(response_labels=[-1, 1]),
                choices=(0, 1),
            )

    def test_include_action_defaults_false(self):
        config = _make_default_config()
        assert config.include_action is False


class TestToHssmConfigDict:
    def test_all_shared_fields_present(self):
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        for field_name in _HSSM_SHARED_FIELDS:
            assert field_name in d, f"Missing field: {field_name}"
            assert d[field_name] is not None, f"Field {field_name} is None"

    def test_inference_placeholders(self):
        config = _make_default_config()
        d = config.to_hssm_config_dict()
        assert d["ssm_logp_func"] is None
        assert d["learning_process"] == {}
        assert d["decision_process_loglik_kind"] == "approx_differentiable"
        assert d["learning_process_kind"] == "blackbox"
        assert "learning_process_loglik_kind" not in d

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
        assert d["response_mapping"] == config.response_to_action
