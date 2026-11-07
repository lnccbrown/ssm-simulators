import random
import uuid

import numpy as np
import pytest

import cssm
import ssms
from ssms.config import model_config, register_drift, register_model_config
from ssms.config.model_config_builder import ModelConfigBuilder
from ssms.transforms import SwapIfLessConstraint


class TestModelConfig:
    def test_model_config_dict_type(self):
        assert isinstance(model_config, ssms.config.CopyOnAccessDict)

    def test_model_config_copy_on_access(self):
        model_name = random.choice(list(model_config))
        selected_model = model_config[model_name]
        list_params = selected_model["params"]
        list_params.append("p_outlier")
        assert "p_outlier" not in model_config[model_name]["params"]


class TestModelConfigBuilder:
    """Test ModelConfigBuilder API."""

    def test_model_config_builder_from_model(self):
        """Test basic model creation from name."""
        config = ModelConfigBuilder.from_model("ddm")

        # Verify basic structure
        assert isinstance(config, dict)
        assert config["name"] == "ddm"
        assert "params" in config
        assert "param_bounds_dict" in config
        assert "nchoices" in config

        # Verify DDM parameters
        expected_params = ["v", "a", "z", "t"]
        assert config["params"] == expected_params
        assert config["nchoices"] == 2

        # Verify param_bounds_dict has all params
        for param in expected_params:
            assert param in config["param_bounds_dict"]

    def test_model_config_builder_with_custom_bounds(self):
        """Test parameter bound customization."""
        # Create config with custom bounds
        config = ModelConfigBuilder.from_model(
            "ddm", param_bounds=[[-4, 0.3, 0.1, 0], [4, 3.0, 0.9, 2.0]]
        )

        # Verify param_bounds was set
        assert config["param_bounds"] == [[-4, 0.3, 0.1, 0], [4, 3.0, 0.9, 2.0]]

        # Verify param_bounds_dict was updated to match param_bounds
        assert config["param_bounds_dict"]["v"] == (-4, 4)
        assert config["param_bounds_dict"]["a"] == (0.3, 3.0)
        assert config["param_bounds_dict"]["z"] == (0.1, 0.9)
        assert config["param_bounds_dict"]["t"] == (0, 2.0)

        # Verify structure (tuples of (lower, upper))
        for param in config["params"]:
            bounds = config["param_bounds_dict"][param]
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]  # Lower < upper

    def test_model_config_builder_invalid_model(self):
        """Test error handling for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfigBuilder.from_model("nonexistent_model")

    def test_model_config_builder_with_constraints(self):
        """Test adding parameter sampling constraints."""
        # Use angle model which has constraints
        config = ModelConfigBuilder.from_model("angle")

        # Angle model should have swap constraints for v parameters
        if "parameter_sampling_constraints" in config:
            constraints = config["parameter_sampling_constraints"]
            assert isinstance(constraints, list)
            # Angle has constraints for ordering v parameters
            assert len(constraints) > 0

    def test_model_config_builder_ornstein(self):
        """Test with Ornstein model (has additional parameters)."""
        config = ModelConfigBuilder.from_model("ornstein")

        # Verify Ornstein has the leak parameter 'g'
        assert "g" in config["params"]
        assert "g" in config["param_bounds_dict"]

    def test_model_config_builder_race_model(self):
        """Test with race model (multi-choice)."""
        config = ModelConfigBuilder.from_model("race_3")

        # Verify race has multiple choice parameters
        assert config["nchoices"] == 3
        assert "v0" in config["params"]
        assert "v1" in config["params"]
        assert "v2" in config["params"]

    def test_lba4_config(self):
        """Test plain 4-choice LBA model config."""
        config = ModelConfigBuilder.from_model("lba4")

        assert config["name"] == "lba4"
        assert config["params"] == ["A", "b", "v0", "v1", "v2", "v3"]
        assert config["n_params"] == 6
        assert config["default_params"] == [0.3, 0.5, 0.25, 0.25, 0.25, 0.25]
        assert config["nchoices"] == 4
        assert config["choices"] == [0, 1, 2, 3]
        assert config["n_particles"] == 4
        assert "param_bounds_dict" in config
        assert set(config["param_bounds_dict"]) == set(config["params"])

    @pytest.mark.parametrize("model_name", ["lba2", "lba3", "lba4"])
    def test_plain_lba_sampling_enforces_threshold_above_start(self, model_name):
        """Test plain LBA sampling keeps public threshold b above start range A."""
        config = ModelConfigBuilder.from_model(model_name)
        transforms = ModelConfigBuilder.get_sampling_transforms(config)

        assert len(transforms) == 1
        transform = transforms[0]
        assert isinstance(transform, SwapIfLessConstraint)
        assert transform.param_a == "b"
        assert transform.param_b == "A"

        theta = {
            "A": np.array([0.8, 0.2], dtype=np.float32),
            "b": np.array([0.3, 0.7], dtype=np.float32),
        }
        transformed = transform.apply(theta)

        assert np.all(transformed["b"] > transformed["A"])

    @pytest.mark.parametrize(
        "model_name", ["lba_3_vs_constraint", "lba_angle_3_vs_constraint"]
    )
    def test_vs_constraint_lba_sampling_satisfies_simulator_constraints(
        self, model_name
    ):
        """Sampled drift rates sum to 1 and z stays below a (#114)."""
        config = ModelConfigBuilder.from_model(model_name)
        transforms = ModelConfigBuilder.get_sampling_transforms(config)

        rng = np.random.default_rng(0)
        lower, upper = (np.asarray(b) for b in config["param_bounds"])
        draws = rng.uniform(lower, upper, size=(1000, len(config["params"])))
        theta = {
            name: draws[:, i].astype(np.float32)
            for i, name in enumerate(config["params"])
        }
        for transform in transforms:
            theta = transform.apply(theta)

        v_sum = theta["v0"] + theta["v1"] + theta["v2"]
        np.testing.assert_allclose(v_sum, 1.0, atol=1e-5)
        assert np.all(theta["z"] < theta["a"])

    def test_model_config_builder_preserves_structure(self):
        """Test that builder preserves all expected config fields."""
        config = ModelConfigBuilder.from_model("ddm")

        # Check for essential fields
        required_fields = [
            "name",
            "params",
            "param_bounds_dict",
            "nchoices",
            "default_params",
        ]

        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

    def test_model_config_builder_from_scratch(self):
        """Test building a configuration from scratch."""

        def my_sim(v, a, **kwargs):
            """Custom simulator function."""
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        config = ModelConfigBuilder.from_scratch(
            name="my_custom_model",
            params=["v", "a"],
            simulator_function=my_sim,
            nchoices=2,
            param_bounds=[[-2, 0.5], [2, 2.0]],
            default_params=[0.0, 1.0],
        )

        # Verify structure
        assert config["name"] == "my_custom_model"
        assert config["params"] == ["v", "a"]
        assert config["n_params"] == 2
        assert config["nchoices"] == 2
        assert config["simulator"] == my_sim
        assert config["param_bounds"] == [[-2, 0.5], [2, 2.0]]
        assert config["default_params"] == [0.0, 1.0]
        assert config["n_particles"] == 1
        assert config["choices"] == [0, 1]

    def test_model_config_builder_from_scratch_with_boundary(self):
        """Test from_scratch with custom boundary function."""

        def my_sim(**kwargs):
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        def my_boundary(t, **kwargs):
            return 1.0  # Constant boundary

        config = ModelConfigBuilder.from_scratch(
            name="custom_with_boundary",
            params=["v", "a"],
            simulator_function=my_sim,
            nchoices=2,
            boundary=my_boundary,
            boundary_name="constant",
            boundary_params=["a"],
        )

        assert config["boundary"] == my_boundary
        assert config["boundary_name"] == "constant"
        assert config["boundary_params"] == ["a"]

    def test_model_config_builder_minimal_config(self):
        """Test minimal configuration creation."""

        def my_sim(**kwargs):
            return {"rts": np.array([0.5]), "choices": np.array([1]), "metadata": {}}

        config = ModelConfigBuilder.minimal_config(
            params=["v", "a", "z", "t"], simulator_function=my_sim, nchoices=2
        )

        # Verify minimal required fields
        assert config["name"] == "custom"
        assert config["params"] == ["v", "a", "z", "t"]
        assert config["n_params"] == 4
        assert config["nchoices"] == 2
        assert config["choices"] == [0, 1]
        assert config["n_particles"] == 1
        assert config["simulator"] == my_sim

    def test_model_config_builder_validate_config_valid(self):
        """Test validation of valid config."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": ["v", "a"],
            "nchoices": 2,
            "simulator": my_sim,
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)
        assert is_valid is True
        assert errors == []

    def test_model_config_builder_validate_config_missing_field(self):
        """Test validation catches missing required fields."""
        config = {
            "params": ["v", "a"],
            # Missing nchoices and simulator
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)
        assert is_valid is False
        assert len(errors) == 2
        assert any("nchoices" in err for err in errors)
        assert any("simulator" in err for err in errors)

    def test_model_config_builder_validate_config_wrong_type(self):
        """Test validation catches wrong types."""
        config = {
            "params": "not_a_list",  # Should be list
            "nchoices": "2",  # Should be int
            "simulator": "not_callable",  # Should be callable
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)
        assert is_valid is False
        assert len(errors) == 3

    def test_model_config_builder_validate_config_inconsistent_n_params(self):
        """Test validation catches inconsistent n_params."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": ["v", "a"],
            "n_params": 5,  # Inconsistent with params length
            "nchoices": 2,
            "simulator": my_sim,
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)
        assert is_valid is False
        assert any("n_params" in err for err in errors)

    def test_model_config_builder_validate_config_inconsistent_bounds(self):
        """Test validation catches inconsistent param_bounds."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": ["v", "a"],
            "nchoices": 2,
            "simulator": my_sim,
            "param_bounds": [[-1, 0.5, 0], [1, 1.5, 2]],  # 3 bounds for 2 params
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)
        assert is_valid is False
        assert any("param_bounds" in err for err in errors)

    def test_model_config_builder_validate_config_strict_mode(self):
        """Test strict validation mode."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": ["v", "a"],
            "nchoices": 2,
            "simulator": my_sim,
            # Missing recommended fields: name, n_particles, choices, param_bounds
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config, strict=True)
        assert is_valid is False
        assert len(errors) >= 4
        assert any("name" in err for err in errors)

    def test_model_config_builder_add_boundary_string(self):
        """Test adding boundary by name."""
        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["v", "a", "z", "t", "theta"],
            param_bounds=[[-3.0, 0.3, 0.1, 0.0, -0.1], [3.0, 2.5, 0.9, 2.0, 1.3]],
        )

        # Add a different boundary
        config = ModelConfigBuilder.add_boundary(config, "angle")

        assert config["boundary_name"] == "angle"
        assert "boundary" in config
        assert "boundary_params" in config
        assert callable(config["boundary"])

    def test_model_config_builder_add_boundary_callable(self):
        """Test adding custom boundary function."""
        config = ModelConfigBuilder.from_model("ddm")

        def custom_boundary(t, a, **kwargs):
            return a * (1 - 0.1 * t)

        config = ModelConfigBuilder.add_boundary(
            config, custom_boundary, boundary_params=["a"]
        )

        assert config["boundary"] == custom_boundary
        assert config["boundary_params"] == ["a"]
        assert config["boundary_name"] == "custom_boundary"

    def test_model_config_builder_add_boundary_callable_no_params(self):
        """Test adding custom boundary without params raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        def custom_boundary(t, **kwargs):
            return 1.0

        with pytest.raises(ValueError, match="Must provide boundary_params"):
            ModelConfigBuilder.add_boundary(config, custom_boundary)

    def test_model_config_builder_add_boundary_invalid_name(self):
        """Test adding boundary with invalid name raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        with pytest.raises(ValueError, match="Unknown boundary"):
            ModelConfigBuilder.add_boundary(config, "nonexistent_boundary")

    def test_model_config_builder_add_boundary_invalid_type(self):
        """Test adding boundary with invalid type raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        with pytest.raises(
            ValueError, match="boundary must be string name or callable"
        ):
            ModelConfigBuilder.add_boundary(config, 123)

    def test_model_config_builder_add_drift_string(self):
        """Test adding drift by name."""
        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["v", "a", "z", "t", "shape", "scale", "c"],
            param_bounds=[
                [-3.0, 0.3, 0.1, 0.0, 2.0, 0.01, -3.0],
                [3.0, 2.5, 0.9, 2.0, 10.0, 1.0, 3.0],
            ],
        )

        # Add a drift function
        config = ModelConfigBuilder.add_drift(config, "gamma_drift")

        assert config["drift_name"] == "gamma_drift"
        assert "drift" in config
        assert "drift_params" in config
        assert callable(config["drift"])

    def test_model_config_builder_add_drift_callable(self):
        """Test adding custom drift function."""
        config = ModelConfigBuilder.from_model("ddm")

        def custom_drift(t, v, **kwargs):
            return v * np.exp(-0.1 * t)

        config = ModelConfigBuilder.add_drift(config, custom_drift, drift_params=["v"])

        assert config["drift"] == custom_drift
        assert config["drift_params"] == ["v"]
        assert config["drift_name"] == "custom_drift"

    def test_model_config_builder_add_drift_callable_no_params(self):
        """Test adding custom drift without params raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        def custom_drift(t, **kwargs):
            return 1.0

        with pytest.raises(ValueError, match="Must provide drift_params"):
            ModelConfigBuilder.add_drift(config, custom_drift)

    def test_model_config_builder_add_drift_invalid_name(self):
        """Test adding drift with invalid name raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        with pytest.raises(ValueError, match="Unknown drift"):
            ModelConfigBuilder.add_drift(config, "nonexistent_drift")

    def test_model_config_builder_add_drift_invalid_type(self):
        """Test adding drift with invalid type raises error."""
        config = ModelConfigBuilder.from_model("ddm")

        with pytest.raises(ValueError, match="drift must be string name or callable"):
            ModelConfigBuilder.add_drift(config, 123)

    # --- Tests for with_deadline() ---

    def test_with_deadline_adds_deadline_param(self):
        """Test that with_deadline adds deadline to params list."""
        config = ModelConfigBuilder.from_model("ddm")
        assert "deadline" not in config["params"]

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert "deadline" in deadline_config["params"]
        assert deadline_config["params"][-1] == "deadline"

    def test_with_deadline_updates_name(self):
        """Test that with_deadline appends _deadline suffix to name."""
        config = ModelConfigBuilder.from_model("ddm")

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert deadline_config["name"] == "ddm_deadline"

    def test_with_deadline_updates_n_params(self):
        """Test that with_deadline increments n_params."""
        config = ModelConfigBuilder.from_model("ddm")
        original_n_params = config["n_params"]

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert deadline_config["n_params"] == original_n_params + 1

    def test_with_deadline_updates_param_bounds_list_format(self):
        """Test that with_deadline updates param_bounds in list format."""
        config = ModelConfigBuilder.from_model("ddm")
        # Ensure we have list format
        assert isinstance(config["param_bounds"], list)
        original_lower_len = len(config["param_bounds"][0])
        original_upper_len = len(config["param_bounds"][1])

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert len(deadline_config["param_bounds"][0]) == original_lower_len + 1
        assert len(deadline_config["param_bounds"][1]) == original_upper_len + 1
        # Check deadline bounds are correct (0.001, 10.0)
        assert deadline_config["param_bounds"][0][-1] == 0.001
        assert deadline_config["param_bounds"][1][-1] == 10.0

    def test_with_deadline_updates_param_bounds_dict(self):
        """Test that with_deadline updates param_bounds_dict."""
        config = ModelConfigBuilder.from_model("ddm")

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert "deadline" in deadline_config["param_bounds_dict"]
        assert deadline_config["param_bounds_dict"]["deadline"] == (0.001, 10.0)

    def test_with_deadline_updates_default_params(self):
        """Test that with_deadline appends default value to default_params."""
        config = ModelConfigBuilder.from_model("ddm")
        original_defaults_len = len(config["default_params"])

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert len(deadline_config["default_params"]) == original_defaults_len + 1
        assert deadline_config["default_params"][-1] == 10.0

    def test_with_deadline_sets_metadata_flag(self):
        """Test that with_deadline sets the deadline metadata flag."""
        config = ModelConfigBuilder.from_model("ddm")

        deadline_config = ModelConfigBuilder.with_deadline(config)

        assert deadline_config.get("deadline") is True

    def test_with_deadline_is_immutable(self):
        """Test that with_deadline does not modify the original config."""
        config = ModelConfigBuilder.from_model("ddm")
        original_params = config["params"].copy()
        original_name = config["name"]

        _ = ModelConfigBuilder.with_deadline(config)

        # Original should be unchanged
        assert config["params"] == original_params
        assert config["name"] == original_name
        assert "deadline" not in config["params"]

    def test_with_deadline_is_idempotent(self):
        """Test that calling with_deadline twice produces equivalent result."""
        config = ModelConfigBuilder.from_model("ddm")

        deadline_config = ModelConfigBuilder.with_deadline(config)
        double_deadline_config = ModelConfigBuilder.with_deadline(deadline_config)

        # Should have exactly one deadline param
        assert double_deadline_config["params"].count("deadline") == 1
        assert double_deadline_config["name"] == "ddm_deadline"
        # n_params should not increase again
        assert double_deadline_config["n_params"] == deadline_config["n_params"]

    def test_with_deadline_works_with_different_models(self):
        """Test with_deadline works with various model types."""
        models_to_test = ["ddm", "angle", "ornstein", "weibull"]

        for model_name in models_to_test:
            config = ModelConfigBuilder.from_model(model_name)
            deadline_config = ModelConfigBuilder.with_deadline(config)

            assert "deadline" in deadline_config["params"]
            assert deadline_config["name"] == f"{model_name}_deadline"
            assert "deadline" in deadline_config["param_bounds_dict"]

    # --- Tests for from_model() with deadline variant ---

    def test_from_model_with_deadline_suffix(self):
        """Test from_model parses _deadline suffix correctly."""
        config = ModelConfigBuilder.from_model("ddm_deadline")

        assert config["name"] == "ddm_deadline"
        assert "deadline" in config["params"]
        assert config.get("deadline") is True
        assert "deadline" in config["param_bounds_dict"]

    def test_from_model_deadline_with_overrides(self):
        """Test from_model with deadline applies overrides before variant."""
        custom_bounds = [[-5, 0.2, 0.05, 0, 0.001], [5, 4.0, 0.95, 3.0, 10.0]]
        config = ModelConfigBuilder.from_model(
            "ddm_deadline", param_bounds=custom_bounds
        )

        assert config["name"] == "ddm_deadline"
        assert "deadline" in config["params"]
        # Note: overrides are applied before with_deadline, so custom_bounds
        # should already include deadline bounds

    def test_from_model_deadline_invalid_base_model(self):
        """Test from_model raises error for unknown base model with deadline."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfigBuilder.from_model("nonexistent_deadline")

    def test_from_model_various_models_with_deadline(self):
        """Test from_model with deadline suffix for various models."""
        models = ["angle", "ornstein", "weibull", "levy"]

        for model_name in models:
            config = ModelConfigBuilder.from_model(f"{model_name}_deadline")

            assert config["name"] == f"{model_name}_deadline"
            assert "deadline" in config["params"]
            assert config.get("deadline") is True

    def test_from_model_deadline_equivalent_to_manual_pipeline(self):
        """Test that from_model with deadline produces same result as manual calls."""
        # Manual pipeline: get base model, then apply with_deadline
        base_config = ModelConfigBuilder.from_model("ddm")
        manual_config = ModelConfigBuilder.with_deadline(base_config)

        # Direct call with deadline suffix
        auto_config = ModelConfigBuilder.from_model("ddm_deadline")

        # Should be equivalent
        assert manual_config["name"] == auto_config["name"]
        assert manual_config["params"] == auto_config["params"]
        assert manual_config["n_params"] == auto_config["n_params"]
        assert manual_config["param_bounds_dict"] == auto_config["param_bounds_dict"]
        assert manual_config.get("deadline") == auto_config.get("deadline")


# Bounds for ddm's [v, a, z, t] plus one extra parameter in [0, 2] (#308).
_DDM_PLUS_ONE_BOUNDS = [[-2.5, 0.5, 0.3, 0.1, 0.0], [2.5, 2.0, 0.7, 0.8, 2.0]]


class TestDeclaredParameters:
    """Boundary and drift parameters live in params, with bounds (#308)."""

    def test_params_override_rebuilds_n_params_and_default_params(self):
        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["v", "a", "z", "t", "rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
        )

        assert config["n_params"] == 5
        # ddm's own defaults by name; the new parameter gets its bounds' midpoint
        assert config["default_params"] == [0.0, 1.0, 0.5, 0.001, 1.0]
        assert ModelConfigBuilder.validate_config(config, strict=True) == (True, [])

    def test_params_override_takes_base_defaults_by_name(self):
        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["t", "z", "a", "v"],
            param_bounds=[[0.0, 0.1, 0.3, -3.0], [2.0, 0.9, 2.5, 3.0]],
        )

        assert config["default_params"] == [0.001, 0.5, 1.0, 0.0]

    def test_explicit_n_params_and_default_params_win(self):
        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["v", "a", "z", "t", "rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
            n_params=5,
            default_params=[0.5, 1.2, 0.4, 0.2, 0.3],
        )

        assert config["default_params"] == [0.5, 1.2, 0.4, 0.2, 0.3]

    def test_params_override_without_bounds_for_new_param_raises(self):
        with pytest.raises(ValueError, match=r"No default value for .*'rate'"):
            ModelConfigBuilder.from_model("ddm", params=["v", "a", "z", "t", "rate"])

    def test_params_override_with_deadline_variant(self):
        config = ModelConfigBuilder.from_model(
            "ddm_deadline",
            params=["v", "a", "z", "t", "rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
        )

        assert config["params"] == ["v", "a", "z", "t", "rate", "deadline"]
        assert config["n_params"] == 6
        assert config["default_params"] == [0.0, 1.0, 0.5, 0.001, 1.0, 10.0]
        assert ModelConfigBuilder.validate_config(config) == (True, [])

    @pytest.mark.parametrize(
        ("add", "spec"),
        [
            (ModelConfigBuilder.add_boundary, "angle"),
            (ModelConfigBuilder.add_drift, "gamma_drift"),
        ],
    )
    def test_add_by_name_raises_on_undeclared_params(self, add, spec):
        config = ModelConfigBuilder.from_model("ddm")
        before = dict(config)

        with pytest.raises(ValueError, match=r"not in config\['params'\]"):
            add(config, spec)
        # Rejected before touching the config, which add_* edits in place
        assert config == before

    def test_add_callable_raises_on_undeclared_params(self):
        def exp_collapse(t, a, rate):
            return a * np.exp(-rate * t)

        with pytest.raises(ValueError, match=r"\['rate'\]"):
            ModelConfigBuilder.add_boundary(
                ModelConfigBuilder.from_model("ddm"), exp_collapse, ["a", "rate"]
            )
        with pytest.raises(ValueError, match=r"\['rate'\]"):
            ModelConfigBuilder.add_drift(
                ModelConfigBuilder.from_model("ddm"), exp_collapse, ["a", "rate"]
            )

    def test_validate_config_flags_undeclared_boundary_and_drift_params(self):
        config = ModelConfigBuilder.from_model("ddm")
        config["boundary_params"] = ["a", "rate"]
        config["drift_params"] = ["v", "decay"]

        is_valid, errors = ModelConfigBuilder.validate_config(config, strict=True)

        assert not is_valid
        assert "boundary_params entries missing from params: ['rate']" in errors
        assert "drift_params entries missing from params: ['decay']" in errors

    def test_no_builtin_config_has_undeclared_boundary_or_drift_params(self):
        registry = ssms.config.get_model_registry()
        for name in registry.list_models():
            _, errors = ModelConfigBuilder.validate_config(registry.get(name))
            assert not [e for e in errors if "missing from params" in e], name

    def test_declared_boundary_param_works_with_array_theta(self):
        """A boundary parameter declared via from_model works end to end.

        Array theta raised for the undeclared form shown in #308.
        """

        def exp_collapse(t, a, rate):
            return a * np.exp(-rate * np.asarray(t))

        config = ModelConfigBuilder.from_model(
            "ddm",
            params=["v", "a", "z", "t", "rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
        )
        config = ModelConfigBuilder.add_boundary(config, exp_collapse, ["a", "rate"])
        sim = ssms.Simulator(config)
        theta = {"v": 0.8, "a": 1.5, "z": 0.5, "t": 0.2, "rate": 0.5}

        from_array = sim.simulate(
            np.array([list(theta.values())]), n_samples=200, random_state=1
        )
        from_dict = sim.simulate(theta, n_samples=200, random_state=1)

        np.testing.assert_array_equal(from_array["rts"], from_dict["rts"])
        no_collapse = sim.simulate(
            {**theta, "rate": 0.0}, n_samples=200, random_state=1
        )
        assert from_dict["rts"].mean() < no_collapse["rts"].mean()


def _decaying_drift(t, v, decay_rate):
    # Uses the module-level numpy import, as user code would.
    return v * np.exp(-decay_rate * t)


class TestAddDriftSimulator:
    """add_drift must give a config whose simulator uses the drift (#426)."""

    _THETA = {"v": 1.0, "a": 1.2, "z": 0.5, "t": 0.3}

    @staticmethod
    def _decay_config(base="ddm", name="ddm_decay"):
        config = ModelConfigBuilder.from_model(
            base,
            name=name,
            params=["v", "a", "z", "t", "decay_rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
        )
        return ModelConfigBuilder.add_drift(
            config, _decaying_drift, ["v", "decay_rate"]
        )

    def test_registered_custom_drift_model_end_to_end(self):
        """The user route from #426: register_drift, add_drift, register."""
        drift_name = f"decaying_drift_{uuid.uuid4().hex}"
        model_name = f"ddm_decay_{uuid.uuid4().hex}"
        register_drift(drift_name, _decaying_drift, params=["v", "decay_rate"])
        config = ModelConfigBuilder.from_model(
            "ddm",
            name=model_name,
            params=["v", "a", "z", "t", "decay_rate"],
            param_bounds=_DDM_PLUS_ONE_BOUNDS,
        )
        config = ModelConfigBuilder.add_drift(config, drift_name)
        assert ModelConfigBuilder.validate_config(config, strict=True) == (True, [])
        register_model_config(model_name, config)

        def simulate(model, **extra):
            return ssms.Simulator(model).simulate(
                theta={**self._THETA, **extra}, n_samples=5000, random_state=1
            )

        ddm = simulate("ddm")
        no_decay = simulate(model_name, decay_rate=0.0)
        decay = simulate(model_name, decay_rate=3.0)

        # A constant drift reproduces ddm exactly; a decaying one slows
        # responses and lowers accuracy (the drift used to be ignored).
        np.testing.assert_array_equal(no_decay["rts"], ddm["rts"])
        np.testing.assert_array_equal(no_decay["choices"], ddm["choices"])
        assert decay["rts"].mean() > ddm["rts"].mean() + 0.2
        assert (decay["choices"] == 1).mean() < (ddm["choices"] == 1).mean() - 0.1

    def test_unregistered_callable_drift_takes_effect(self):
        config = self._decay_config()
        sim = ssms.Simulator(config)

        fast = sim.simulate({**self._THETA, "decay_rate": 0.0}, 2000, random_state=3)
        slow = sim.simulate({**self._THETA, "decay_rate": 3.0}, 2000, random_state=3)

        assert slow["rts"].mean() > fast["rts"].mean() + 0.2

    @pytest.mark.parametrize("base", ["ddm", "angle", "ddm_deadline"])
    def test_ddm_flexbound_configs_switch_to_ddm_flex(self, base):
        config = ModelConfigBuilder.from_model(base)
        boundary = config["boundary_name"]

        config = ModelConfigBuilder.add_drift(config, "constant")

        assert config["simulator"] is cssm.ddm_flex
        assert config["boundary_name"] == boundary

    @pytest.mark.parametrize(
        ("base", "drift", "simulator"),
        [
            ("gamma_drift", "constant", cssm.ddm_flex),
            ("conflict_stimflexrel1_leak", "constant", cssm.ddm_flex_leak),
        ],
    )
    def test_drift_capable_simulator_is_kept(self, base, drift, simulator):
        config = ModelConfigBuilder.add_drift(
            ModelConfigBuilder.from_model(base), drift
        )
        assert config["simulator"] is simulator

    @pytest.mark.parametrize("base", ["ddm_legacy", "full_ddm", "ornstein", "levy"])
    def test_simulator_without_drift_function_raises(self, base):
        config = ModelConfigBuilder.from_model(base)
        before = dict(config)

        with pytest.raises(ValueError, match="takes no drift function"):
            ModelConfigBuilder.add_drift(config, "constant")
        assert config == before

    def test_simulator_drift_argument_switches_ddm(self):
        """Simulator(model, drift=...) had the same silent-ignore gap."""
        sim = ssms.Simulator(
            "ddm",
            drift=lambda t, v: v * np.exp(-3.0 * t),
            drift_params=["v"],
        )
        assert sim.config["simulator"] is cssm.ddm_flex

        decay = sim.simulate(self._THETA, n_samples=2000, random_state=5)
        ddm = ssms.Simulator("ddm").simulate(self._THETA, 2000, random_state=5)
        assert decay["rts"].mean() > ddm["rts"].mean() + 0.2

    def test_simulator_drift_argument_raises_without_drift_function(self):
        with pytest.raises(ValueError, match="takes no drift function"):
            ssms.Simulator("ornstein", drift="constant")
