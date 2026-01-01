"""Tests for constraint factory."""

import pytest
from ssms.dataset_generators.parameter_samplers.constraints.factory import (
    create_constraint_from_config,
    get_constraints_from_model_config,
)
from ssms.dataset_generators.parameter_samplers.constraints import (
    SwapIfLessConstraint,
    NormalizeToSumConstraint,
)


class TestConstraintFactory:
    """Test suite for transform factory functions."""

    def test_create_swap_transform(self):
        """Test creating swap transform from config."""
        config = {"type": "swap", "param_a": "a", "param_b": "z"}
        transform = create_constraint_from_config(config)

        assert isinstance(transform, SwapIfLessConstraint)
        assert transform.param_a == "a"
        assert transform.param_b == "z"

    def test_create_normalize_transform(self):
        """Test creating normalize transform from config."""
        config = {"type": "normalize", "param_names": ["v1", "v2", "v3"]}
        transform = create_constraint_from_config(config)

        assert isinstance(transform, NormalizeToSumConstraint)
        assert transform.param_names == ["v1", "v2", "v3"]

    def test_unknown_constraint_type(self):
        """Test that unknown transform types raise error."""
        config = {"type": "unknown_type"}

        with pytest.raises(ValueError, match="Unknown constraint type"):
            create_constraint_from_config(config)

    def test_swap_missing_parameters(self):
        """Test that swap config with missing params raises error."""
        # Missing param_b
        config = {"type": "swap", "param_a": "a"}
        with pytest.raises(ValueError, match="requires 'param_a' and 'param_b'"):
            create_constraint_from_config(config)

        # Missing param_a
        config = {"type": "swap", "param_b": "z"}
        with pytest.raises(ValueError, match="requires 'param_a' and 'param_b'"):
            create_constraint_from_config(config)

    def test_normalize_missing_parameters(self):
        """Test that normalize config with missing params raises error."""
        config = {"type": "normalize"}

        with pytest.raises(ValueError, match="requires 'param_names'"):
            create_constraint_from_config(config)

    def test_get_constraints_from_model_config_with_transforms(self):
        """Test extracting transforms from model config."""
        model_config = {
            "name": "test_model",
            "parameter_sampling_constraints": [
                {"type": "swap", "param_a": "a", "param_b": "z"},
                {"type": "normalize", "param_names": ["v1", "v2"]},
            ],
        }

        transforms = get_constraints_from_model_config(model_config)

        assert len(transforms) == 2
        assert isinstance(transforms[0], SwapIfLessConstraint)
        assert isinstance(transforms[1], NormalizeToSumConstraint)

    def test_get_constraints_from_model_config_empty(self):
        """Test extracting transforms from model config with no transforms."""
        model_config = {"name": "test_model"}

        transforms = get_constraints_from_model_config(model_config)

        assert transforms == []

    def test_get_constraints_from_model_config_empty_list(self):
        """Test model config with explicitly empty transform list."""
        model_config = {"name": "test_model", "parameter_sampling_constraints": []}

        transforms = get_constraints_from_model_config(model_config)

        assert transforms == []

    def test_get_transforms_from_real_model_config(self):
        """Test with actual model configs from the system."""
        from ssms.config import model_config

        # Test lba_angle_3 (should have swap transform)
        lba_config = model_config["lba_angle_3"]
        lba_transforms = get_constraints_from_model_config(lba_config)

        assert len(lba_transforms) == 1
        assert isinstance(lba_transforms[0], SwapIfLessConstraint)
        assert lba_transforms[0].param_a == "a"
        assert lba_transforms[0].param_b == "z"

        # Test dev_rlwm_lba_race_v1 (should have 3 transforms)
        rlwm_config = model_config["dev_rlwm_lba_race_v1"]
        rlwm_transforms = get_constraints_from_model_config(rlwm_config)

        assert len(rlwm_transforms) == 3
        assert isinstance(rlwm_transforms[0], NormalizeToSumConstraint)
        assert isinstance(rlwm_transforms[1], NormalizeToSumConstraint)
        assert isinstance(rlwm_transforms[2], SwapIfLessConstraint)

        # Test model without transforms (e.g., ddm)
        ddm_config = model_config["ddm"]
        ddm_transforms = get_constraints_from_model_config(ddm_config)

        assert ddm_transforms == []

    def test_multiple_transforms_application_order(self):
        """Test that transforms are applied in the order defined in config."""
        model_config = {
            "name": "test_model",
            "parameter_sampling_constraints": [
                {"type": "normalize", "param_names": ["v1", "v2"]},
                {"type": "swap", "param_a": "a", "param_b": "z"},
            ],
        }

        transforms = get_constraints_from_model_config(model_config)

        # First should be normalize, second should be swap
        assert isinstance(transforms[0], NormalizeToSumConstraint)
        assert isinstance(transforms[1], SwapIfLessConstraint)
