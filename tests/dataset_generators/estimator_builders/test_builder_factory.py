"""Tests for builder factory.

These tests verify that the builder factory correctly:
1. Selects KDE builder by default
2. Respects explicit estimator_type configuration
3. Handles legacy use_pyddm_pdf flag
4. Provides helpful errors for unsupported estimators
"""

import pytest

from ssms.config import model_config
from ssms.dataset_generators.estimator_builders import (
    KDEEstimatorBuilder,
    create_estimator_builder,
)


@pytest.fixture
def ddm_model_config():
    """DDM model configuration."""
    return model_config["ddm"]


@pytest.fixture
def race_model_config():
    """Race model configuration (incompatible with PyDDM)."""
    return model_config["race_3"]


def test_factory_defaults_to_kde(ddm_model_config):
    """Test that factory defaults to KDE when no estimator_type is specified."""
    generator_config = {}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    assert isinstance(builder, KDEEstimatorBuilder)


def test_factory_explicit_kde(ddm_model_config):
    """Test that factory creates KDE builder when explicitly requested."""
    generator_config = {"estimator_type": "kde"}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    assert isinstance(builder, KDEEstimatorBuilder)


def test_factory_kde_case_insensitive(ddm_model_config):
    """Test that estimator_type is case-insensitive."""
    for variant in ["kde", "KDE", "Kde", "kDe"]:
        generator_config = {"estimator_type": variant}

        builder = create_estimator_builder(generator_config, ddm_model_config)

        assert isinstance(builder, KDEEstimatorBuilder)


def test_factory_pyddm_creates_builder(ddm_model_config):
    """Test that PyDDM estimator builder is created when pyddm is installed."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    generator_config = {"estimator_type": "pyddm"}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    from ssms.dataset_generators.estimator_builders.pyddm_builder import (
        PyDDMEstimatorBuilder,
    )

    assert isinstance(builder, PyDDMEstimatorBuilder)


def test_factory_legacy_pyddm_flag(ddm_model_config):
    """Test that legacy use_pyddm_pdf flag is recognized."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    generator_config = {"use_pyddm_pdf": True}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    from ssms.dataset_generators.estimator_builders.pyddm_builder import (
        PyDDMEstimatorBuilder,
    )

    assert isinstance(builder, PyDDMEstimatorBuilder)


def test_factory_legacy_flag_overrides_explicit_kde(ddm_model_config):
    """Test that use_pyddm_pdf=True overrides estimator_type='kde'."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    generator_config = {"estimator_type": "kde", "use_pyddm_pdf": True}

    # Legacy flag takes precedence
    builder = create_estimator_builder(generator_config, ddm_model_config)

    from ssms.dataset_generators.estimator_builders.pyddm_builder import (
        PyDDMEstimatorBuilder,
    )

    assert isinstance(builder, PyDDMEstimatorBuilder)


def test_factory_legacy_flag_false_uses_kde(ddm_model_config):
    """Test that use_pyddm_pdf=False doesn't override kde."""
    generator_config = {"estimator_type": "kde", "use_pyddm_pdf": False}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    assert isinstance(builder, KDEEstimatorBuilder)


def test_factory_unknown_estimator_type(ddm_model_config):
    """Test that unknown estimator_type raises ValueError."""
    generator_config = {"estimator_type": "unknown"}

    with pytest.raises(ValueError, match="Unknown estimator_type: 'unknown'"):
        create_estimator_builder(generator_config, ddm_model_config)


def test_factory_error_message_helpful(ddm_model_config):
    """Test that error messages are helpful and informative."""
    generator_config = {"estimator_type": "analytical"}

    with pytest.raises(ValueError) as exc_info:
        create_estimator_builder(generator_config, ddm_model_config)

    error_msg = str(exc_info.value)
    # Should mention supported types
    assert "kde" in error_msg.lower()
    assert "pyddm" in error_msg.lower()


def test_factory_pyddm_incompatible_model_fails(race_model_config):
    """Test that PyDDM with incompatible model raises ValueError."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    generator_config = {"estimator_type": "pyddm"}

    # Race model is incompatible with PyDDM
    with pytest.raises(ValueError, match="not compatible with PyDDM"):
        create_estimator_builder(generator_config, race_model_config)


def test_factory_pyddm_error_suggests_kde(race_model_config):
    """Test that PyDDM incompatibility error suggests using KDE."""
    pytest.importorskip("pyddm")  # Skip if pyddm not installed

    generator_config = {"estimator_type": "pyddm"}

    with pytest.raises(ValueError) as exc_info:
        create_estimator_builder(generator_config, race_model_config)

    error_msg = str(exc_info.value)
    # Should suggest alternative
    assert "kde" in error_msg.lower()


def test_factory_with_kde_specific_config(ddm_model_config):
    """Test that factory passes config to KDE builder correctly."""
    generator_config = {"estimator_type": "kde", "kde_displace_t": True}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    assert isinstance(builder, KDEEstimatorBuilder)
    assert builder.displace_t is True


def test_factory_preserves_generator_config(ddm_model_config):
    """Test that factory doesn't modify the original config."""
    original_config = {"estimator_type": "kde", "n_samples": 1000}
    config_copy = original_config.copy()

    builder = create_estimator_builder(original_config, ddm_model_config)

    # Config should be unchanged
    assert original_config == config_copy
    assert isinstance(builder, KDEEstimatorBuilder)


def test_factory_with_empty_config(ddm_model_config):
    """Test that factory works with minimal/empty config."""
    generator_config = {}

    builder = create_estimator_builder(generator_config, ddm_model_config)

    assert isinstance(builder, KDEEstimatorBuilder)
    # Should use defaults
    assert builder.displace_t is False


def test_factory_integration_with_different_models(ddm_model_config):
    """Test that factory works with different model configs."""
    generator_config = {"estimator_type": "kde"}

    # Test with DDM
    ddm_builder = create_estimator_builder(generator_config, ddm_model_config)
    assert isinstance(ddm_builder, KDEEstimatorBuilder)

    # Test with Ornstein
    ornstein_config = model_config["ornstein"]
    ornstein_builder = create_estimator_builder(generator_config, ornstein_config)
    assert isinstance(ornstein_builder, KDEEstimatorBuilder)
