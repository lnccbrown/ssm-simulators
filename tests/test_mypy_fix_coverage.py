"""Tests covering code paths introduced by mypy type fixes.

These tests ensure the type-narrowing branches and signature changes
from the mypy alignment work are exercised by the test suite.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import bin_simulator_output_pointwise
from ssms.support_utils.kde_class import LogKDE


class TestBinSimulatorOutputPointwise:
    """Tests for bin_simulator_output_pointwise type fix coverage."""

    def test_none_input_raises(self):
        """Test that passing None raises ValueError."""
        with pytest.raises(ValueError, match="out is not supplied"):
            bin_simulator_output_pointwise(out=None)

    def test_dict_input(self):
        """Test that the function works with dict input (new signature)."""
        out = {
            0: np.array([[0.5], [0.3], [0.8], [1.2], [0.1]]),
            1: np.array([[-1], [1], [-1], [1], [-1]]),
            "metadata": {"max_t": 2.0},
        }
        result = bin_simulator_output_pointwise(out=out, bin_dt=0.5)
        assert result.dtype == np.int32
        assert result.shape[0] == 5
        assert result.shape[1] == 2

    def test_dict_input_with_nbins(self):
        """Test with explicit nbins parameter."""
        out = {
            0: np.array([[0.5], [1.5]]),
            1: np.array([[-1], [1]]),
            "metadata": {"max_t": 2.0},
        }
        result = bin_simulator_output_pointwise(out=out, nbins=4)
        assert result.dtype == np.int32
        assert result.shape == (2, 2)


class TestKdeClassAlternateChoiceP:
    """Tests for LogKDE.kde_sample alternate_choice_p type branches."""

    @pytest.fixture
    def simple_kde(self):
        """Create a LogKDE with minimal data for testing."""
        rts = np.abs(np.random.default_rng(42).normal(0.5, 0.2, size=(200, 1)))
        choices = np.random.default_rng(42).choice([-1, 1], size=(200, 1))
        data = {
            "rts": rts,
            "choices": choices,
            "metadata": {
                "possible_choices": [-1, 1],
            },
        }
        return LogKDE(data)

    def test_kde_sample_with_ndarray_choice_p(self, simple_kde):
        """Test kde_sample with np.ndarray alternate_choice_p (new branch)."""
        result = simple_kde.kde_sample(
            n_samples=100,
            use_empirical_choice_p=False,
            alternate_choice_p=np.array([0.5, 0.5]),
            random_state=42,
        )
        assert "rts" in result
        assert "choices" in result

    def test_kde_sample_with_list_choice_p(self, simple_kde):
        """Test kde_sample with list alternate_choice_p (else branch)."""
        result = simple_kde.kde_sample(
            n_samples=100,
            use_empirical_choice_p=False,
            alternate_choice_p=[0.5, 0.5],
            random_state=42,
        )
        assert "rts" in result
        assert "choices" in result

    def test_kde_sample_with_float_choice_p_validates_length(self, simple_kde):
        """Test that a single float raises ValueError for a 2-choice model.

        A single float wraps to [0.5], length 1, which doesn't match 2 choices.
        """
        with pytest.raises(ValueError, match="same length as the number of choices"):
            simple_kde.kde_sample(
                n_samples=100,
                use_empirical_choice_p=False,
                alternate_choice_p=0.5,
                random_state=42,
            )
