"""Tests for boundary_functions module."""

import numpy as np

from ssms.basic_simulators import boundary_functions


class TestConstantBoundary:
    """Test constant boundary function."""

    def test_constant_scalar(self):
        """Test constant boundary with scalar input."""
        result = boundary_functions.constant(t=0, a=1.5)
        assert result == 1.5

    def test_constant_array(self):
        """Test constant boundary with array input."""
        t = np.array([0, 1, 2, 3])
        result = boundary_functions.constant(t=t, a=2.0)
        assert isinstance(result, np.ndarray)
        assert np.all(result == 2.0)
        assert result.shape == t.shape

    def test_constant_default_values(self):
        """Test constant boundary with default parameter values."""
        result = boundary_functions.constant()
        assert result == 1.0


class TestAngleBoundary:
    """Test angle (linear collapsing) boundary function."""

    def test_angle_scalar(self):
        """Test angle boundary with scalar input."""
        result = boundary_functions.angle(t=1, a=1.0, theta=0.0)
        assert isinstance(result, (float, np.ndarray))
        # At theta=0, should be approximately constant (tan(0)=0)
        assert np.isclose(result, 1.0)

    def test_angle_array(self):
        """Test angle boundary with array input."""
        t = np.array([0, 1, 2, 3])
        result = boundary_functions.angle(t=t, a=2.0, theta=0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        # Boundary should decrease over time for positive theta
        assert np.all(np.diff(result) < 0)

    def test_angle_collapse(self):
        """Test that angle boundary collapses linearly."""
        t = np.linspace(0, 5, 10)
        a = 2.0
        theta = 1.0
        result = boundary_functions.angle(t=t, a=a, theta=theta)

        # Check that it follows: a + t * (-sin(theta)/cos(theta))
        expected = a + t * (-np.sin(theta) / np.cos(theta))
        assert np.allclose(result, expected)


class TestGeneralizedLogisticBoundary:
    """Test generalized logistic boundary function."""

    def test_generalized_logistic_scalar(self):
        """Test generalized logistic boundary with scalar input."""
        result = boundary_functions.generalized_logistic(
            t=1, a=1.0, B=2.0, M=3.0, v=0.5
        )
        assert isinstance(result, (float, np.ndarray))
        # Should be close to a at t=1 (before inflection at M=3)
        assert result >= 1.0

    def test_generalized_logistic_array(self):
        """Test generalized logistic boundary with array input."""
        t = np.array([0, 1, 2, 3, 4, 5])
        result = boundary_functions.generalized_logistic(
            t=t, a=1.0, B=2.0, M=3.0, v=0.5
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        # All values should be >= a (since it's an increasing offset)
        assert np.all(result >= 1.0)

    def test_generalized_logistic_shape(self):
        """Test that generalized logistic has expected shape."""
        t = np.linspace(0, 10, 100)
        result = boundary_functions.generalized_logistic(
            t=t, a=1.0, B=1.0, M=5.0, v=0.5
        )

        # Generalized logistic is a collapsing boundary (decreasing over time)
        # The offset is: 1 - (1 / (1 + exp(-B*(t-M)))^(1/v))
        # which is negative and becomes more negative over time
        assert np.all(np.diff(result) <= 0)


class TestWeibullCDFBoundary:
    """Test Weibull CDF (decay) boundary function."""

    def test_weibull_cdf_scalar(self):
        """Test Weibull boundary with scalar input."""
        result = boundary_functions.weibull_cdf(t=0, a=1.5, alpha=1.0, beta=1.0)
        assert isinstance(result, (float, np.ndarray))
        # At t=0, should equal a
        assert np.isclose(result, 1.5)

    def test_weibull_cdf_array(self):
        """Test Weibull boundary with array input."""
        t = np.array([0, 1, 2, 3, 4])
        result = boundary_functions.weibull_cdf(t=t, a=2.0, alpha=1.0, beta=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape

    def test_weibull_cdf_decay(self):
        """Test that Weibull boundary decays over time."""
        t = np.linspace(0, 5, 10)
        a = 2.0
        alpha = 1.0
        beta = 1.0
        result = boundary_functions.weibull_cdf(t=t, a=a, alpha=alpha, beta=beta)

        # Should decay: a * exp(-(t/beta)^alpha)
        expected = a * np.exp(-np.power(t / beta, alpha))
        assert np.allclose(result, expected)

        # Should be monotonically decreasing
        assert np.all(np.diff(result) <= 0)

    def test_weibull_cdf_at_zero(self):
        """Test that Weibull boundary equals a at t=0."""
        result = boundary_functions.weibull_cdf(t=0, a=3.0, alpha=2.0, beta=1.5)
        assert np.isclose(result, 3.0)


class TestConflictGammaBoundary:
    """Test conflict gamma boundary function."""

    def test_conflict_gamma_default(self):
        """Test conflict gamma boundary with default parameters."""
        result = boundary_functions.conflict_gamma()
        assert isinstance(result, np.ndarray)
        # Default t is np.arange(0, 20, 0.1)
        assert len(result) == 200

    def test_conflict_gamma_custom_t(self):
        """Test conflict gamma boundary with custom time array."""
        t = np.linspace(0, 10, 50)
        result = boundary_functions.conflict_gamma(
            t=t, a=1.5, theta=0.5, scale=1.0, alphaGamma=1.01, scaleGamma=0.3
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape

    def test_conflict_gamma_scalar(self):
        """Test conflict gamma boundary with scalar input."""
        result = boundary_functions.conflict_gamma(
            t=1.0, a=1.0, theta=0.5, scale=1.0, alphaGamma=1.01, scaleGamma=0.3
        )
        # Should work with scalar and return array-like
        assert isinstance(result, (float, np.ndarray))

    def test_conflict_gamma_components(self):
        """Test that conflict gamma has expected components."""
        t = np.linspace(0, 5, 100)
        a = 2.0
        theta = 0.3
        scale = 1.0
        alphaGamma = 1.5
        scaleGamma = 0.5

        result = boundary_functions.conflict_gamma(
            t=t,
            a=a,
            theta=theta,
            scale=scale,
            alphaGamma=alphaGamma,
            scaleGamma=scaleGamma,
        )

        # Should have a gamma bump (some values > a + linear_collapse)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(t)

        # At t=0, should be close to a (gamma pdf at 0 is small)
        assert result[0] >= a - 0.5  # Allow some tolerance

    def test_conflict_gamma_shape(self):
        """Test that conflict gamma produces reasonable boundary shape."""
        t = np.linspace(0, 10, 200)
        result = boundary_functions.conflict_gamma(t=t, a=1.5)

        # Should have a bump early (gamma peak)
        # Then decay due to angle collapse
        assert np.max(result) > result[0]  # Bump exists
        assert result[-1] < result[0]  # Overall collapse


class TestBoundaryFunctionTypes:
    """Test that boundary functions have correct type annotations."""

    def test_boundary_function_type_alias(self):
        """Test that BoundaryFunction type alias is defined."""
        assert hasattr(boundary_functions, "BoundaryFunction")

    def test_all_functions_are_callable(self):
        """Test that all boundary functions are callable."""
        funcs = [
            boundary_functions.constant,
            boundary_functions.angle,
            boundary_functions.generalized_logistic,
            boundary_functions.weibull_cdf,
            boundary_functions.conflict_gamma,
        ]

        for func in funcs:
            assert callable(func)
