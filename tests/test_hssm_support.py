"""Tests for ssms.hssm_support module."""

import random

import numpy as np
import pytest
from unittest.mock import Mock, patch

from ssms.hssm_support import (
    _create_arg_arrays,
    _extract_size_val,
    _calculate_n_replicas,
    _extract_size,
    _get_seed,
    _prepare_theta_and_shape,
    _reshape_sims_out,
    _validate_size,
    _get_p_outlier,
    _validate_simulator_fun_arg,
    decorate_atomic_simulator,
    hssm_sim_wrapper,
    _build_decorated_simulator,
    get_simulator_fun_internal,
    validate_simulator_fun,
)


class MockHasListParams:
    """Mock class that implements _HasListParams protocol."""

    def __init__(self, list_params):
        self._list_params = list_params


class TestCreateArgArrays:
    """Tests for _create_arg_arrays function."""

    def test_create_arg_arrays_basic(self):
        """Test basic functionality of _create_arg_arrays."""
        cls = MockHasListParams(["a", "b", "c"])
        args = (1, 2, 3, 4)
        result = _create_arg_arrays(cls, args)
        assert result == [np.array(i) for i in (1, 2, 3)]

    def test_create_arg_arrays_fewer_args_than_params(self):
        """Test _create_arg_arrays when args has fewer elements than params."""
        cls = MockHasListParams(["a", "b", "c", "d"])
        args = (1, 2)
        result = _create_arg_arrays(cls, args)
        assert result == [np.array(i) for i in (1, 2)]

    def test_create_arg_arrays_array_inputs(self):
        """Test _create_arg_arrays with array inputs."""
        cls = MockHasListParams(["a", "b"])
        args = ([1, 2, 3], np.array([4, 5, 6]))
        result = _create_arg_arrays(cls, args)

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[1], np.array([4, 5, 6]))


class TestExtractSizeVal:
    """Tests for _extract_size_val function."""

    def test_extract_size_val_tuple(self):
        """Test _extract_size_val with tuple input."""
        assert _extract_size_val((5, 3)) == 5
        assert _extract_size_val((10,)) == 10

    def test_extract_size_val_scalar(self):
        """Test _extract_size_val with scalar input."""
        assert _extract_size_val(7) == 7
        assert _extract_size_val(1) == 1


class TestCalculateNReplicas:
    """Tests for _calculate_n_replicas function."""

    def test_calculate_n_replicas_all_scalar(self):
        """Test _calculate_n_replicas when all args are scalar."""
        result = _calculate_n_replicas(True, 10, 5)
        assert result == 1

    def test_calculate_n_replicas_size_none(self):
        """Test _calculate_n_replicas when size is None."""
        result = _calculate_n_replicas(False, None, 5)
        assert result == 1

    def test_calculate_n_replicas_size_one(self):
        """Test _calculate_n_replicas when size is 1."""
        result = _calculate_n_replicas(False, 1, 5)
        assert result == 1

    def test_calculate_n_replicas_valid_division(self):
        """Test _calculate_n_replicas with valid size division."""
        result = _calculate_n_replicas(False, 20, 5)
        assert result == 4

    def test_calculate_n_replicas_tuple_size(self):
        """Test _calculate_n_replicas with tuple size."""
        result = _calculate_n_replicas(False, (15, 2), 3)
        assert result == 5

    def test_calculate_n_replicas_invalid_size(self):
        """Test _calculate_n_replicas with invalid size that doesn't divide evenly."""
        with pytest.raises(
            ValueError, match="`size` needs to be a multiple of the size of data"
        ):
            _calculate_n_replicas(False, 7, 3)


class TestExtractSize:
    """Tests for _extract_size function."""

    def test_extract_size_from_kwargs(self):
        """Test _extract_size when size is in kwargs."""
        args = (1, 2, 3)
        kwargs = {"size": 10, "other": "value"}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 10
        assert new_args == (1, 2, 3)
        assert new_kwargs == {"other": "value"}

    def test_extract_size_from_args(self):
        """Test _extract_size when size is in args."""
        args = (1, 2, 3, 15)
        kwargs = {"other": "value"}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 15
        assert new_args == (1, 2, 3)
        assert new_kwargs == {"other": "value"}

    def test_extract_size_none_default(self):
        """Test _extract_size when size is None, should default to 1."""
        args = (1, 2, 3, None)
        kwargs = {}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 1
        assert new_args == (1, 2, 3)
        assert new_kwargs == {}


class TestGetSeed:
    """Tests for _get_seed function."""

    def test_get_seed_returns_uint32(self):
        """Test that _get_seed returns a valid uint32 value."""
        rng = np.random.default_rng(42)
        seed = _get_seed(rng)

        assert isinstance(seed, (int, np.integer))
        assert 0 <= seed <= np.iinfo(np.uint32).max

    def test_get_seed_different_values(self):
        """Test that _get_seed returns different values on multiple calls."""
        rng = np.random.default_rng(42)
        seed1 = _get_seed(rng)
        seed2 = _get_seed(rng)

        # They should be different (with very high probability)
        assert seed1 != seed2


class TestPrepareThetaAndShape:
    """Tests for _prepare_theta_and_shape function."""

    def test_prepare_theta_all_scalars(self):
        """Test _prepare_theta_and_shape with all scalar inputs."""
        arg_arrays = [np.array(1.0), np.array(2.0), np.array(3.0)]
        size = 5

        is_all_scalar, theta, max_shape, new_data_size = _prepare_theta_and_shape(
            arg_arrays, size
        )

        assert is_all_scalar is True
        assert theta.shape == (5, 3)
        assert max_shape is None
        assert new_data_size is None
        np.testing.assert_array_equal(theta[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(theta[4], [1.0, 2.0, 3.0])

    def test_prepare_theta_with_arrays(self):
        """Test _prepare_theta_and_shape with array inputs."""
        arg_arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0]),  # Will be broadcast to match largest array
            np.array([6.0, 7.0, 8.0]),
        ]
        size = 10  # Not used when arrays are provided

        is_all_scalar, theta, max_shape, new_data_size = _prepare_theta_and_shape(
            arg_arrays, size
        )

        assert is_all_scalar is False
        assert max_shape == (3,)
        assert new_data_size == 3
        assert theta.shape == (3, 3)
        # Check that the scalar value was broadcast correctly
        np.testing.assert_array_equal(theta[:, 1], [4.0, 4.0, 4.0])

    def test_prepare_theta_multidimensional(self):
        """Test _prepare_theta_and_shape with multidimensional arrays."""
        arg_arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
        size = 10

        is_all_scalar, theta, max_shape, new_data_size = _prepare_theta_and_shape(
            arg_arrays, size
        )

        assert is_all_scalar is False
        assert max_shape == (2, 2)
        assert new_data_size == 2
        assert theta.shape == (4, 2)  # Flattened


class TestReshapeSimsOut:
    """Tests for _reshape_sims_out function."""

    def test_reshape_sims_out_single_replica(self):
        """Test _reshape_sims_out with single replica."""
        max_shape = (3, 4)
        n_replicas = 1
        obs_dim_int = 2

        result = _reshape_sims_out(max_shape, n_replicas, obs_dim_int)

        assert result == (3, 4, 2)

    def test_reshape_sims_out_multiple_replicas(self):
        """Test _reshape_sims_out with multiple replicas."""
        max_shape = (2, 3)
        n_replicas = 5
        obs_dim_int = 2

        result = _reshape_sims_out(max_shape, n_replicas, obs_dim_int)

        assert result == (2, 3, 5, 2)

    def test_reshape_sims_out_1d_shape(self):
        """Test _reshape_sims_out with 1D max_shape."""
        max_shape = (4,)
        n_replicas = 3
        obs_dim_int = 1

        result = _reshape_sims_out(max_shape, n_replicas, obs_dim_int)

        assert result == (4, 3, 1)


class TestValidateSize:
    """Tests for _validate_size function."""

    def test_validate_size_valid_multiple(self):
        """Test _validate_size with valid multiple."""
        # Should not raise an exception
        assert _validate_size(12, 3) is None
        assert _validate_size(10, 5) is None
        assert _validate_size(7, 1) is None

    random.seed(123)
    max_odd = random.randint(9, 15)
    cases = [
        (odd, smaller_even)
        for odd in range(5, max_odd + 1, 2)
        for smaller_even in range(2, odd, 2)
    ]

    @pytest.mark.parametrize("size, data_size", cases)
    def test_validate_size_invalid_multiple(self, size, data_size):
        """Test _validate_size with invalid multiple."""
        match = "`size` needs to be a multiple of the size of data"
        with pytest.raises(ValueError, match=match):
            _validate_size(size, data_size)


class TestGetPOutlier:
    """Tests for _get_p_outlier function."""

    def test_get_p_outlier_present(self):
        """Test _get_p_outlier when p_outlier is present."""
        cls = MockHasListParams(["a", "b", "p_outlier"])
        arg_arrays = [np.array([1, 2]), np.array([3, 4]), np.array([0.1, 0.2])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        np.testing.assert_array_equal(p_outlier, np.array([0.1, 0.2]))
        assert len(new_arg_arrays) == 2
        np.testing.assert_array_equal(new_arg_arrays[0], np.array([1, 2]))
        np.testing.assert_array_equal(new_arg_arrays[1], np.array([3, 4]))

    def test_get_p_outlier_not_present(self):
        """Test _get_p_outlier when p_outlier is not present."""
        cls = MockHasListParams(["a", "b", "c"])
        arg_arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        assert p_outlier is None
        assert len(new_arg_arrays) == 3
        assert new_arg_arrays is arg_arrays  # Should be the same object

    def test_get_p_outlier_empty_params(self):
        """Test _get_p_outlier when _list_params is empty."""
        cls = MockHasListParams([])
        arg_arrays = [np.array([1, 2])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        assert p_outlier is None
        assert new_arg_arrays is arg_arrays


class TestValidateSimulatorFunArg:
    """Tests for _validate_simulator_fun_arg function."""

    def test_validate_simulator_fun_arg_string(self):
        """Test _validate_simulator_fun_arg with string input."""
        # Should not raise an exception
        assert _validate_simulator_fun_arg("ddm") is None

    def test_validate_simulator_fun_arg_callable(self):
        """Test _validate_simulator_fun_arg with callable input."""

        assert _validate_simulator_fun_arg(lambda: None) is None

    bad_simulator_types = [123, [1, 2, 3], None]

    @pytest.mark.parametrize("bad_input", bad_simulator_types)
    def test_validate_simulator_fun_arg_invalid(self, bad_input):
        """Test _validate_simulator_fun_arg with invalid input."""
        match = "`simulator_fun` must be a string or a callable"
        with pytest.raises(ValueError, match=match):
            _validate_simulator_fun_arg(bad_input)


class TestDecorateAtomicSimulator:
    """Tests for decorate_atomic_simulator function."""

    def test_decorate_atomic_simulator_default_params(self):
        """Test decorate_atomic_simulator with default parameters."""

        @decorate_atomic_simulator("test_model")
        def mock_simulator():
            return "test"

        assert mock_simulator.model_name == "test_model"
        assert mock_simulator.choices == [-1, 1]
        assert mock_simulator.obs_dim == 2

    def test_decorate_atomic_simulator_custom_params(self):
        """Test decorate_atomic_simulator with custom parameters."""

        @decorate_atomic_simulator("custom_model", choices=[0, 1, 2], obs_dim=3)
        def mock_simulator():
            return "test"

        assert mock_simulator.model_name == "custom_model"
        assert mock_simulator.choices == [0, 1, 2]
        assert mock_simulator.obs_dim == 3

    def test_decorate_atomic_simulator_preserves_function(self):
        """Test that decorate_atomic_simulator preserves the original function."""

        @decorate_atomic_simulator("test_model")
        def mock_simulator(x, y):
            return x + y

        assert mock_simulator(3, 4) == 7
        assert mock_simulator.model_name == "test_model"


class TestSsmsSimWrapper:
    """Tests for hssm_sim_wrapper function."""

    def test_hssm_sim_wrapper_basic(self):
        """Test basic functionality of hssm_sim_wrapper."""

        def mock_simulator_fun(theta, model, n_samples, random_state, **kwargs):
            return {"rts": np.array([1.0, 2.0]), "choices": np.array([0, 1])}

        theta = np.array([[1, 2, 3]])
        result = hssm_sim_wrapper(
            simulator_fun=mock_simulator_fun,
            theta=theta,
            model="ddm",
            n_replicas=1,
            random_state=42,
        )

        expected = np.array([[1.0, 0], [2.0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_hssm_sim_wrapper_with_kwargs(self):
        """Test hssm_sim_wrapper passes through kwargs."""

        def mock_simulator_fun(
            theta, model, n_samples, random_state, custom_param=None, **kwargs
        ):
            assert custom_param == "test_value"
            return {"rts": np.array([1.5]), "choices": np.array([1])}

        theta = np.array([[1, 2]])
        result = hssm_sim_wrapper(
            simulator_fun=mock_simulator_fun,
            theta=theta,
            model="ddm",
            n_replicas=1,
            random_state=42,
            custom_param="test_value",
        )

        expected = np.array([1.5, 1])
        np.testing.assert_array_equal(result, expected)


class TestBuildDecoratedSimulator:
    """Tests for _build_decorated_simulator function."""

    @patch("ssms.hssm_support.simulator")
    def test_build_decorated_simulator(self, mock_simulator):
        """Test _build_decorated_simulator creates properly decorated function."""
        result = _build_decorated_simulator("ddm", [0, 1])

        # Check that the result is callable
        assert callable(result)

        # Check that it has the required attributes
        assert hasattr(result, "model_name")
        assert hasattr(result, "choices")
        assert hasattr(result, "obs_dim")

        assert result.model_name == "ddm"
        assert result.choices == [0, 1]
        assert result.obs_dim == 2


class TestGetSimulatorFunInternal:
    """Tests for get_simulator_fun_internal function."""

    def test_get_simulator_fun_internal_callable(self):
        """Test get_simulator_fun_internal with callable input."""

        def mock_simulator():
            pass

        result = get_simulator_fun_internal(mock_simulator)
        assert result is mock_simulator

    @patch("ssms.hssm_support.ssms_model_config")
    @patch("ssms.hssm_support._build_decorated_simulator")
    def test_get_simulator_fun_internal_string_known_model(
        self, mock_build, mock_config
    ):
        """Test get_simulator_fun_internal with known model string."""
        mock_config.__contains__ = Mock(return_value=True)
        mock_config.get = Mock(return_value={"choices": [0, 1]})
        mock_build.return_value = Mock()

        result = get_simulator_fun_internal("ddm")

        mock_build.assert_called_once_with(model_name="ddm", choices=[0, 1])
        assert result == mock_build.return_value

    @patch("ssms.hssm_support.ssms_model_config")
    @patch("ssms.hssm_support._build_decorated_simulator")
    @patch("ssms.hssm_support._logger")
    def test_get_simulator_fun_internal_string_unknown_model(
        self, mock_logger, mock_build, mock_config
    ):
        """Test get_simulator_fun_internal with unknown model string."""
        mock_config.__contains__ = Mock(return_value=False)
        mock_config.get = Mock(return_value={"choices": [0, 1, 2]})
        mock_build.return_value = Mock()

        get_simulator_fun_internal("unknown_model")

        mock_logger.warning.assert_called_once()
        mock_build.assert_called_once_with(
            model_name="unknown_model", choices=[0, 1, 2]
        )

    def test_get_simulator_fun_internal_invalid_type(self):
        """Test get_simulator_fun_internal with invalid type."""
        match = "`simulator_fun` must be a string or a callable"
        with pytest.raises(ValueError, match=match):
            get_simulator_fun_internal(123)


class TestValidateSimulatorFun:
    """Tests for validate_simulator_fun function."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a valid mock simulator for testing."""
        mock = Mock()
        mock.model_name = "ddm"
        mock.choices = [0, 1]
        mock.obs_dim = 2
        return mock

    def _missing_attr_msg(self, attribute):
        """Helper to generate missing attribute error messages."""
        return f"The simulator function must have a `{attribute}` attribute"

    def test_validate_simulator_fun_valid(self, mock_simulator):
        """Test validate_simulator_fun with valid simulator function."""
        model_name, choices, obs_dim_int = validate_simulator_fun(mock_simulator)

        assert model_name == "ddm"
        assert choices == [0, 1]
        assert obs_dim_int == 2

    def test_validate_simulator_fun_missing_model_name(self, mock_simulator):
        """Test validate_simulator_fun with missing model_name."""
        del mock_simulator.model_name

        with pytest.raises(ValueError, match=self._missing_attr_msg("model_name")):
            validate_simulator_fun(mock_simulator)

    def test_validate_simulator_fun_missing_choices(self, mock_simulator):
        """Test validate_simulator_fun with missing choices."""
        del mock_simulator.choices

        with pytest.raises(ValueError, match=self._missing_attr_msg("choices")):
            validate_simulator_fun(mock_simulator)

    def test_validate_simulator_fun_missing_obs_dim(self, mock_simulator):
        """Test validate_simulator_fun with missing obs_dim."""
        del mock_simulator.obs_dim

        with pytest.raises(ValueError, match=self._missing_attr_msg("obs_dim")):
            validate_simulator_fun(mock_simulator)

    def test_validate_simulator_fun_invalid_obs_dim_type(self, mock_simulator):
        """Test validate_simulator_fun with invalid obs_dim type."""
        mock_simulator.obs_dim = "2"  # String instead of int

        with pytest.raises(
            ValueError, match="The obs_dim attribute must be an integer"
        ):
            validate_simulator_fun(mock_simulator)
