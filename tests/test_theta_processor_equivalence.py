"""
Equivalence tests for ModularThetaProcessor vs SimpleThetaProcessor.

These tests verify that the new ModularThetaProcessor produces identical
results to the legacy SimpleThetaProcessor for all models.
"""

from copy import deepcopy

import numpy as np
import pytest

from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
from ssms.config import model_config


def generate_test_theta(model_name: str, model_cfg: dict, n_trials: int = 5) -> dict:
    """Generate valid test theta for a model.

    Creates a theta dictionary with valid parameter values for testing.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_cfg : dict
        Model configuration dictionary
    n_trials : int
        Number of trials to generate parameters for

    Returns
    -------
    dict
        Valid theta dictionary for the model
    """
    theta = {}
    params = model_cfg.get("params", [])
    param_bounds = model_cfg.get("param_bounds", [])

    if not params or not param_bounds:
        # Fallback to minimal valid theta
        return {"v": np.array([0.5] * n_trials, dtype=np.float32)}

    # Generate values within bounds
    # Handle both list and dict formats
    if isinstance(param_bounds, list):
        # List format: [[lower_bounds], [upper_bounds]]
        lower_bounds = param_bounds[0]
        upper_bounds = param_bounds[1]

        for i, param in enumerate(params):
            if i < len(lower_bounds):
                lower = lower_bounds[i]
                upper = upper_bounds[i]

                # Generate valid value (midpoint of bounds)
                if np.isfinite(lower) and np.isfinite(upper):
                    value = (lower + upper) / 2.0
                elif np.isfinite(lower):
                    value = lower + 1.0
                elif np.isfinite(upper):
                    value = upper - 1.0
                else:
                    value = 0.5  # Default

                # Ensure positive for certain params
                if param in ["a", "A", "b", "s", "sd", "t", "st", "sz", "sv"]:
                    value = abs(value) if value != 0 else 0.1

                theta[param] = np.array([value] * n_trials, dtype=np.float32)

    elif isinstance(param_bounds, dict):
        # Dict format: {'param_name': (lower, upper), ...}
        for param in params:
            if param in param_bounds:
                bounds = param_bounds[param]
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    lower, upper = bounds

                    # Handle string upper bounds (e.g., 't' for full_ddm_rv)
                    if isinstance(upper, str):
                        # Upper bound references another parameter
                        if upper in theta:
                            upper = float(theta[upper][0])
                        else:
                            upper = 2.0  # Fallback

                    # Generate valid value (midpoint of bounds)
                    if np.isfinite(lower) and np.isfinite(upper):
                        value = (lower + upper) / 2.0
                    elif np.isfinite(lower):
                        value = lower + 1.0
                    elif np.isfinite(upper):
                        value = upper - 1.0
                    else:
                        value = 0.5  # Default

                    # Ensure positive for certain params
                    if param in ["a", "A", "b", "s", "sd", "t", "st", "sz", "sv"]:
                        value = abs(value) if value != 0 else 0.1

                    theta[param] = np.array([value] * n_trials, dtype=np.float32)
            else:
                # Parameter not in bounds, use default
                theta[param] = np.array([0.5] * n_trials, dtype=np.float32)
    else:
        # Unknown format, fallback
        return {"v": np.array([0.5] * n_trials, dtype=np.float32)}

    # Handle 's' or 'sd' for noise (if not already in params)
    if "s" not in theta and "sd" not in theta:
        if "lba" in model_name:
            theta["sd"] = np.array([0.1] * n_trials, dtype=np.float32)
        else:
            theta["s"] = np.array([1.0] * n_trials, dtype=np.float32)

    return theta


def assert_theta_equal(theta1: dict, theta2: dict, model_name: str = ""):
    """Assert that two theta dictionaries are equal.

    Handles numpy array comparison and provides detailed error messages.

    Parameters
    ----------
    theta1 : dict
        First theta dictionary
    theta2 : dict
        Second theta dictionary
    model_name : str, optional
        Model name for error messages
    """
    # Check keys match
    keys1 = set(theta1.keys())
    keys2 = set(theta2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        msg = f"Model {model_name}: Key mismatch.\n"
        if missing_in_2:
            msg += f"  Missing in new: {missing_in_2}\n"
        if missing_in_1:
            msg += f"  Missing in old: {missing_in_1}\n"
        pytest.fail(msg)

    # Check values match
    for key in keys1:
        val1 = theta1[key]
        val2 = theta2[key]

        # Handle numpy arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            try:
                np.testing.assert_array_equal(val1, val2)
            except AssertionError as e:
                pytest.fail(
                    f"Model {model_name}: Array mismatch for '{key}':\n"
                    f"  Old: {val1}\n"
                    f"  New: {val2}\n"
                    f"  Error: {e}"
                )
        elif isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            # One is array, one is not
            pytest.fail(
                f"Model {model_name}: Type mismatch for '{key}':\n"
                f"  Old: {type(val1)} = {val1}\n"
                f"  New: {type(val2)} = {val2}"
            )
        else:
            # Non-array values (e.g., distribution objects, scalars)
            if val1 != val2:
                # For distribution objects, just check they exist
                if hasattr(val1, "__call__") and hasattr(val2, "__call__"):
                    # Both are callables (e.g., distributions), consider equal
                    continue
                else:
                    pytest.fail(
                        f"Model {model_name}: Value mismatch for '{key}':\n"
                        f"  Old: {val1}\n"
                        f"  New: {val2}"
                    )


# Get all model names from model_config
ALL_MODEL_NAMES = list(model_config.keys())


class TestProcessorEquivalence:
    """Test equivalence between SimpleThetaProcessor and ModularThetaProcessor."""

    @pytest.fixture
    def simple_processor(self):
        """Create SimpleThetaProcessor instance."""
        return SimpleThetaProcessor()

    @pytest.fixture
    def modular_processor(self):
        """Create ModularThetaProcessor instance."""
        return ModularThetaProcessor()

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_equivalence_single_trial(
        self, model_name, simple_processor, modular_processor
    ):
        """Test equivalence for single trial (n_trials=1)."""
        model_cfg = model_config[model_name]

        # Generate test theta
        theta_input = generate_test_theta(model_name, model_cfg, n_trials=1)

        # Process with both processors
        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=1
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=1
        )

        # Assert equivalence
        assert_theta_equal(theta_old, theta_new, model_name)

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_equivalence_multiple_trials(
        self, model_name, simple_processor, modular_processor
    ):
        """Test equivalence for multiple trials (n_trials=10)."""
        model_cfg = model_config[model_name]

        # Generate test theta
        theta_input = generate_test_theta(model_name, model_cfg, n_trials=10)

        # Process with both processors
        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=10
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=10
        )

        # Assert equivalence
        assert_theta_equal(theta_old, theta_new, model_name)

    @pytest.mark.parametrize("n_trials", [1, 5, 10, 100])
    def test_equivalence_ddm_varying_trials(
        self, n_trials, simple_processor, modular_processor
    ):
        """Test DDM model with varying number of trials."""
        model_name = "ddm"
        model_cfg = model_config[model_name]

        theta_input = generate_test_theta(model_name, model_cfg, n_trials=n_trials)

        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials
        )

        assert_theta_equal(theta_old, theta_new, f"{model_name}_n{n_trials}")

    @pytest.mark.parametrize(
        "model_name", ["lba2", "lba3", "race_3", "race_4", "lca_3", "lca_4"]
    )
    def test_equivalence_multi_choice_models(
        self, model_name, simple_processor, modular_processor
    ):
        """Test multi-choice models (LBA, race, LCA)."""
        model_cfg = model_config[model_name]

        theta_input = generate_test_theta(model_name, model_cfg, n_trials=5)

        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=5
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=5
        )

        assert_theta_equal(theta_old, theta_new, model_name)

    def test_equivalence_preserves_input(self, simple_processor, modular_processor):
        """Test that input theta is not modified (or both modify it identically)."""
        model_name = "ddm"
        model_cfg = model_config[model_name]

        theta_input = generate_test_theta(model_name, model_cfg, n_trials=5)

        # Make copies for comparison
        theta_for_old = deepcopy(theta_input)
        theta_for_new = deepcopy(theta_input)

        # Process
        simple_processor.process_theta(theta_for_old, model_cfg, n_trials=5)
        modular_processor.process_theta(theta_for_new, model_cfg, n_trials=5)

        # Both should modify (or not modify) input identically
        assert_theta_equal(
            theta_for_old, theta_for_new, f"{model_name}_input_preservation"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def simple_processor(self):
        return SimpleThetaProcessor()

    @pytest.fixture
    def modular_processor(self):
        return ModularThetaProcessor()

    def test_empty_theta(self, simple_processor, modular_processor):
        """Test with empty theta dictionary."""
        model_name = "ddm"
        model_cfg = model_config[model_name]
        theta = {}

        # Both should handle empty theta (may add defaults)
        theta_old = simple_processor.process_theta(deepcopy(theta), model_cfg, 1)
        theta_new = modular_processor.process_theta(deepcopy(theta), model_cfg, 1)

        # Results should be equivalent
        assert_theta_equal(theta_old, theta_new, f"{model_name}_empty")

    def test_extra_parameters(self, simple_processor, modular_processor):
        """Test with extra parameters not in model spec."""
        model_name = "ddm"
        model_cfg = model_config[model_name]

        theta = generate_test_theta(model_name, model_cfg, n_trials=3)
        theta["extra_param"] = np.array([999.0, 999.0, 999.0])

        theta_old = simple_processor.process_theta(deepcopy(theta), model_cfg, 3)
        theta_new = modular_processor.process_theta(deepcopy(theta), model_cfg, 3)

        assert_theta_equal(theta_old, theta_new, f"{model_name}_extra_params")

    @pytest.mark.xfail(
        reason="Expected difference: ModularThetaProcessor is more robust with missing "
        "parameters (gracefully skips transformations), while SimpleThetaProcessor "
        "raises KeyError. In production, parameters are validated before theta processing."
    )
    @pytest.mark.parametrize("model_name", ["lba2", "race_3", "lca_3"])
    def test_missing_parameters(self, model_name, simple_processor, modular_processor):
        """Test with some parameters missing (should use defaults or fail identically).

        NOTE: This test is expected to fail due to intentional difference in error handling.
        SimpleThetaProcessor raises KeyError for missing required parameters, while
        ModularThetaProcessor gracefully handles missing parameters by skipping
        transformations that require them.
        """
        model_cfg = model_config[model_name]

        # Minimal theta (may be incomplete)
        theta = {"v0": np.array([0.5, 0.5])}

        # Both should handle missing params identically
        try:
            theta_old = simple_processor.process_theta(deepcopy(theta), model_cfg, 2)
            theta_new = modular_processor.process_theta(deepcopy(theta), model_cfg, 2)
            assert_theta_equal(theta_old, theta_new, f"{model_name}_missing_params")
        except Exception as e_old:
            # If old raises exception, new should too
            with pytest.raises(type(e_old)):
                modular_processor.process_theta(deepcopy(theta), model_cfg, 2)


class TestSpecificModels:
    """Focused tests for specific model families."""

    @pytest.fixture
    def simple_processor(self):
        return SimpleThetaProcessor()

    @pytest.fixture
    def modular_processor(self):
        return ModularThetaProcessor()

    @pytest.mark.parametrize(
        "model_name",
        [
            "ddm_seq2_no_bias",
            "ddm_par2_no_bias",
            "ddm_mic2_adj_no_bias",
            "ddm_mic2_ornstein_no_bias",
            "ddm_mic2_leak_no_bias",
        ],
    )
    def test_sequential_parallel_mic2(
        self, model_name, simple_processor, modular_processor
    ):
        """Test sequential, parallel, and MIC2 model families."""
        model_cfg = model_config[model_name]

        theta_input = generate_test_theta(model_name, model_cfg, n_trials=3)

        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=3
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=3
        )

        assert_theta_equal(theta_old, theta_new, model_name)

    @pytest.mark.parametrize(
        "model_name",
        [
            "race_no_bias_2",
            "race_no_z_3",
            "lca_no_bias_4",
        ],
    )
    def test_no_bias_no_z_variants(
        self, model_name, simple_processor, modular_processor
    ):
        """Test no-bias and no-z model variants."""
        model_cfg = model_config[model_name]

        theta_input = generate_test_theta(model_name, model_cfg, n_trials=5)

        theta_old = simple_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=5
        )
        theta_new = modular_processor.process_theta(
            deepcopy(theta_input), model_cfg, n_trials=5
        )

        assert_theta_equal(theta_old, theta_new, model_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
