"""Tests for the Efficient-FPT-style multi-stage race simulator."""

import numpy as np
import pytest

import cssm


def _inputs(n_trials=1, n_accumulators=2, n_stages=1):
    """Return a valid, zero-drift/noise race-array fixture."""
    shape = (n_trials, n_accumulators, n_stages)
    return dict(
        mu_array=np.zeros(shape),
        sigma_array=np.zeros(shape),
        node_array=np.zeros(shape),
        d_array=np.full((n_trials, n_accumulators), n_stages, dtype=np.int32),
        upper_intercept_array=np.ones(shape),
        upper_slope_array=np.zeros(shape),
        x0_array=np.zeros((n_trials, n_accumulators)),
    )


def test_one_accumulator_deterministic_crossing():
    inputs = _inputs(n_accumulators=1)
    inputs["mu_array"][:] = 1.0
    out = cssm.race_multistage(
        **inputs, n_samples=3, delta_t=0.1, max_t=2.0, random_state=1
    )

    assert out["rts"].shape == (3, 1, 1)
    assert np.all(out["choices"] == 0)
    # Crossing is detected at t=1 and follows the Efficient-FPT midpoint rule.
    np.testing.assert_allclose(out["rts"].reshape(-1), 0.95, atol=1e-6)
    assert out["metadata"]["possible_choices"] == [0]


def test_fastest_accumulator_wins_the_race():
    inputs = _inputs(n_accumulators=2)
    inputs["mu_array"][0, :, 0] = [1.0, 2.0]
    out = cssm.race_multistage(
        **inputs, n_samples=1, delta_t=0.1, max_t=2.0, random_state=2
    )

    assert out["choices"][0, 0, 0] == 1
    np.testing.assert_allclose(out["rts"][0, 0, 0], 0.45, atol=1e-6)


def test_native_stage_switches_are_used():
    inputs = _inputs(n_accumulators=1, n_stages=2)
    inputs["mu_array"][0, 0] = [0.0, 2.0]
    inputs["node_array"][0, 0] = [0.0, 0.5]
    out = cssm.race_multistage(
        **inputs, n_samples=1, delta_t=0.1, max_t=2.0, random_state=3
    )

    assert out["choices"][0, 0, 0] == 0
    np.testing.assert_allclose(out["rts"][0, 0, 0], 0.95, atol=1e-6)


def test_omission_uses_ssms_sentinel():
    inputs = _inputs(n_accumulators=2)
    inputs["x0_array"][:] = [0.2, -0.3]
    out = cssm.race_multistage(**inputs, n_samples=2, max_t=0.0, random_state=4)
    assert np.all(out["rts"] == -999.0)
    np.testing.assert_allclose(
        out["metadata"]["x_final"],
        np.broadcast_to(inputs["x0_array"], (2, 1, 2)),
    )


def test_seeded_output_is_thread_count_independent():
    inputs = _inputs(n_trials=4, n_accumulators=2)
    inputs["mu_array"][:] = 0.3
    inputs["sigma_array"][:] = 1.0
    common = dict(**inputs, n_samples=10, delta_t=0.01, max_t=1.0, random_state=77)
    one = cssm.race_multistage(**common, n_threads=1)
    four = cssm.race_multistage(**common, n_threads=4)
    np.testing.assert_array_equal(one["rts"], four["rts"])
    np.testing.assert_array_equal(one["choices"], four["choices"])


def test_negative_nondecision_time_is_rejected():
    """Negative nondecision_time must be rejected."""
    with pytest.raises(ValueError, match="nondecision_time must be non-negative"):
        cssm.race_multistage(
            mu_array=np.ones((1, 1, 1)),
            sigma_array=np.ones((1, 1, 1)),
            node_array=np.zeros((1, 1, 1)),
            d_array=np.ones((1, 1), dtype=np.int32),
            upper_intercept_array=np.ones((1, 1, 1)),
            upper_slope_array=np.zeros((1, 1, 1)),
            x0_array=np.zeros((1, 1)),
            nondecision_time=-0.5,
            n_samples=1,
            delta_t=0.1,
            max_t=1.0,
            random_state=3,
        )


def test_negative_deadline_is_rejected():
    """Negative deadline must be rejected."""
    with pytest.raises(ValueError, match="deadline must be non-negative"):
        cssm.race_multistage(
            mu_array=np.ones((1, 1, 1)),
            sigma_array=np.ones((1, 1, 1)),
            node_array=np.zeros((1, 1, 1)),
            d_array=np.ones((1, 1), dtype=np.int32),
            upper_intercept_array=np.ones((1, 1, 1)),
            upper_slope_array=np.zeros((1, 1, 1)),
            x0_array=np.zeros((1, 1)),
            deadline=-1.0,
            n_samples=1,
            delta_t=0.1,
            max_t=1.0,
            random_state=3,
        )


def test_deadline_less_than_nondecision_time_is_rejected():
    """deadline must be >= nondecision_time."""
    with pytest.raises(ValueError, match="deadline must be >= nondecision_time"):
        cssm.race_multistage(
            mu_array=np.ones((1, 1, 1)),
            sigma_array=np.ones((1, 1, 1)),
            node_array=np.zeros((1, 1, 1)),
            d_array=np.ones((1, 1), dtype=np.int32),
            upper_intercept_array=np.ones((1, 1, 1)),
            upper_slope_array=np.zeros((1, 1, 1)),
            x0_array=np.zeros((1, 1)),
            nondecision_time=0.5,
            deadline=0.2,
            n_samples=1,
            delta_t=0.1,
            max_t=1.0,
            random_state=3,
        )
