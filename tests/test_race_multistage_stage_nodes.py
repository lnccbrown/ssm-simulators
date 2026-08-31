"""Regression tests for multi-stage race simulator node transitions."""

import numpy as np
import pytest

import cssm


def test_euler_step_stops_at_the_next_stage_node():
    """Dynamics after a node must not be applied retroactively to its step."""
    inputs = dict(
        mu_array=np.array([[[0.0, 2.0]]]),
        sigma_array=np.zeros((1, 1, 2)),
        node_array=np.array([[[0.0, 0.5]]]),
        d_array=np.array([[2]], dtype=np.int32),
        upper_intercept_array=np.ones((1, 1, 2)),
        upper_slope_array=np.zeros((1, 1, 2)),
        x0_array=np.zeros((1, 1)),
    )

    out = cssm.race_multistage(
        **inputs, n_samples=1, delta_t=0.6, max_t=2.0, random_state=3
    )

    assert out["choices"][0, 0, 0] == 0
    # The 0.5 node splits the first 0.6 step. The second stage then crosses
    # during [0.5, 1.1], whose midpoint is 0.8.
    np.testing.assert_allclose(out["rts"][0, 0, 0], 0.8, atol=1e-12)


def test_initial_boundary_crossing_has_zero_response_time():
    """An initial crossing must win before the first Euler propagation."""
    out = cssm.race_multistage(
        mu_array=np.zeros((1, 2, 1)),
        sigma_array=np.zeros((1, 2, 1)),
        node_array=np.zeros((1, 2, 1)),
        d_array=np.ones((1, 2), dtype=np.int32),
        upper_intercept_array=np.array([[[1.0], [0.5]]]),
        upper_slope_array=np.zeros((1, 2, 1)),
        x0_array=np.array([[1.0, 0.5]]),
        n_samples=1,
        delta_t=0.1,
        max_t=1.0,
        random_state=3,
    )

    # Both particles cross at time zero; the lowest accumulator index wins.
    assert out["choices"][0, 0, 0] == 0
    np.testing.assert_allclose(out["rts"][0, 0, 0], 0.0, atol=1e-12)


def test_decreasing_active_nodes_are_rejected():
    """The simulator must not take a negative Euler step at a stale node."""
    with pytest.raises(ValueError, match="active stage nodes must be nondecreasing"):
        cssm.race_multistage(
            mu_array=np.zeros((1, 1, 3)),
            sigma_array=np.ones((1, 1, 3)),
            node_array=np.array([[[0.0, 0.5, 0.2]]]),
            d_array=np.array([[3]], dtype=np.int32),
            upper_intercept_array=np.ones((1, 1, 3)),
            upper_slope_array=np.zeros((1, 1, 3)),
            x0_array=np.zeros((1, 1)),
            n_samples=1,
            delta_t=0.1,
            max_t=1.0,
            random_state=3,
        )
