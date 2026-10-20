"""Regression tests: stage-split Euler steps must still reach the horizon (#383).

Every path here is deterministic (unit drift, zero noise), so a particle that
starts at zero sits at position ``t`` at time ``t``. A path that never crosses
must therefore end exactly at ``max_t``, and a boundary below ``max_t`` must be
crossed before the horizon whatever the stage partition is.

``race_multistage`` takes ``delta_t`` as a C ``float``; most cases use
``delta_t=0.125`` so the nominal grid is exact in single precision. The
reproducer from the issue uses ``delta_t=0.1`` and is checked to ``1e-6``.
"""

import numpy as np
import pytest

import cssm
from cssm.race_multistage_models import _simulate_race_multistage

OMISSION = -999.0
NO_HIT = 10.0

SPLIT_SCHEDULES = [
    pytest.param([0.0, 0.55], id="one-off-grid-node"),
    pytest.param([0.0, 0.15, 0.37, 0.55, 0.83], id="several-off-grid-nodes"),
    pytest.param([0.0, 0.55, 0.55], id="repeated-node"),
]


def _race(schedules, boundaries, delta_t=0.125, max_t=1.0, **kwargs):
    """Run a unit-drift, zero-noise race with one stage schedule per accumulator.

    Stages within an accumulator share their parameters, so the stage split
    only changes the Euler grid, never the dynamics.
    """
    n_accumulators = len(schedules)
    max_stages = max(len(nodes) for nodes in schedules)
    shape = (1, n_accumulators, max_stages)
    node_array = np.zeros(shape)
    for i, nodes in enumerate(schedules):
        node_array[0, i, : len(nodes)] = nodes
    upper_intercept = np.empty(shape)
    upper_intercept[0] = np.asarray(boundaries, dtype=float)[:, None]
    kwargs.setdefault("x0_array", np.zeros((1, n_accumulators)))
    return cssm.race_multistage(
        mu_array=np.ones(shape),
        sigma_array=np.zeros(shape),
        node_array=node_array,
        d_array=np.array([[len(nodes) for nodes in schedules]], dtype=np.int32),
        upper_intercept_array=upper_intercept,
        upper_slope_array=np.zeros(shape),
        delta_t=delta_t,
        max_t=max_t,
        n_samples=1,
        random_state=1,
        return_option="full",
        **kwargs,
    )


def _rt(out):
    return out["rts"][0, 0, 0]


def _x_final(out):
    return out["metadata"]["x_final"][0, 0]


def test_issue_reproducer_no_hit_endpoint():
    out = _race([[0.0, 0.55]], [NO_HIT], delta_t=0.1, max_t=1.0, deadline=1.0)

    assert _rt(out) == OMISSION
    np.testing.assert_allclose(_x_final(out), [1.0], atol=1e-6)


def test_issue_reproducer_late_response():
    out = _race([[0.0, 0.55]], [0.98], delta_t=0.1, max_t=1.0, deadline=1.0)

    assert out["choices"][0, 0, 0] == 0
    # The crossing step ends at the horizon; its midpoint lies inside it.
    assert 0.9 - 1e-6 <= _rt(out) <= 1.0 + 1e-6


@pytest.mark.parametrize("nodes", SPLIT_SCHEDULES)
def test_no_hit_path_ends_at_the_horizon(nodes):
    out = _race([nodes], [NO_HIT])

    assert _rt(out) == OMISSION
    np.testing.assert_allclose(_x_final(out), [1.0], atol=1e-12)


@pytest.mark.parametrize("nodes", SPLIT_SCHEDULES)
def test_crossing_before_the_horizon_is_reported(nodes):
    out = _race([nodes], [0.98])

    assert out["choices"][0, 0, 0] == 0
    assert 0.98 - 0.125 <= _rt(out) <= 1.0


def test_different_schedules_across_accumulators_reach_the_horizon():
    out = _race([[0.0, 0.55], [0.0, 0.33, 0.71]], [NO_HIT, NO_HIT])

    assert _rt(out) == OMISSION
    np.testing.assert_allclose(_x_final(out), [1.0, 1.0], atol=1e-12)


def test_late_crossing_by_the_more_finely_split_accumulator_wins():
    out = _race([[0.0, 0.55], [0.0, 0.33, 0.71]], [NO_HIT, 0.98])

    assert out["choices"][0, 0, 0] == 1
    assert 0.98 - 0.125 <= _rt(out) <= 1.0


def test_coincident_nodes_across_accumulators_reach_the_horizon():
    schedules = [[0.0, 0.55], [0.0, 0.55, 0.55], [0.0, 0.3, 0.55]]
    out = _race(schedules, [NO_HIT] * 3)

    assert _rt(out) == OMISSION
    np.testing.assert_allclose(_x_final(out), [1.0, 1.0, 1.0], atol=1e-12)


def test_final_step_shorter_than_delta_t():
    # 0.9375 = 7.5 nominal steps, so the last step is half of delta_t.
    out = _race([[0.0, 0.55]], [NO_HIT], max_t=0.9375)

    assert _rt(out) == OMISSION
    np.testing.assert_allclose(_x_final(out), [0.9375], atol=1e-12)


def test_immediate_crossing_reports_the_nondecision_time():
    out = _race(
        [[0.0, 0.55]],
        [1.0],
        x0_array=np.ones((1, 1)),
        nondecision_time=0.2,
        deadline=2.0,
    )

    assert out["choices"][0, 0, 0] == 0
    np.testing.assert_allclose(_rt(out), 0.2, atol=1e-6)
    np.testing.assert_allclose(_x_final(out), [1.0], atol=1e-12)


def test_boundary_drop_at_an_off_grid_node_uses_the_preceding_step_midpoint():
    shape = (1, 1, 2)
    out = cssm.race_multistage(
        mu_array=np.ones(shape),
        sigma_array=np.zeros(shape),
        node_array=np.array([[[0.0, 0.55]]]),
        d_array=np.array([[2]], dtype=np.int32),
        upper_intercept_array=np.array([[[NO_HIT, 0.5]]]),
        upper_slope_array=np.zeros(shape),
        x0_array=np.zeros((1, 1)),
        delta_t=0.125,
        max_t=1.0,
        n_samples=1,
        random_state=1,
    )

    # The step [0.5, 0.55] is cut at the node, where the boundary drops below
    # the particle; the reported RT is that step's midpoint.
    assert out["choices"][0, 0, 0] == 0
    np.testing.assert_allclose(_rt(out), 0.525, atol=1e-6)


def test_nondecision_time_shifts_a_late_response():
    plain = _race([[0.0, 0.55]], [0.98])
    shifted = _race([[0.0, 0.55]], [0.98], nondecision_time=0.3, deadline=2.0)

    assert _rt(plain) != OMISSION
    np.testing.assert_allclose(_rt(shifted), _rt(plain) + 0.3, atol=1e-6)


def test_late_response_after_the_deadline_is_an_omission():
    out = _race([[0.0, 0.55]], [0.98], deadline=0.9)

    assert _rt(out) == OMISSION
    assert out["choices"][0, 0, 0] == 0
    # The deadline censors the response, not the simulated path.
    np.testing.assert_allclose(_x_final(out), [1.0], atol=1e-12)


def test_exhausted_step_budget_raises_instead_of_omitting():
    shape = (1, 1, 1)
    with pytest.raises(RuntimeError):
        _simulate_race_multistage(
            np.ones(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.ones((1, 1), dtype=np.int32),
            np.full(shape, NO_HIT),
            np.zeros(shape),
            np.zeros((1, 1)),
            0.125,
            1.0,
            np.array([1], dtype=np.uint64),
            max_steps=2,
        )
