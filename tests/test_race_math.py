"""Tests for the one-sided single-stage quantities in race-model equation (14)."""

import numpy as np
import pytest

from ssms.basic_simulators.race_math import big_F, q, small_f


PARAMS = dict(mu=0.75, sigma=1.0, a=1.0, b=0.0, T=1.5, x0=0.0)


def test_small_f_is_truncated_to_the_observation_window():
    values = small_f(np.array([-0.1, 0.0, 0.5, PARAMS["T"] + 0.1]), **PARAMS)
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] > 0.0
    assert values[3] == 0.0


def test_big_F_is_monotone_and_constant_after_T():
    times = np.array([-1.0, 0.0, 0.1, 0.5, PARAMS["T"], 3.0])
    values = big_F(times, **PARAMS)
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert np.all(np.diff(values) >= 0.0)
    assert values[-1] == pytest.approx(values[-2])


def test_q_is_zero_on_or_above_the_boundary():
    boundary = PARAMS["a"] + PARAMS["b"] * PARAMS["T"]
    values = q(np.array([boundary - 0.1, boundary, boundary + 0.1]), **PARAMS)
    assert values[0] > 0.0
    assert values[1] == 0.0
    assert values[2] == 0.0


def test_f_and_q_have_the_expected_probability_masses():
    time_grid = np.linspace(1e-6, PARAMS["T"], 100_000)
    position_grid = np.linspace(-10.0, PARAMS["a"] + PARAMS["b"] * PARAMS["T"], 100_000)
    passage_probability = float(big_F(PARAMS["T"], **PARAMS))

    f_mass = np.trapezoid(small_f(time_grid, **PARAMS), time_grid)
    q_mass = np.trapezoid(q(position_grid, **PARAMS), position_grid)

    assert f_mass == pytest.approx(passage_probability, abs=2e-4)
    assert q_mass == pytest.approx(1.0 - passage_probability, abs=2e-4)
