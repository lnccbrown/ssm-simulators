"""Validation tests for analytical one-sided race-model quantities."""

import pytest

from ssms.basic_simulators.race_math import big_F, q, small_f


PARAMS = dict(mu=0.75, sigma=1.0, a=1.0, b=0.0, T=1.5, x0=0.0)


@pytest.mark.parametrize(
    ("function", "argument"),
    [(small_f, 0.5), (big_F, 0.5), (q, 0.0)],
)
@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("sigma", 0.0, "sigma must be positive"),
        ("T", 0.0, "T must be positive"),
        ("x0", 1.0, "x0 must be less than a"),
    ],
)
def test_analytical_race_functions_validate_shared_parameters(
    function, argument, parameter, value, message
):
    params = {**PARAMS, parameter: value}
    with pytest.raises(ValueError, match=message):
        function(argument, **params)
