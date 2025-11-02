import pytest

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df
from ssms.config._modelconfig.utils import _new_param, _get, _new_config


def make_sample_params():
    return {
        "v": _new_param(0.0, -3.0, 3.0),
        "a": _new_param(1.0, 0.3, 2.5),
        "z": _new_param(0.5, 0.1, 0.9),
        "t": _new_param(1e-3, 0.0, 2.0),
    }


def test_new_param_structure():
    p = _new_param(1.5, -1.0, 2.5)
    assert isinstance(p, dict)
    assert p["default"] == 1.5
    assert p["bounds"] == [-1.0, 2.5]


def test_get_name_defaults_bounds():
    params = make_sample_params()
    assert _get(params, "name") == ["v", "a", "z", "t"]
    assert _get(params, "defaults") == [0.0, 1.0, 0.5, 1e-3]
    assert _get(params, "bounds") == [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]]


def test_get_unknown_field_raises():
    params = make_sample_params()
    with pytest.raises(ValueError):
        _get(params, "unknown_field")


def test_new_config_contents_and_counts():
    name = "ddm"
    params = make_sample_params()
    boundary_name = "constant"
    boundary = bf.constant
    drift_name = "constant"
    drift_fun = df.constant
    choices = [-1, 1]
    n_particles = 1
    simulator = cssm.ddm_flexbound

    cfg = _new_config(
        name=name,
        param_dict=params,
        boundary_name=boundary_name,
        boundary=boundary,
        drift_name=drift_name,
        drift_fun=drift_fun,
        choices=choices,
        n_particles=n_particles,
        simulator=simulator,
    )

    assert cfg["name"] == name
    assert cfg["params"] == _get(params, "name")
    assert cfg["param_bounds"] == _get(params, "bounds")
    assert cfg["default_params"] == _get(params, "defaults")
    assert cfg["n_params"] == len(params)
    assert cfg["boundary_name"] == boundary_name
    assert cfg["boundary"] is boundary
    assert cfg["drift_name"] == drift_name
    assert cfg["drift_fun"] is drift_fun
    assert cfg["choices"] == choices
    assert cfg["nchoices"] == len(choices)
    assert cfg["n_particles"] == n_particles
    assert cfg["simulator"] == simulator
