"""Tests for the ``full_ddm_lan`` model configuration.

``full_ddm_lan`` is the full DDM on the box HSSM's ``full_ddm_lan`` network was
trained on. These tests pin that box, the meaning of the three variability
parameters as the simulator draws them (``sz`` a relative half-width, ``sv`` an
SD, ``st`` a half-width in seconds), and the support edge at ``t - st``.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import KDE_NO_DISPLACE_T, model_config
from ssms.support_utils.kde_class import LogKDE

N_SAMPLES = 20_000
RANDOM_STATE = 11

# The network's training box, in parameter order.
TRAINING_BOX = {
    "v": (-3.0, 3.0),
    "a": (0.3, 2.5),
    "z": (0.3, 0.7),
    "t": (0.25, 2.25),
    "sz": (1e-3, 0.2),
    "sv": (0.3, 2.5),
    "st": (1e-3, 0.35),
}

MID_BOX = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 1.0, "sz": 0.1, "sv": 1.0, "st": 0.1}


def test_full_ddm_lan_is_registered_on_its_training_box():
    config = model_config["full_ddm_lan"]
    assert config["name"] == "full_ddm_lan"
    assert config["params"] == ["v", "a", "z", "t", "sz", "sv", "st"]
    assert config["params"] == model_config["full_ddm"]["params"]
    assert config["param_bounds_dict"] == TRAINING_BOX
    assert config["choices"] == [-1, 1]
    assert config["nchoices"] == 2


def test_full_ddm_lan_differs_from_full_ddm_in_exactly_three_bounds():
    """sv's floor and ceiling and st's ceiling move; every other bound is full_ddm's."""
    lan = model_config["full_ddm_lan"]["param_bounds_dict"]
    ref = model_config["full_ddm"]["param_bounds_dict"]
    assert {p for p in lan if lan[p] != ref[p]} == {"sv", "st"}
    assert ref["sv"] == (1e-3, 2.0) and lan["sv"] == (0.3, 2.5)
    assert ref["st"] == (1e-3, 0.25) and lan["st"] == (1e-3, 0.35)


def test_full_ddm_lan_declares_the_uniform_st_edge():
    assert model_config["full_ddm_lan"]["ndt_edge_shift"] == {
        "param": "st",
        "scale": 1.0,
    }


def test_full_ddm_lan_is_in_kde_no_displace_t():
    assert "full_ddm_lan" in KDE_NO_DISPLACE_T


def test_default_params_lie_inside_the_box():
    config = model_config["full_ddm_lan"]
    for param, value in zip(config["params"], config["default_params"]):
        lower, upper = config["param_bounds_dict"][param]
        assert lower <= value <= upper, param


def test_full_ddm_lan_simulates():
    result = simulator(
        model="full_ddm_lan", theta=MID_BOX, n_samples=200, random_state=RANDOM_STATE
    )
    assert result["rts"].shape == (200, 1)
    assert result["choices"].shape == (200, 1)
    assert set(np.unique(result["choices"])).issubset({-1, 1})
    assert np.all(result["rts"] > 0)


def test_rts_start_at_t_minus_st():
    """The support edge is t - st: fast decisions reach below t but never below it."""
    theta = {**MID_BOX, "a": 0.3, "t": 1.0, "st": 0.3}
    rts = simulator(
        model="full_ddm_lan",
        theta=theta,
        n_samples=N_SAMPLES,
        random_state=RANDOM_STATE,
    )["rts"]
    assert rts.min() >= theta["t"] - theta["st"] - 1e-9
    assert np.mean(rts < theta["t"]) > 0.05


def _draws(mapping, value, size=200_000):
    return np.asarray(mapping(value)(size=size, random_state=RANDOM_STATE))


def test_st_is_a_half_width_in_seconds():
    """t_dist draws U(-st, st): support [-st, st], SD st / sqrt(3)."""
    st = 0.3
    draws = _draws(
        model_config["full_ddm_lan"]["simulator_param_mappings"]["t_dist"], st
    )
    assert draws.min() >= -st and draws.max() <= st
    assert draws.std() == pytest.approx(st / np.sqrt(3), rel=0.02)


def test_sz_is_a_relative_half_width():
    """z_dist draws U(-sz, sz) in units of the separation, so the trial's starting
    point is z + U(-sz, sz): support [-sz, sz], SD sz / sqrt(3)."""
    sz = 0.2
    draws = _draws(
        model_config["full_ddm_lan"]["simulator_param_mappings"]["z_dist"], sz
    )
    assert draws.min() >= -sz and draws.max() <= sz
    assert draws.std() == pytest.approx(sz / np.sqrt(3), rel=0.02)


def test_sv_is_a_standard_deviation():
    """v_dist draws Normal(0, sv)."""
    sv = 1.5
    draws = _draws(
        model_config["full_ddm_lan"]["simulator_param_mappings"]["v_dist"], sv
    )
    assert draws.mean() == pytest.approx(0.0, abs=0.02)
    assert draws.std() == pytest.approx(sv, rel=0.02)


def test_displace_t_is_refused():
    data = simulator(
        model="full_ddm_lan", theta=MID_BOX, n_samples=2_000, random_state=RANDOM_STATE
    )
    with pytest.raises(ValueError, match=r"model 'full_ddm_lan'.*displace_t=False"):
        LogKDE(simulator_data=data, displace_t=True)
    kde = LogKDE(simulator_data=data, displace_t=False)
    assert not kde.displace_t
