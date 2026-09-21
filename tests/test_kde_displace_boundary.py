"""Tests for displacing KDE RTs by the admissibility boundary rt = t - st.

A model with trial-to-trial non-decision time puts density below `t`: a uniform
st kernel has support [t - st, t + st], so the smallest admissible RT is t - st,
not t, and displacing by t instead floored the band [t - st, t].
These tests pin `_recover_admissibility_boundary` on the two metadata shapes
that carry `st` (folded into `t_dist`, as `ddm_st` does, or as a plain entry, as
`full_ddm` does), and pin `LogKDE` to displacing by that boundary.
"""

import functools
import re
import types
from copy import deepcopy

import numpy as np
import pytest
import scipy.stats as sps

from ssms.basic_simulators.simulator import simulator
from ssms.support_utils.kde_class import LogKDE, _recover_admissibility_boundary

T = 0.6
ST = 0.2
BOUNDARY = T - ST


@pytest.fixture
def ddm_st_data():
    """ddm_st simulator output: st is folded into metadata['t_dist'], loc=-st."""
    return simulator(
        model="ddm_st",
        theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": T, "st": ST},
        n_samples=2000,
        random_state=42,
    )


@pytest.fixture
def full_ddm_data():
    """full_ddm simulator output: st is a plain metadata entry, no t_dist."""
    return simulator(
        model="full_ddm",
        theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": T, "sz": 0.05, "sv": 0.2, "st": ST},
        n_samples=2000,
        random_state=42,
    )


@pytest.fixture
def ddm_data():
    """ddm simulator output: no ndt variability, so the boundary really is t."""
    return simulator(
        model="ddm",
        theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": T},
        n_samples=2000,
        random_state=42,
    )


def test_boundary_recovers_t_minus_st_from_t_dist(ddm_st_data):
    """ddm_st hides st in a functools.partial with loc=-st; the boundary is t - st."""
    metadata = ddm_st_data["metadata"]
    assert "st" not in metadata
    assert isinstance(metadata["t_dist"], functools.partial)
    assert _recover_admissibility_boundary(metadata) == pytest.approx(
        BOUNDARY, abs=1e-6
    )


def test_boundary_recovers_t_minus_st_from_plain_metadata(full_ddm_data):
    """full_ddm keeps st as a numeric metadata entry and emits no t_dist."""
    metadata = full_ddm_data["metadata"]
    assert "t_dist" not in metadata
    assert _recover_admissibility_boundary(metadata) == pytest.approx(
        BOUNDARY, abs=1e-6
    )


def test_boundary_offsets_by_the_t_dist_loc_keyword():
    """loc centers the ndt kernel, so the boundary rides with it, sign and all."""
    metadata = {
        "t": np.array([1.0]),
        "t_dist": functools.partial(sps.uniform.rvs, loc=-0.05, scale=0.1),
    }
    assert _recover_admissibility_boundary(metadata) == pytest.approx(0.95)

    metadata["t_dist"] = functools.partial(sps.uniform.rvs, loc=0.1, scale=0.2)
    assert _recover_admissibility_boundary(metadata) == pytest.approx(1.1)


def test_boundary_is_t_for_a_model_without_ndt_variability():
    """ddm_sdv's degenerate t_dist (loc=0, scale=0) leaves the boundary at t."""
    metadata = simulator(
        model="ddm_sdv",
        theta={"v": 0.5, "a": 1.0, "z": 0.5, "t": T, "sv": 0.3},
        n_samples=100,
        random_state=42,
    )["metadata"]
    assert _recover_admissibility_boundary(metadata) == pytest.approx(T, abs=1e-6)


def test_boundary_is_t_without_st_or_t_dist():
    """Metadata from a model with no ndt kernel puts the boundary at t itself."""
    assert _recover_admissibility_boundary({"t": np.array([T])}) == pytest.approx(T)


def test_boundary_raises_on_a_t_dist_that_is_not_a_partial():
    """A t_dist we cannot read keywords off is malformed, not a licence to use t."""
    with pytest.raises(ValueError, match="not a functools.partial"):
        _recover_admissibility_boundary({"t": np.array([T]), "t_dist": "not a partial"})


def test_boundary_raises_on_a_t_dist_whose_keywords_are_not_a_mapping():
    """A `keywords` attribute that is not a mapping is malformed metadata too."""
    with pytest.raises(ValueError, match="not a functools.partial"):
        _recover_admissibility_boundary(
            {
                "t": np.array([T]),
                "t_dist": types.SimpleNamespace(keywords=None, func=None),
            }
        )


def test_boundary_raises_on_multiple_t_values():
    """A per-trial t has no single boundary to displace a whole KDE by."""
    with pytest.raises(ValueError, match="Multiple t values"):
        _recover_admissibility_boundary({"t": np.array([0.3, 0.4])})


def test_boundary_raises_on_multiple_st_values():
    """A per-trial st is refused for the same reason a per-trial t is."""
    with pytest.raises(ValueError, match="Multiple st values"):
        _recover_admissibility_boundary(
            {"t": np.array([0.6]), "st": np.array([0.1, 0.2])}
        )


def test_boundary_raises_when_t_dist_has_no_loc():
    """A t_dist that does not say where it sits cannot place the boundary."""
    metadata = {
        "t": np.array([0.6]),
        "t_dist": functools.partial(sps.uniform.rvs, scale=0.4),
    }
    with pytest.raises(ValueError, match="no 'loc' keyword"):
        _recover_admissibility_boundary(metadata)


def test_boundary_raises_on_a_vector_valued_t_dist_loc():
    """A per-trial loc holding two offsets has no single boundary to displace by."""
    metadata = {
        "t": np.array([T]),
        "t_dist": functools.partial(
            sps.uniform.rvs, loc=np.array([-0.1, -0.2]), scale=0.4
        ),
    }
    with pytest.raises(ValueError, match="Multiple t_dist loc values"):
        _recover_admissibility_boundary(metadata)


def test_boundary_accepts_a_broadcast_st_with_one_unique_value():
    """Per-trial broadcasting repeats one st; that is not ambiguity."""
    metadata = {"t": np.array([T]), "st": np.full(7, ST)}
    assert _recover_admissibility_boundary(metadata) == pytest.approx(BOUNDARY)


def test_boundary_accepts_a_broadcast_t_dist_loc_with_one_unique_value():
    """The same for a loc broadcast across trials by the simulator."""
    metadata = {
        "t": np.array([T]),
        "t_dist": functools.partial(sps.uniform.rvs, loc=np.full(7, -ST), scale=2 * ST),
    }
    assert _recover_admissibility_boundary(metadata) == pytest.approx(BOUNDARY)


def _with_metadata(data, **overrides):
    """Copy simulator output with metadata entries replaced, for ambiguity tests."""
    out = deepcopy(data)
    out["metadata"].update(overrides)
    return out


def test_logkde_raises_on_multiple_st_values(full_ddm_data):
    """Ambiguous st must propagate, not fall back to displacing by t."""
    data = _with_metadata(full_ddm_data, st=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="Multiple st values"):
        LogKDE(simulator_data=data, displace_t=True)


def test_logkde_raises_on_a_vector_valued_t_dist_loc(ddm_st_data):
    """A t_dist loc carrying two offsets must propagate for the same reason."""
    data = _with_metadata(
        ddm_st_data,
        t_dist=functools.partial(
            sps.uniform.rvs, loc=np.array([-0.1, -0.2]), scale=0.4
        ),
    )
    with pytest.raises(ValueError, match="Multiple t_dist loc values"):
        LogKDE(simulator_data=data, displace_t=True)


def test_logkde_raises_on_a_t_dist_without_loc(ddm_st_data):
    """Malformed variability metadata is refused rather than papered over with t."""
    data = _with_metadata(
        ddm_st_data, t_dist=functools.partial(sps.uniform.rvs, scale=0.4)
    )
    with pytest.raises(ValueError, match="no 'loc' keyword"):
        LogKDE(simulator_data=data, displace_t=True)


def test_logkde_accepts_a_broadcast_st(full_ddm_data):
    """One st repeated per trial still yields the t - st boundary."""
    data = _with_metadata(full_ddm_data, st=np.full(7, ST))
    kde = LogKDE(simulator_data=data, displace_t=True)
    assert kde.displace_t_val == pytest.approx(BOUNDARY, abs=1e-6)


@pytest.mark.parametrize("fixture_name", ["ddm_st_data", "full_ddm_data"])
def test_logkde_displaces_by_the_boundary_not_by_t(fixture_name, request):
    """LogKDE(displace_t=True) subtracts t - st for both metadata shapes."""
    data = request.getfixturevalue(fixture_name)
    kde = LogKDE(simulator_data=data, displace_t=True)
    assert kde.displace_t_val == pytest.approx(BOUNDARY, abs=1e-6)


def test_logkde_falls_back_to_t_without_st(ddm_data):
    """With neither 'st' nor a 't_dist' the boundary is t, the old behaviour."""
    kde = LogKDE(simulator_data=ddm_data, displace_t=True)
    assert kde.displace_t_val == pytest.approx(T, abs=1e-6)


def test_logkde_keeps_density_between_the_boundary_and_t(ddm_st_data):
    """RTs in [t - st, t] are admissible, so the KDE must still be a density there.

    Displacing by t instead sends every training RT below t negative, the logs
    come out nan, and both choices collapse to the flat 'no_base_data' fallback.
    """
    lb = -66.774
    rts = np.array([BOUNDARY + 0.25 * ST, BOUNDARY + 0.75 * ST])
    kde = LogKDE(simulator_data=ddm_st_data, displace_t=True)

    assert all(base != "no_base_data" for base in kde.base_kdes)

    result = kde.kde_eval({"rts": rts, "choices": np.array([1, 1])}, lb=lb)
    assert np.all(result > lb)
    assert result[0] != result[1]


def test_logkde_samples_land_above_the_boundary(ddm_st_data):
    """Sampling re-adds the boundary, so draws start at t - st, not at t."""
    kde = LogKDE(simulator_data=ddm_st_data, displace_t=True)
    samples = kde.kde_sample(n_samples=2000, random_state=42)

    assert np.all(samples["rts"] > BOUNDARY - 1e-6)
    assert np.any(samples["rts"] < T)


@pytest.mark.parametrize(
    ("metadata", "name"),
    [
        pytest.param({"t": np.array([T]), "st": np.nan}, "st", id="st_nan"),
        pytest.param({"t": np.array([T]), "st": np.inf}, "st", id="st_inf"),
        pytest.param(
            {"t": np.array([T]), "st": np.array([0.1, np.nan])}, "st", id="st_mixed"
        ),
        pytest.param({"t": np.array([np.nan]), "st": ST}, "t", id="t_nan"),
        pytest.param(
            {
                "t": np.array([T]),
                "t_dist": functools.partial(sps.uniform.rvs, loc=-np.inf, scale=0.4),
            },
            "t_dist loc",
            id="t_dist_loc_neg_inf",
        ),
    ],
)
def test_boundary_raises_on_non_finite_variability(metadata, name):
    """NaN or inf metadata yields a NaN/inf boundary, so it is refused outright."""
    with pytest.raises(ValueError, match=f"Non-finite {re.escape(name)} values"):
        _recover_admissibility_boundary(metadata)


def test_logkde_raises_on_a_non_finite_st(full_ddm_data):
    """A non-finite st must propagate, not shift every RT by NaN."""
    data = _with_metadata(full_ddm_data, st=np.nan)
    with pytest.raises(ValueError, match="Non-finite st values"):
        LogKDE(simulator_data=data, displace_t=True)


def test_logkde_raises_on_a_non_finite_t_dist_loc(ddm_st_data):
    """The same for an infinite t_dist loc, which would send the boundary to -inf."""
    data = _with_metadata(
        ddm_st_data, t_dist=functools.partial(sps.uniform.rvs, loc=-np.inf, scale=0.4)
    )
    with pytest.raises(ValueError, match="Non-finite t_dist loc values"):
        LogKDE(simulator_data=data, displace_t=True)
