"""`angle_extended` is `angle` with a wider drift box and nothing else.

That relationship is the thing worth testing. A structural check that the
config is well-formed would pass just as happily if someone edited one of the
two configs and not the other, and the two silently diverging is the failure
mode a bounds-only variant actually has.
"""

import numpy as np
import pytest

from ssms.basic_simulators import Simulator
from ssms.config._modelconfig import get_model_config

BASE, EXT = "angle", "angle_extended"


@pytest.fixture(scope="module")
def configs():
    cfg = get_model_config()
    return cfg[BASE], cfg[EXT]


class TestItIsAngleExceptForTheDriftBox:
    def test_only_the_bounds_differ(self, configs):
        base, ext = configs
        # `name` differs by construction; `param_bounds_dict` is derived from
        # `param_bounds` by `_normalize_param_bounds`, so it differs with it.
        expected = {"name", "param_bounds", "param_bounds_dict"}
        differing = {
            k for k in set(base) | set(ext) if repr(base.get(k)) != repr(ext.get(k))
        }
        assert differing == expected, (
            f"angle_extended diverged from angle in {differing - expected} "
            "-- the two configs are meant to stay trivially diffable"
        )

    def test_only_v_moved_and_only_outward(self, configs):
        base, ext = configs
        for i, name in enumerate(base["params"]):
            lo_b, hi_b = base["param_bounds"][0][i], base["param_bounds"][1][i]
            lo_e, hi_e = ext["param_bounds"][0][i], ext["param_bounds"][1][i]
            if name == "v":
                assert (lo_e, hi_e) == (-6.0, 6.0)
                assert lo_e < lo_b and hi_e > hi_b, "the point is a WIDER box"
            else:
                assert (lo_e, hi_e) == (lo_b, hi_b), f"{name} should not have moved"

    def test_the_name_field_and_the_registry_key_agree(self, configs):
        # `shrink_spot_extended`, the registry's other bounds-only variant,
        # sets its own "name" to "shrink_spot" -- key and name disagree there.
        # This asserts we did not inherit that.
        _, ext = configs
        assert ext["name"] == EXT


class TestTheConfigIsWellFormed:
    def test_counts_are_consistent(self, configs):
        _, ext = configs
        assert len(ext["params"]) == ext["n_params"]
        assert len(ext["default_params"]) == ext["n_params"]
        assert len(ext["param_bounds"][0]) == ext["n_params"]
        assert len(ext["param_bounds"][1]) == ext["n_params"]

    def test_bounds_are_ordered_and_defaults_lie_inside(self, configs):
        _, ext = configs
        for name, lo, hi, default in zip(
            ext["params"], *ext["param_bounds"], ext["default_params"]
        ):
            assert lo < hi, f"{name}: lower {lo} >= upper {hi}"
            assert lo <= default <= hi, (
                f"{name}: default {default} outside [{lo}, {hi}]"
            )


class TestTheSimulatorReachesTheExtension:
    def test_default_params_simulate(self, configs):
        _, ext = configs
        result = Simulator(model=EXT).simulate(
            theta=ext["default_params"], n_samples=1000
        )
        assert np.asarray(result["rts"]).shape[0] == 1000
        observed = {
            int(c) for c in np.asarray(result["choices"]).flatten() if c != -999.0
        }
        assert observed.issubset(set(ext["choices"]))

    def test_random_draws_from_the_whole_box_simulate(self, configs):
        _, ext = configs
        sim = Simulator(model=EXT)
        rng = np.random.default_rng(42)
        for _ in range(5):
            theta = [rng.uniform(lo, hi) for lo, hi in zip(*ext["param_bounds"])]
            assert (
                np.asarray(sim.simulate(theta=theta, n_samples=100)["rts"]).shape[0]
                == 100
            )

    @pytest.mark.parametrize("v", [-6.0, -4.0, 4.0, 6.0])
    def test_the_new_range_behaves_the_way_a_drift_should(self, v):
        # Not just "does not crash": beyond angle's old +/-3 the choice should
        # be decided by the sign of v and the RT should be short, which is what
        # makes this region worth having a network for.
        result = Simulator(model=EXT).simulate(
            theta=[v, 1.0, 0.5, 0.3, 0.5], n_samples=2000
        )
        rt = np.asarray(result["rts"]).reshape(-1)
        choices = np.asarray(result["choices"]).reshape(-1)
        assert (choices > 0).mean() == pytest.approx(1.0 if v > 0 else 0.0, abs=0.02)
        assert np.median(rt) < 1.0
