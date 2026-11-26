"""`angle_extended` is `angle` under a second name.

Up to 0.16.0 `angle` had `v` in (-3, 3) and `angle_extended` had (-6, 6).
`angle` now has the wide box. `angle_extended` stays a registry key because
HSSM 0.5.0 and the pipeline config that trained the network look the model up
by that name, so the two configs must not drift apart.
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


class TestItIsAngleUnderASecondName:
    def test_only_the_name_differs(self, configs):
        base, ext = configs
        differing = {
            k for k in set(base) | set(ext) if repr(base.get(k)) != repr(ext.get(k))
        }
        assert differing == {"name"}

    def test_angle_has_the_wide_drift_box(self, configs):
        base, _ = configs
        assert dict(zip(base["params"], zip(*base["param_bounds"]))) == {
            "v": (-6.0, 6.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (1e-3, 2.0),
            "theta": (-0.1, 1.3),
        }

    def test_the_name_field_and_the_registry_key_agree(self, configs):
        base, ext = configs
        assert (base["name"], ext["name"]) == (BASE, EXT)


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
