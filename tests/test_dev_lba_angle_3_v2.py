"""Tests for the ``dev_lba_angle_3_v2`` model.

``dev_lba_angle_3_v2`` is a 3-choice angle-LBA whose start-point parameter ``z``
is *relative*: it is the width of the uniform start-point distribution expressed
as a fraction of the threshold ``a``, so start points come from ``U(0, z * a)``
instead of ``U(0, z)``. That single change is what these tests pin down, since it
is the only thing separating this model from ``lba_angle_3`` and the only thing a
future refactor could silently break.

The model was ported from the ``origin/dev_lba_angle_3_v2`` development branch;
trained likelihood-approximation networks for it exist, so its parameter bounds
and simulated output must not drift.
"""

import numpy as np
import pytest

from ssms import Simulator
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder, get_boundary_registry, get_model_registry
from ssms.config._modelconfig import get_model_config

MODEL = "dev_lba_angle_3_v2"

# Bounds are reproduced from the dev branch and must stay put: the trained LAN
# for this model was fit inside this box.
EXPECTED_PARAMS = ["v0", "v1", "v2", "a", "z", "theta"]
EXPECTED_BOUNDS = [
    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    [6.0, 6.0, 6.0, 3.0, 0.9, 1.3],
]


@pytest.fixture
def config():
    """The registered model config."""
    return get_model_config()[MODEL]


@pytest.fixture
def sim():
    """Simulator instance for the model."""
    return Simulator(model=MODEL)


# ---------------------------------------------------------------------------
# 1. Config structure
# ---------------------------------------------------------------------------


def test_model_is_registered(config):
    """Model resolves through the config dict, registry and builder alike."""
    assert config["name"] == MODEL
    assert MODEL in get_model_registry().list_models()
    assert ModelConfigBuilder.from_model(MODEL)["name"] == MODEL


def test_config_param_counts_are_consistent(config):
    """params/default_params/param_bounds all agree with n_params."""
    n = config["n_params"]
    assert config["params"] == EXPECTED_PARAMS
    assert len(config["params"]) == n
    assert len(set(config["params"])) == n, "duplicate parameter names"
    assert len(config["default_params"]) == n
    assert len(config["param_bounds"][0]) == n
    assert len(config["param_bounds"][1]) == n
    assert len(config["choices"]) == config["nchoices"] == 3
    assert config["n_particles"] == 3


def test_config_bounds_are_pinned(config):
    """Bounds must match the dev branch exactly -- the trained LAN depends on them."""
    assert config["param_bounds"] == EXPECTED_BOUNDS


def test_config_bounds_are_valid_and_contain_defaults(config):
    """Every lower bound is below its upper bound, and defaults sit inside."""
    lowers, uppers = config["param_bounds"]
    for name, lo, hi in zip(config["params"], lowers, uppers):
        assert lo < hi, f"{name}: lower {lo} >= upper {hi}"
    for name, val, lo, hi in zip(
        config["params"], config["default_params"], lowers, uppers
    ):
        assert lo <= val <= hi, f"{name}: default {val} outside [{lo}, {hi}]"


def test_config_boundary_and_normalization(config):
    """Boundary resolves in the registry and param_bounds_dict was generated."""
    assert config["boundary_name"] in get_boundary_registry().list_boundaries()
    assert callable(config["boundary"])
    assert "param_bounds_dict" in config, "_normalize_param_bounds did not run"


def test_config_has_no_az_swap_constraint(config):
    """Relative z is consistent with any a, so no a/z swap may be applied.

    ``lba_angle_3`` needs ``SwapIfLessConstraint("a", "z")`` during sampling
    because its ``z`` is absolute. Applying it here would corrupt the
    parameterization, so the sampling pipeline must stay empty.
    """
    assert config["parameter_transforms"]["sampling"] == []


# ---------------------------------------------------------------------------
# 2. Smoke tests
# ---------------------------------------------------------------------------


def test_simulate_runs(sim, config):
    """Basic execution with default parameters."""
    res = sim.simulate(theta=config["default_params"], n_samples=1000)
    assert res["rts"].shape == (1000, 1)
    assert res["choices"].shape == (1000, 1)
    assert "metadata" in res


def test_choices_are_in_config_set(sim, config):
    """Simulated choices never leave the declared choice set."""
    res = sim.simulate(theta=config["default_params"], n_samples=2000)
    observed = set(int(c) for c in res["choices"].ravel())
    assert observed.issubset(set(config["choices"]))
    assert observed == {0, 1, 2}, "all three accumulators should win sometimes"


def test_rts_are_positive_and_finite(sim, config):
    """No non-positive or non-finite RTs without a deadline."""
    res = sim.simulate(theta=config["default_params"], n_samples=2000)
    assert np.all(np.isfinite(res["rts"]))
    assert np.all(res["rts"] > 0)


def test_metadata_identifies_the_simulator(config):
    """Metadata names this simulator, not the lba_angle it was derived from."""
    res = simulator(config["default_params"], model=MODEL, n_samples=10)
    assert res["metadata"]["simulator"] == "dev_lba_angle_v2"
    assert res["metadata"]["possible_choices"] == [0, 1, 2]


def test_n_threads_is_accepted(sim, config):
    """The n_threads argument is accepted (LBA models run single-threaded)."""
    res = sim.simulate(theta=config["default_params"], n_samples=100, n_threads=2)
    assert res["rts"].shape == (100, 1)


# ---------------------------------------------------------------------------
# 3. Relative-z semantics -- the defining property of this model
# ---------------------------------------------------------------------------


def test_mean_rt_scales_linearly_with_a_at_fixed_z():
    """Because z is a fraction of a, mean RT is proportional to a.

    Finishing time is ``(a - zs) / (v + tan(theta))`` with ``zs ~ U(0, z*a)``,
    so ``E[rt] = a * (1 - z/2) / (v + tan(theta))`` -- exactly linear in ``a``.
    With an absolute ``z`` the ``a`` term would not factor out.
    """
    theta = {"v0": 0.5, "v1": 0.3, "v2": 0.2, "z": 0.4, "theta": 0.0}
    np.random.seed(11)
    rt_a1 = simulator({**theta, "a": 0.5}, model=MODEL, n_samples=40000)["rts"].mean()
    np.random.seed(12)
    rt_a2 = simulator({**theta, "a": 1.5}, model=MODEL, n_samples=40000)["rts"].mean()
    assert rt_a2 / rt_a1 == pytest.approx(3.0, rel=0.02)


def test_relative_z_differs_from_absolute_z_model():
    """The same numbers mean different things here than in lba_angle_3.

    Holding ``z`` fixed and doubling ``a``, the relative-z model must scale mean
    RT by exactly 2, while ``lba_angle_3`` (absolute ``z``) scales it by
    ``(1.0 - z/2) / (0.5 - z/2)`` = 2.25 for z = 0.2. If the two models ever
    agree here, the reparameterization has been lost.
    """
    base = {"v0": 0.5, "v1": 0.3, "v2": 0.2, "z": 0.2, "theta": 0.0}
    ratios = {}
    for model in (MODEL, "lba_angle_3"):
        np.random.seed(21)
        lo = simulator({**base, "a": 0.5}, model=model, n_samples=40000)["rts"].mean()
        np.random.seed(22)
        hi = simulator({**base, "a": 1.0}, model=model, n_samples=40000)["rts"].mean()
        ratios[model] = hi / lo
    assert ratios[MODEL] == pytest.approx(2.00, rel=0.02)
    assert ratios["lba_angle_3"] == pytest.approx(2.25, rel=0.02)


def test_equivalent_to_lba_angle_with_scaled_z():
    """Bitwise: dev_lba_angle_v2(v,a,z,theta) == lba_angle(v,a,z*a,theta).

    This is the algebraic identity behind the model: rescaling ``z`` by ``a``
    turns the relative parameterization into the absolute one. Guards against a
    refactor that changes the start-point draw.
    """
    import cssm

    f32 = np.float32
    v = np.array([[0.5, 0.3, 0.2]], dtype=f32)
    a = np.array([[1.7]], dtype=f32)
    z = np.array([[0.6]], dtype=f32)
    th = np.array([[0.2]], dtype=f32)
    kwargs = dict(
        deadline=np.array([999.0], dtype=f32),
        sd=np.full((1, 3), 0.1, dtype=f32),
        t=np.array([0.0], dtype=f32),
        n_samples=3000,
        n_trials=1,
        nact=3,
    )

    np.random.seed(5)
    new = cssm.dev_lba_angle_v2(v=v, a=a, z=z, theta=th, **kwargs)
    np.random.seed(5)
    ref = cssm.lba_angle(v=v, a=a, z=(z * a).astype(f32), theta=th, **kwargs)

    np.testing.assert_array_equal(new["rts"], ref["rts"])
    np.testing.assert_array_equal(new["choices"], ref["choices"])


def test_z_zero_means_all_accumulators_start_at_zero():
    """With z = 0 there is no start-point variability, only drift variability."""
    theta = {"v0": 1.0, "v1": 1.0, "v2": 1.0, "a": 1.0, "z": 0.0, "theta": 0.0}
    np.random.seed(31)
    rts_z0 = simulator(theta, model=MODEL, n_samples=20000)["rts"]
    np.random.seed(31)
    rts_z9 = simulator({**theta, "z": 0.9}, model=MODEL, n_samples=20000)["rts"]
    # Start points eat into the distance to threshold, so z > 0 is faster
    # and more variable relative to its mean.
    assert rts_z9.mean() < rts_z0.mean()
    assert rts_z0.std() / rts_z0.mean() < rts_z9.std() / rts_z9.mean()


def test_threshold_above_lba_angle_3_ceiling_is_usable():
    """a up to 3.0 is in-bounds here, well above lba_angle_3's 1.1 ceiling.

    The wider threshold range is only coherent because z is relative; this
    checks it actually simulates rather than merely being declared.
    """
    theta = {"v0": 0.5, "v1": 0.3, "v2": 0.2, "a": 3.0, "z": 0.9, "theta": 0.0}
    res = simulator(theta, model=MODEL, n_samples=2000)
    assert np.all(np.isfinite(res["rts"]))
    assert np.all(res["rts"] > 0)


# ---------------------------------------------------------------------------
# 4. Parameter-space coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,theta_list",
    [
        ("theta_min", [0.5, 0.3, 0.2, 0.5, 0.2, 0.0]),
        ("theta_max", [0.5, 0.3, 0.2, 0.5, 0.2, 1.3]),
        ("a_min", [0.5, 0.3, 0.2, 0.1, 0.2, 0.3]),
        ("a_max", [0.5, 0.3, 0.2, 3.0, 0.2, 0.3]),
        ("z_min", [0.5, 0.3, 0.2, 0.5, 0.0, 0.3]),
        ("z_max", [0.5, 0.3, 0.2, 0.5, 0.9, 0.3]),
        ("v_min", [0.0, 0.0, 0.0, 0.5, 0.2, 0.3]),
        ("v_max", [6.0, 6.0, 6.0, 0.5, 0.2, 0.3]),
        ("all_max", [6.0, 6.0, 6.0, 3.0, 0.9, 1.3]),
        ("all_min", [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    ],
)
def test_bound_corners_simulate_cleanly(sim, tag, theta_list):
    """Every corner of the parameter box simulates without error or NaNs."""
    res = sim.simulate(theta=theta_list, n_samples=500)
    assert np.all(np.isfinite(res["rts"])), tag
    assert np.all(res["rts"] > 0), tag
    assert set(int(c) for c in res["choices"].ravel()).issubset({0, 1, 2}), tag


def test_random_in_bounds_draws(sim, config):
    """Random parameter vectors drawn from the configured box all run."""
    rng = np.random.default_rng(42)
    lowers, uppers = config["param_bounds"]
    for _ in range(10):
        theta_list = [rng.uniform(lo, hi) for lo, hi in zip(lowers, uppers)]
        res = sim.simulate(theta=theta_list, n_samples=100)
        assert res["rts"].shape[0] == 100
        assert np.all(np.isfinite(res["rts"]))


def test_multi_trial_simulation():
    """Per-trial parameter vectors produce per-trial output and choice probs."""
    theta = {
        "v0": np.array([0.5, 1.0, 2.0]),
        "v1": np.array([0.3, 0.4, 0.1]),
        "v2": np.array([0.2, 0.3, 0.05]),
        "a": np.array([0.5, 2.0, 1.0]),
        "z": np.array([0.2, 0.7, 0.0]),
        "theta": np.array([0.0, 0.5, 1.2]),
    }
    res = simulator(theta, model=MODEL, n_samples=300)
    assert res["rts"].shape == (300, 3, 1)
    assert res["choices"].shape == (300, 3, 1)
    assert res["choice_p"].shape == (3, 3)
    np.testing.assert_allclose(res["choice_p"].sum(axis=1), 1.0)
    # Trial 2 has a strongly dominant first accumulator
    assert res["choice_p"][2, 0] > res["choice_p"][2, 1]


def test_reproducible_under_numpy_seed(sim, config):
    """Same NumPy seed gives identical output.

    Like every simulator in ``lba_models``, this one draws from NumPy's global
    legacy stream and ignores ``random_state``; seeding NumPy is what makes it
    reproducible.
    """
    np.random.seed(1234)
    first = sim.simulate(theta=config["default_params"], n_samples=500)
    np.random.seed(1234)
    second = sim.simulate(theta=config["default_params"], n_samples=500)
    np.testing.assert_array_equal(first["rts"], second["rts"])
    np.testing.assert_array_equal(first["choices"], second["choices"])


# ---------------------------------------------------------------------------
# 5. Deadline variant
# ---------------------------------------------------------------------------


def test_deadline_variant_produces_omissions(config):
    """The _deadline suffix works and marks timeouts with the omission sentinel."""
    from ssms.basic_simulators.simulator import OMISSION_SENTINEL

    deadline_cfg = ModelConfigBuilder.from_model(MODEL + "_deadline")
    assert deadline_cfg["params"][-1] == "deadline"

    np.random.seed(77)
    res = simulator(
        list(config["default_params"]) + [0.7],
        model=MODEL + "_deadline",
        n_samples=4000,
    )
    n_omissions = int((res["rts"] == OMISSION_SENTINEL).sum())
    assert n_omissions > 0, "a 0.7s deadline should cut off some responses"
    assert n_omissions < 4000, "it should not cut off all of them"
    assert res["omission_p"][0, 0] == pytest.approx(n_omissions / 4000)


def test_generous_deadline_produces_no_omissions(config):
    """A deadline beyond the RT distribution leaves every response intact."""
    from ssms.basic_simulators.simulator import OMISSION_SENTINEL

    np.random.seed(78)
    res = simulator(
        list(config["default_params"]) + [10.0],
        model=MODEL + "_deadline",
        n_samples=2000,
    )
    assert not np.any(res["rts"] == OMISSION_SENTINEL)


# ---------------------------------------------------------------------------
# 6. Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_z", [1.5, -0.1])
def test_out_of_range_relative_z_is_rejected(bad_z):
    """z outside [0, 1] is not a valid fraction of the threshold."""
    with pytest.raises(ValueError, match=r"Relative starting point z"):
        simulator([0.5, 0.3, 0.2, 0.5, bad_z, 0.0], model=MODEL, n_samples=10)


def test_z_greater_than_a_is_allowed():
    """z > a is legitimate here, unlike in lba_angle_3.

    z is a fraction, so z = 0.9 with a = 0.15 is perfectly valid. The absolute-z
    check that guards lba_angle_3 must not be applied to this model.
    """
    res = simulator([0.5, 0.3, 0.2, 0.15, 0.9, 0.0], model=MODEL, n_samples=500)
    assert np.all(res["rts"] > 0)
