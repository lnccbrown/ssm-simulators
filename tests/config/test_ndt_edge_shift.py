"""Tests for the ``ndt_edge_shift`` model-config declaration.

``"ndt_edge_shift": {"param": p, "scale": s}`` declares that a model's
response-time support starts at ``t - s * p``; a config without the key has a
support that starts at ``t``. These tests check the declarations against the
simulators, and that the builder and validators carry and police the key.
"""

import math

import pytest

from ssms.basic_simulators.simulator import OMISSION_SENTINEL, simulator
from ssms.config import KDE_NO_DISPLACE_T, ModelConfigBuilder, model_config
from ssms.config._modelconfig import _validate_configs, get_model_config
from ssms.config._modelconfig.validation import get_invalid_ndt_edge_shift_configs

N_SAMPLES = 3000
RANDOM_STATE = 7
TOL = 1e-9

# The one declaration shipped today: non-decision time t + U(-st, st).
UNIFORM_ST_DECLARATION = {"param": "st", "scale": 1.0}
DECLARING_MODELS = sorted(
    name for name, cfg in get_model_config().items() if "ndt_edge_shift" in cfg
)
PLAIN_MODELS = [
    "ddm",
    "ddm_sdv",
    "angle",
    "weibull",
    "levy",
    "ornstein",
    "race_3",
    "lca_3",
    "ddm_seq2_no_bias",
    "gamma_drift",
]
# Positions in [0, 1] of each parameter's bound range; unlisted params sit mid-box.
# The corner (t at its lower bound, the shift parameter at its upper bound) is
# where the simulator emits RTs below t; a small "a" makes decisions fast so
# that the negative shift shows clearly.
EDGE_PARAMETER_SETS = [
    {},
    {"t": 0.9, "st": 0.9},
    {"t": 0.0, "st": 1.0},
    {"t": 0.0, "st": 1.0, "a": 0.0},
]
# Non-vacuity is proven at the a-lower corner only: with fast decisions
# hundreds of RTs fall below t, whereas at the plain corner a handful do and
# the count depends on the RNG stream.
A_LOWER_CORNER = EDGE_PARAMETER_SETS[3]


def _resolve(bound, theta):
    """Resolve a bound that names another parameter (e.g. ``"st": (1e-3, "t")``)."""
    return theta[bound] if isinstance(bound, str) else bound


def _theta_in_box(cfg, **positions):
    """Build a parameter set from ``param_bounds_dict``.

    Each parameter sits at ``positions[param]`` (a fraction in [0, 1]) of its
    bound range, defaulting to the middle of the box.
    """
    theta = {}
    for param in cfg["params"]:
        lower, upper = (_resolve(b, theta) for b in cfg["param_bounds_dict"][param])
        theta[param] = lower + positions.get(param, 0.5) * (upper - lower)
    return theta


def _valid_rts(model, theta):
    """Simulate ``model`` at ``theta`` and return the RTs that are not omissions."""
    result = simulator(
        model=model, theta=theta, n_samples=N_SAMPLES, random_state=RANDOM_STATE
    )
    rts = result["rts"][result["rts"] != OMISSION_SENTINEL]
    assert rts.size > 0
    return rts


class TestDeclaredEdge:
    """Models that declare the key really have RTs starting at t - scale * param."""

    def test_declaring_models_are_the_uniform_st_family(self):
        """Only the t + U(-st, st) simulators (and the full_ddm2 alias) declare it."""
        assert DECLARING_MODELS == ["ddm_st", "full_ddm", "full_ddm2", "full_ddm_rv"]

    @pytest.mark.parametrize("model", DECLARING_MODELS)
    def test_declaration_is_t_minus_st(self, model):
        """Each declaring model pins the edge to exactly t - st."""
        assert model_config[model]["ndt_edge_shift"] == UNIFORM_ST_DECLARATION

    @pytest.mark.parametrize("model", DECLARING_MODELS)
    def test_rts_never_fall_below_declared_edge(self, model):
        """min(rt) >= t - scale * param for every parameter set, corner included."""
        cfg = model_config[model]
        shift = cfg["ndt_edge_shift"]
        for positions in EDGE_PARAMETER_SETS:
            theta = _theta_in_box(cfg, **positions)
            edge = theta["t"] - shift["scale"] * theta[shift["param"]]
            assert _valid_rts(model, theta).min() >= edge - TOL, positions

    @pytest.mark.parametrize("model", DECLARING_MODELS)
    def test_corner_emits_rts_below_t(self, model):
        """The declaration is not vacuous: at the corner the simulator goes below t."""
        cfg = model_config[model]
        theta = _theta_in_box(cfg, **A_LOWER_CORNER)
        assert _valid_rts(model, theta).min() < theta["t"]

    @pytest.mark.parametrize("model", DECLARING_MODELS)
    def test_corner_rts_fall_below_half_the_declared_shift(self, model):
        """The declared scale is tight, not merely a safe lower bound.

        At the a-lower corner the fastest RTs land within a few hundredths of
        t - st, so they fall below t - scale * param / 2 only if the scale is
        right: a scale twice too large passes the lower-bound check but puts
        this bar at t - st, which the simulator never crosses.
        """
        cfg = model_config[model]
        shift = cfg["ndt_edge_shift"]
        theta = _theta_in_box(cfg, **A_LOWER_CORNER)
        half_shift = 0.5 * shift["scale"] * theta[shift["param"]]
        assert _valid_rts(model, theta).min() < theta["t"] - half_shift


class TestUndeclaredSupportStartsAtT:
    """A config without the key has RTs that start at t."""

    @pytest.mark.parametrize("model", PLAIN_MODELS)
    def test_plain_model_rts_start_at_t(self, model):
        """No key in the config, and min(rt) >= t for a mid-box parameter set."""
        cfg = model_config[model]
        assert "ndt_edge_shift" not in cfg
        theta = _theta_in_box(cfg)
        assert _valid_rts(model, theta).min() >= theta["t"] - TOL


class TestKdeConsistency:
    """The KDE label estimator must not displace t for any declaring model."""

    def test_declaring_models_are_in_kde_no_displace_t(self):
        """Checked by config name, which is how lan_mlp consults KDE_NO_DISPLACE_T."""
        for model in DECLARING_MODELS:
            assert model_config[model]["name"] in KDE_NO_DISPLACE_T


class TestBuilderPlumbing:
    """ModelConfigBuilder carries and validates the key."""

    EXPECTED = UNIFORM_ST_DECLARATION

    def test_full_ddm2_alias_inherits_declaration(self):
        """full_ddm2 shares get_full_ddm_config and so carries the key."""
        assert model_config["full_ddm2"]["ndt_edge_shift"] == self.EXPECTED

    def test_from_model_carries_declaration(self):
        """from_model keeps the key of a declaring model."""
        assert ModelConfigBuilder.from_model("ddm_st")["ndt_edge_shift"] == (
            self.EXPECTED
        )

    def test_with_deadline_keeps_declaration(self):
        """Both the with_deadline call and the _deadline suffix keep the key."""
        base = ModelConfigBuilder.from_model("ddm_st")
        assert ModelConfigBuilder.with_deadline(base)["ndt_edge_shift"] == (
            self.EXPECTED
        )
        assert ModelConfigBuilder.from_model("ddm_st_deadline")["ndt_edge_shift"] == (
            self.EXPECTED
        )

    def test_from_scratch_keeps_declaration(self):
        """from_scratch whitelists the key alongside the other optional fields."""

        def my_sim(**kwargs):
            return {}

        config = ModelConfigBuilder.from_scratch(
            name="custom_st",
            params=["v", "a", "z", "t", "st"],
            simulator_function=my_sim,
            nchoices=2,
            ndt_edge_shift=self.EXPECTED,
        )

        assert config["ndt_edge_shift"] == self.EXPECTED

    def test_validate_config_accepts_declaration(self):
        """A well-formed declaration produces no error."""
        is_valid, errors = ModelConfigBuilder.validate_config(
            ModelConfigBuilder.from_model("ddm_st")
        )

        assert is_valid is True
        assert errors == []

    @pytest.mark.parametrize(
        "ndt_edge_shift, params",
        [
            pytest.param("st", ["v", "a", "z", "t", "st"], id="not-a-dict"),
            pytest.param({"param": "st"}, ["v", "a", "z", "t", "st"], id="no-scale"),
            pytest.param({"scale": 1.0}, ["v", "a", "z", "t", "st"], id="no-param"),
            pytest.param(
                {"param": "st", "scale": 1.0, "offset": 0.0},
                ["v", "a", "z", "t", "st"],
                id="extra-key",
            ),
            pytest.param(
                {"param": "sz", "scale": 1.0},
                ["v", "a", "z", "t", "st"],
                id="param-not-in-params",
            ),
            pytest.param(
                {"param": "st", "scale": 1.0}, ["v", "a", "z", "st"], id="no-t"
            ),
            pytest.param(
                {"param": "st", "scale": -1.0},
                ["v", "a", "z", "t", "st"],
                id="negative-scale",
            ),
            pytest.param(
                {"param": "st", "scale": math.nan},
                ["v", "a", "z", "t", "st"],
                id="nan-scale",
            ),
            pytest.param(
                {"param": "st", "scale": math.inf},
                ["v", "a", "z", "t", "st"],
                id="inf-scale",
            ),
            pytest.param(
                {"param": "st", "scale": True},
                ["v", "a", "z", "t", "st"],
                id="bool-scale",
            ),
            pytest.param(
                {"param": "st", "scale": "1.0"},
                ["v", "a", "z", "t", "st"],
                id="string-scale",
            ),
        ],
    )
    def test_validate_config_rejects_malformed_declaration(
        self, ndt_edge_shift, params
    ):
        """Each malformed shape is reported as an ndt_edge_shift error."""

        def my_sim(**kwargs):
            return {}

        config = {
            "params": params,
            "nchoices": 2,
            "simulator": my_sim,
            "ndt_edge_shift": ndt_edge_shift,
        }

        is_valid, errors = ModelConfigBuilder.validate_config(config)

        assert is_valid is False
        assert any("ndt_edge_shift" in err for err in errors)


class TestImportTimeValidator:
    """get_invalid_ndt_edge_shift_configs names the offending configs."""

    def test_reports_offending_config_name(self):
        """A synthetic malformed config is named; a well-formed one is not."""
        good = ModelConfigBuilder.from_model("ddm_st")
        bad = ModelConfigBuilder.from_model(
            "ddm_st", ndt_edge_shift={"param": "st", "scale": -1.0}
        )

        assert get_invalid_ndt_edge_shift_configs({"good": good, "bad": bad}) == ["bad"]

    def test_registry_is_clean(self):
        """Every shipped config passes the import-time check."""
        assert get_invalid_ndt_edge_shift_configs(get_model_config()) == []

    def test_validate_configs_raises_on_malformed_registry(self, monkeypatch):
        """_validate_configs runs the check on the registry it loads.

        It reads get_model_config from its module at call time, so patching
        the module attribute swaps in a registry whose parameter names pass
        the first check and whose only problem is the declaration.
        """
        bad = ModelConfigBuilder.from_model(
            "ddm_st", ndt_edge_shift={"param": "st", "scale": -1.0}
        )
        monkeypatch.setattr(
            "ssms.config._modelconfig.get_model_config", lambda: {"bad": bad}
        )

        with pytest.raises(ValueError, match="ndt_edge_shift.*\\['bad'\\]"):
            _validate_configs()
