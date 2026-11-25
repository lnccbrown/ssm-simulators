"""Numerical and public-interface checks for the fixed planar exit model."""

import numpy as np
import pytest
from scipy.special import i0e, i1e

from ssms.basic_simulators import (
    OBSERVATION_SCHEMA_VERSION,
    Simulator,
    validate_observation_result,
)
from ssms.basic_simulators.circular_diffusion import sample_circular_diffusion
from ssms.basic_simulators.simulator import (
    bin_simulator_output,
    bin_simulator_output_pointwise,
    simulator,
)
from ssms.config import get_model_registry
from ssms.hssm_support import _get_seed, get_simulator_fun_internal, rng_fn
from ssms.support_utils.kde_class import LogKDE


def test_time_and_angle_laws_and_trial_order():
    """Check scaled time's Laplace transform and angular moments, not specific draws."""
    rows = [[0, 0, 0.1, 0], [3, -3, 3, 2], [-1.3, 0.6, 0.9, 0.12]]
    n = 16000
    draws = sample_circular_diffusion(rows, 227, n)
    for (vx, vy, a, t), (rt, angle) in zip(rows, draws.transpose(0, 2, 1), strict=True):
        dt = (rt - t) / a**2
        assert (dt > 0).all()
        assert ((angle >= -np.pi) & (angle < np.pi)).all()
        k = a * np.hypot(vx, vy)
        u = 2.0
        b = np.sqrt(k**2 + 2 * u)
        phi = angle - np.arctan2(vy, vx)
        for stat, expected in (
            (np.exp(-u * dt), np.exp(k - b) * i0e(k) / i0e(b)),
            (np.cos(phi), i1e(k) / i0e(k)),
            (np.sin(phi), 0.0),
            (dt, 0.5 if k == 0 else i1e(k) / (k * i0e(k))),
        ):
            assert abs(stat.mean() - expected) < 5 * stat.std() / np.sqrt(n)


def test_observation_schema_is_discoverable_before_sampling():
    registry = get_model_registry()
    config = registry.get("circular_diffusion")
    schema = config["observation_schema"]
    assert config["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert schema == (
        {"name": "rt", "kind": "continuous", "lower": 0, "lower_inclusive": False},
        {"name": "response", "kind": "circular", "lower": -np.pi, "upper": np.pi},
    )

    opts = {"theta": [0, 0, 1, 0.2], "n_samples": 2, "random_state": 23}
    results = (
        simulator(model="circular_diffusion", **opts),
        Simulator(model="circular_diffusion").simulate(**opts),
    )
    for result in results:
        meta = validate_observation_result(result)["metadata"]
        assert (
            meta["observation_schema_version"] == config["observation_schema_version"]
        )
        assert meta["observation_schema"] == schema

    # A caller can edit its result without redefining the model or another result.
    results[0]["metadata"]["observation_schema"][1]["upper"] = 0
    assert results[1]["metadata"]["observation_schema"] == schema
    assert registry.get("circular_diffusion")["observation_schema"] == schema
    assert schema[1]["upper"] == np.pi


def test_registry_preserves_coordinates_and_skips_categorical_summaries():
    rows = [[1.0, 0.5, 1.0, 0.25], [-1.0, 0.5, 2.0, 1.0]]
    theta = dict(zip(["vx", "vy", "a", "t"], np.transpose(rows), strict=True))
    opts = {"n_samples": 12, "random_state": 357}
    out = simulator(theta=theta, model="circular_diffusion", **opts)
    native = validate_observation_result(out)["observations"]
    assert native.shape == (12, 2, 2)
    np.testing.assert_array_equal(native[..., 0], out["rts"][..., 0])
    np.testing.assert_array_equal(native[..., 1], out["choices"][..., 0])
    assert not np.issubdtype(out["choices"].dtype, np.integer)
    assert "choice_p" not in out and "binned_128" not in out
    for categorical_only in (
        bin_simulator_output,
        bin_simulator_output_pointwise,
        LogKDE,
    ):
        with pytest.raises(ValueError, match="categorical responses"):
            categorical_only(out)
    alt = Simulator(model="circular_diffusion").simulate(theta=theta, **opts)
    np.testing.assert_array_equal(alt["observations"], native)


@pytest.mark.parametrize(
    ("trials", "replicas"), [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)]
)
def test_circular_sampler_layouts(trials, replicas):
    """Only the HSSM interface drops the singleton replica axis, never the trial axis."""
    theta = np.array([[1.0, 0.5, 1.0, 0.25], [-1.0, 0.5, 1.0, 0.75]])[:trials]
    seed = _get_seed(np.random.default_rng(357))
    out = simulator(
        theta=theta, model="circular_diffusion", n_samples=replicas, random_state=seed
    )
    native = validate_observation_result(out)["observations"]
    assert native.shape == (replicas, trials, 2)
    expected = native[0] if replicas == 1 else native.swapaxes(0, 1)
    named = get_simulator_fun_internal("circular_diffusion")
    np.testing.assert_array_equal(
        named(theta=theta, random_state=seed, n_replicas=replicas), expected
    )
    np.testing.assert_array_equal(
        sample_circular_diffusion(theta, seed, replicas), expected
    )
    if trials > 1:
        # The HSSM RNG must not transpose already-trial-first structured observations.
        actual = rng_fn(
            list(theta.T), trials * replicas, np.random.default_rng(357), named, 2
        )
        np.testing.assert_array_equal(actual, expected)


def test_generator_advances_and_omissions_are_explicit():
    row = [1.0, 1.0, 1.0, 0.2]
    rng = np.random.default_rng(312)
    first = sample_circular_diffusion(row, rng, 20)
    np.testing.assert_array_equal(first, sample_circular_diffusion(row, 312, 20))
    assert not np.array_equal(first, sample_circular_diffusion(row, rng, 20))
    out = simulator(
        theta=[*row, 0.1],
        model="circular_diffusion_deadline",
        n_samples=8,
        random_state=1,
    )
    validate_observation_result(out)
    assert np.isnan(out["observations"]).all() and out["omission_mask"].all()
    assert (out["rts"] == -999).all() and (out["choices"] == -999).all()
    assert (out["omission_p"] == 1).all()
    named = get_simulator_fun_internal("circular_diffusion_deadline")
    np.testing.assert_array_equal(
        named(theta=[*row, 0.1], random_state=1, n_replicas=8),
        out["observations"].swapaxes(0, 1),
    )


def test_no_noise_and_invalid_parameters():
    kwargs = {
        "model": "circular_diffusion",
        "n_samples": 2,
        "random_state": 1,
        "no_noise": True,
    }
    # At speed one along the negative x-axis, deterministic exits are at -pi.
    out = simulator(theta=[-1.0, 0.0, 1.5, 0.5], **kwargs)
    np.testing.assert_allclose(out["observations"][:, 0], [[2, -np.pi]] * 2)
    still = simulator(theta=[0.0, 0.0, 1.5, 0.5], **kwargs)
    assert still["omission_mask"].all()
    for bad in ([0, 0, 0, 0], [1, 1, 1, -0.1], [0, np.nan, 1, 0], [0, 1, 1]):
        with pytest.raises(ValueError, match="theta|radius"):
            sample_circular_diffusion(bad, 1, 1)
