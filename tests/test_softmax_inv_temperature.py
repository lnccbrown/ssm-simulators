"""HSSM's choice-only ``softmax_inv_temperature_<n>`` models."""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder
from ssms.hssm_support import get_simulator_fun_internal

CASES = [
    (2, [-1, 1]),
    (3, [0, 1, 2]),
    (4, [0, 1, 2, 3]),
]


def _softmax(beta, logits):
    z = beta * np.array([0.0, *logits])
    p = np.exp(z - z.max())
    return p / p.sum()


@pytest.mark.parametrize(("n", "choices"), CASES)
def test_config_matches_hssm_definition(n, choices):
    cfg = ModelConfigBuilder.from_model(f"softmax_inv_temperature_{n}")

    assert cfg["params"] == ["beta", *[f"logit{i}" for i in range(1, n)]]
    assert cfg["choices"] == choices
    assert cfg["nchoices"] == n
    assert cfg["n_params"] == n


@pytest.mark.parametrize(("n", "choices"), CASES)
def test_choice_frequencies_match_softmax(n, choices):
    beta, logits = 1.5, [0.8, -0.4, 1.2][: n - 1]
    # Dict theta in reverse order: the simulator must bind columns by name.
    theta = {f"logit{i}": v for i, v in reversed(list(enumerate(logits, 1)))}
    theta["beta"] = beta

    out = simulator(
        theta=theta,
        model=f"softmax_inv_temperature_{n}",
        n_samples=40_000,
        random_state=7,
    )

    expected = _softmax(beta, logits)
    freqs = np.array([np.mean(out["choices"] == c) for c in choices])
    np.testing.assert_allclose(freqs, expected, atol=0.01)
    np.testing.assert_allclose(out["choice_p"].ravel(), freqs)
    assert set(np.unique(out["choices"])) <= set(choices)
    assert out["metadata"]["possible_choices"] == choices
    assert np.all(out["rts"] == -1.0)


def test_two_choice_labels_are_minus_one_and_one():
    out = simulator(
        theta={"beta": 0.0, "logit1": 0.0},
        model="softmax_inv_temperature_2",
        n_samples=2000,
        random_state=1,
    )

    assert set(np.unique(out["choices"])) == {-1, 1}
    # go_p counts the max label, so it is P(choice == 1) here.
    np.testing.assert_allclose(out["go_p"], out["choice_p"][:, [1]])


def test_trialwise_theta():
    out = simulator(
        theta={"beta": np.array([10.0, 10.0]), "logit1": np.array([-5.0, 5.0])},
        model="softmax_inv_temperature_2",
        n_samples=500,
        random_state=3,
    )

    assert out["choices"].shape == (500, 2, 1)
    assert np.all(out["choices"][:, 0] == -1)
    assert np.all(out["choices"][:, 1] == 1)


@pytest.mark.parametrize(("n", "choices"), CASES)
def test_hssm_simulator_fun_returns_rt_response_pairs(n, choices):
    sim = get_simulator_fun_internal(f"softmax_inv_temperature_{n}")
    theta = np.array([[2.0, *([0.3] * (n - 1))]] * 5)

    out = sim(theta=theta, n_replicas=1, random_state=5)

    assert sim.choices == choices
    assert out.shape == (5, 2)
    assert np.all(out[:, 0] == -1.0)
    assert set(np.unique(out[:, 1])) <= set(choices)


def test_label_count_mismatch_raises():
    from ssms.basic_simulators.inv_temp_softmax import softmax_inv_temperature

    with pytest.raises(ValueError, match="choice labels"):
        softmax_inv_temperature(beta=1.0, logit1=0.0, choices=[0, 1, 2])


def test_stray_q_keys_in_theta_are_ignored():
    out = simulator(
        theta={"beta": 1.0, "logit1": 0.0, "q0": 5.0},
        model="softmax_inv_temperature_2",
        n_samples=10,
        random_state=1,
    )

    assert set(np.unique(out["choices"])) <= {-1, 1}
