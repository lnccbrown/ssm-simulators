"""Library omission contract for every registered model and deadline variant.

A censored trial puts -999 in ``rts`` only, keeps its latent choice inside
``possible_choices``, and ``choice_p`` over the returned choices is finite and
sums to one (cf. upstream tests/test_omission_labels.py).
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator

SENTINEL = -999.0

THETAS = {
    "trdm_2": {
        "v0": 1.0,
        "v1": 0.6,
        "b0": 1.5,
        "b1": 1.5,
        "t": 0.3,
        "vT": 0.4,
        "sT": 0.5,
        "tT": 0.05,
    },
    "trdm_2_guess": {
        "v0": 1.0,
        "v1": 0.6,
        "b0": 1.5,
        "b1": 1.5,
        "t": 0.3,
        "vT": 0.4,
        "sT": 0.5,
        "tT": 0.05,
        "g": 0.8,
    },
    "trdm_3": {
        "v0": 1.0,
        "v1": 0.6,
        "v2": 0.3,
        "A": 0.3,
        "b": 1.5,
        "t": 0.3,
        "vT": 0.4,
        "sT": 0.5,
        "tT": 0.05,
    },
}


def _check(out, n_choices):
    rts = np.asarray(out["rts"]).ravel()
    choices = np.asarray(out["choices"]).ravel()
    possible = out["metadata"]["possible_choices"]
    assert possible == list(range(n_choices))
    omitted = rts == SENTINEL
    assert omitted.any(), "fixture must force omissions"
    assert not (choices == SENTINEL).any()
    assert np.isin(choices, possible).all()
    cp = np.asarray(out["choice_p"])
    assert np.all(np.isfinite(cp))
    assert cp.sum(axis=-1) == pytest.approx(1.0, abs=1e-12)
    assert float(np.asarray(out["omission_p"]).ravel()[0]) == pytest.approx(
        omitted.mean()
    )
    # omitted trials still carry every kind of latent choice
    assert len(np.unique(choices[omitted])) == n_choices
    return rts, choices


@pytest.mark.parametrize("model", list(THETAS))
def test_forced_deadline_omissions(model):
    n_choices = 3 if model == "trdm_3" else 2
    th = dict(THETAS[model], deadline=0.9)
    out = simulator(
        theta=th, model=model + "_deadline", n_samples=20000, random_state=0
    )
    rts, _ = _check(out, n_choices)
    valid = rts[rts != SENTINEL]
    assert valid.max() < 0.9
    # same seed without deadline: omitted exactly where RT >= deadline, choices unchanged
    free = simulator(theta=THETAS[model], model=model, n_samples=20000, random_state=0)
    rts_free = np.asarray(free["rts"]).ravel()
    np.testing.assert_array_equal(rts == SENTINEL, rts_free >= np.float32(0.9))
    np.testing.assert_array_equal(
        np.asarray(out["choices"]).ravel(), np.asarray(free["choices"]).ravel()
    )


@pytest.mark.parametrize("model", list(THETAS))
def test_forced_max_t_omissions(model):
    n_choices = 3 if model == "trdm_3" else 2
    out = simulator(
        theta=THETAS[model], model=model, n_samples=20000, random_state=0, max_t=0.4
    )
    rts, _ = _check(out, n_choices)
    assert rts[rts != SENTINEL].max() <= 0.3 + 0.4 + 1e-6


def test_never_finishing_race_is_an_omission_with_valid_choice():
    th = {
        "v0": 0.0,
        "v1": 0.0,
        "b0": 1.0,
        "b1": 1.0,
        "t": 0.3,
        "vT": 0.0,
        "sT": 0.0,
        "tT": 0.05,
    }
    out = simulator(
        theta=th, model="trdm_2", n_samples=2000, random_state=0, no_noise=True
    )
    rts = np.asarray(out["rts"]).ravel()
    assert np.all(rts == SENTINEL)
    assert np.isin(np.asarray(out["choices"]).ravel(), [0, 1]).all()
    assert np.asarray(out["choice_p"]).sum() == pytest.approx(1.0)


def _pipeline_labels(choices, rts, possible):
    """The training-label computation used by tests/test_omission_labels.py."""
    from ssms.dataset_generators.pipelines.simulation_pipeline import (
        SimulationPipeline,
    )

    return SimulationPipeline._compute_auxiliary_labels(
        None,
        {
            "choices": choices,
            "rts": rts,
            "metadata": {"possible_choices": list(possible)},
        },
    )


@pytest.mark.parametrize("model", list(THETAS))
def test_trdm_marks_omissions_in_rts_not_choices_for_training_labels(model):
    """TRDM case of the omission-label contract (tests/test_omission_labels.py).

    That module asks for a case for every simulator written outside
    ``src/cssm``; the TRDM censors in Python, so it is covered here.
    """
    n_choices = 3 if model == "trdm_3" else 2
    out = simulator(
        theta=dict(THETAS[model], deadline=0.9),
        model=model + "_deadline",
        n_samples=8000,
        random_state=0,
    )
    choices = np.asarray(out["choices"]).ravel()
    rts = np.asarray(out["rts"]).ravel()
    omitted = rts == SENTINEL
    assert omitted.any(), "no omissions: lower the deadline"
    assert not (choices == SENTINEL).any()

    labels = _pipeline_labels(choices, rts, possible=range(n_choices))
    assert labels["omission_p"][0, 0] == pytest.approx(float(omitted.mean()), abs=1e-9)
    assert labels["choice_p"][0].sum() == pytest.approx(1.0, abs=1e-9)
    kept = choices[~omitted]
    for i in range(n_choices):
        assert labels["choice_p_no_omission"][0, i] == pytest.approx(
            float((kept == i).mean()), abs=1e-9
        )
