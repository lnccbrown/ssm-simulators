"""The declared reductions, run exactly as ``MLBA_SPEC`` states them.

For each entry, the MLBA model is simulated with the declared ``settings``,
and the target library model is simulated with its own defaults overridden by
``map`` (which only renames parameters).  Nothing is hand-converted.

Seeding note.  ``cssm.lba_vanilla`` ignores ``random_state`` and draws from
NumPy's global legacy RNG (see ``ssms/basic_simulators/lba_seeded.py``), so the built-in
``lba2``/``lba3`` cannot be driven from ``random_state`` at all.  To put both
races on the same RNG stream the global seed is set before the lba call, and
the same value is handed to the MLBA call, whose wrapper seeds the same
generator.  Both models then consume draws in the same order.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder
from ssms.transforms import MLBAFrontEnd

N_SAMPLES = 200_000
SEED = 20140137


def _reduction_cases(declarations):
    for model in declarations["models"]:
        for reduction in model["reductions"]:
            yield pytest.param(
                model["name"], reduction, id=f"{model['name']}->{reduction['to']}"
            )


def _run_pair(source_model: str, reduction: dict, n_samples: int = N_SAMPLES):
    settings = reduction["settings"]
    source_cfg = ModelConfigBuilder.from_model(source_model)
    assert set(settings) == set(source_cfg["params"]), (
        "the reduction settings must pin every parameter of the source model"
    )

    target_cfg = ModelConfigBuilder.from_model(reduction["to"])
    target_theta = dict(zip(target_cfg["params"], target_cfg["default_params"]))
    for source_param, target_param in reduction["map"].items():
        assert target_param in target_theta, target_param
        target_theta[target_param] = settings[source_param]

    np.random.seed(SEED)
    target = simulator(
        theta=target_theta,
        model=reduction["to"],
        n_samples=n_samples,
        random_state=SEED,
    )
    np.random.seed(SEED)
    source = simulator(
        theta=dict(settings), model=source_model, n_samples=n_samples, random_state=SEED
    )
    return source, target, target_theta


def test_declared_reductions_exist(declarations):
    cases = list(_reduction_cases(declarations))
    assert len(cases) == 2, [c.id for c in cases]


def test_mlba2_reduces_to_lba2_identically(declarations, declared_models):
    """expect: identical -- same engine, same seed, bit-for-bit same samples."""
    reduction = declared_models["mlba2"]["reductions"][0]
    assert reduction["to"] == "lba2" and reduction["expect"] == "identical"
    source, target, _ = _run_pair("mlba2", reduction)
    np.testing.assert_array_equal(source["rts"], target["rts"])
    np.testing.assert_array_equal(source["choices"], target["choices"])


def test_mlba3_reduces_to_lba3_in_distribution(declarations, declared_models):
    """expect: distribution.

    lba3's default drifts (0.25, 0.5, 0.25) are not reachable bit-exactly:
    the attribute sums that produce them involve 1/12, and in float32 the
    drifts land one ulp (3e-8) away.  Everything else is shared, so the two
    runs are compared as distributions -- and, because the perturbation is
    tiny, the per-sample difference is reported too.
    """
    reduction = declared_models["mlba3"]["reductions"][0]
    assert reduction["to"] == "lba3" and reduction["expect"] == "distribution"
    source, target, _ = _run_pair("mlba3", reduction)

    p_source = source["choice_p"].ravel()
    p_target = target["choice_p"].ravel()
    # Choice proportions: 3-sigma band at n = 200k is ~0.003.
    np.testing.assert_allclose(p_source, p_target, atol=0.005)

    for choice in (0, 1, 2):
        rt_s = source["rts"].ravel()[source["choices"].ravel() == choice]
        rt_t = target["rts"].ravel()[target["choices"].ravel() == choice]
        assert stats.ks_2samp(rt_s, rt_t).pvalue > 0.001, choice

    # The drift perturbation is 1 float32 ulp, so the paired samples barely move.
    max_abs = float(np.max(np.abs(source["rts"] - target["rts"])))
    assert max_abs < 1e-4, max_abs
    assert float((source["choices"] != target["choices"]).mean()) < 1e-3


def test_reduction_drifts_hit_the_target_models_defaults(declared_models):
    """The front end really does produce the target model's default drifts."""
    expectations = {"mlba2": ([0.5, 0.5], 0.0), "mlba3": ([0.25, 0.5, 0.25], 3e-8)}
    for model, (want, atol) in expectations.items():
        settings = declared_models[model]["reductions"][0]["settings"]
        n_options = len(want)
        theta = {k: np.array([v], dtype=np.float32) for k, v in settings.items()}
        theta = MLBAFrontEnd(n_options).apply(theta, {}, 1)
        got = [float(theta[f"v{i}"][0]) for i in range(n_options)]
        assert got == pytest.approx(want, abs=atol), (model, got)


@pytest.mark.parametrize("model", ["mlba2", "mlba3"])
def test_neutral_front_end_does_not_depend_on_attributes(model):
    """With lambda1 = lambda2 = 0 and identical options the drift is just I0.

    This is the structural statement behind the reductions: the back end of
    the MLBA is the library's LBA, and the front end only sets drift rates.
    """
    n_options = int(model[-1])
    theta = {
        "m": np.array([5.0]),
        "lambda1": np.array([0.3]),
        "lambda2": np.array([0.7]),
        "beta": np.array([2.0]),
        "I0": np.array([4.25]),
        "s": np.array([1.0]),
    }
    for i in range(n_options):
        theta[f"P{i}"] = np.array([2.5])
        theta[f"Q{i}"] = np.array([7.5])
    theta = MLBAFrontEnd(n_options).apply(theta, {}, 1)
    for i in range(n_options):
        assert float(theta[f"v{i}"][0]) == pytest.approx(4.25, abs=1e-7)
