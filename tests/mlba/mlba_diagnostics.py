"""Offline MLBA numerical/convergence audit (not collected by pytest).

Prints, for the paper's qualitative-analysis parameters:

* the mean drift rates and choice probabilities of every reported choice set,
  analytically and by simulation, next to the paper's printed values;
* the convergence of the analytic quadrature;
* diagnostics for the two declared reductions.

Run from the repository root with ``python -B tests/mlba/mlba_diagnostics.py``
(500k-trial simulations per choice set; takes a few minutes).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlba_reference import (
    CHOICE_SETS,
    CONDITIONS,
    MLBA_SPEC,
    PAPER_PARAMS,
    choice_probabilities,
    simulate_paper,
)

from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder
from ssms.transforms import MLBAFrontEnd
from ssms.transforms.simulation.mlba_front_end import drift_rates

A, B, S = 1.0, 2.0, 1.0
N_SIM = 500_000

PAPER_VALUES = {
    "attraction_range_set1": 0.58,
    "attraction_range_frequency_set1": 0.52,
    "attraction_frequency_set1": 0.45,
    "similarity_set1": 0.41,
    "compromise_set2": 0.62,
    "binary_xy": 0.50,
}


def condition_drifts(condition: str) -> np.ndarray:
    options = CHOICE_SETS[condition]
    P = np.array([[p for p, _ in options]])
    Q = np.array([[q for _, q in options]])
    pp = PAPER_PARAMS
    return drift_rates(
        P, Q, pp["m"], pp["lambda1"], pp["lambda2"], pp["I0"], pp["beta"]
    )[0]


def choice_shares() -> None:
    print("== choice shares at m=5, lambda1=.2, lambda2=.4, I0=5, A=1, chi=2, s=1 ==")
    print(
        f"{'condition':34s} {'drifts':26s} {'p(analytic)':26s} "
        f"{'rel(analytic)':>13s} {'rel(sim)':>9s} {'se':>7s} {'paper':>6s}"
    )
    for condition in CONDITIONS:
        d = condition_drifts(condition)
        p = choice_probabilities(d, A, B, S, folded=True)
        rel_a = p[0] / (p[0] + p[1])

        out = simulate_paper(PAPER_PARAMS, condition, N_SIM, 0)
        n_x = int((out["choice"] == 1).sum())
        n_y = int((out["choice"] == 2).sum())
        rel_s = n_x / (n_x + n_y)
        se = float(np.sqrt(rel_a * (1 - rel_a) / (n_x + n_y)))

        print(
            f"{condition:34s} {np.array2string(d, precision=4):26s} "
            f"{np.array2string(p, precision=4):26s} {rel_a:13.4f} {rel_s:9.4f} "
            f"{se:7.4f} {PAPER_VALUES[condition]:6.2f}"
        )


def quadrature_convergence() -> None:
    print("\n== analytic quadrature: max |P(i) - P_refined(i)| ==")
    print(
        f"{'condition':34s} {'sum':>12s} {'vs 4x grid':>12s} {'vs 4x range':>12s} {'vs h=0.1':>10s}"
    )
    for condition in CONDITIONS:
        d = condition_drifts(condition)
        base = choice_probabilities(d, A, B, S, True, 400.0, 400_001)
        finer = choice_probabilities(d, A, B, S, True, 400.0, 1_600_001)
        wider = choice_probabilities(d, A, B, S, True, 1600.0, 1_600_001)
        coarse = choice_probabilities(d, A, B, S, True, 400.0, 4_001)
        print(
            f"{condition:34s} {base.sum():12.10f} {np.abs(base - finer).max():12.2e} "
            f"{np.abs(base - wider).max():12.2e} {np.abs(base - coarse).max():10.2e}"
        )


def drift_convention() -> None:
    print("\n== |N(d,s)| (cssm) vs N(d,s) (textbook LBA): max |dP(i)| ==")
    for condition in CONDITIONS:
        d = condition_drifts(condition)
        folded = choice_probabilities(d, A, B, S, True)
        unfolded = choice_probabilities(d, A, B, S, False)
        print(f"{condition:34s} {np.abs(folded - unfolded).max():.3e}")
    d = np.array([0.25, 0.5, 0.25])
    folded = choice_probabilities(d, 0.3, 0.5, 0.5, True)
    unfolded = choice_probabilities(d, 0.3, 0.5, 0.5, False)
    print(
        f"{'(sensitivity: d=(.25,.5,.25), s=.5)':34s} "
        f"{np.abs(folded - unfolded).max():.3e}"
    )


def reductions() -> None:
    declarations = MLBA_SPEC

    print("\n== declared reductions ==")
    for model in declarations["models"]:
        for reduction in model["reductions"]:
            settings = reduction["settings"]
            n_options = int(model["name"][-1])
            theta32 = {k: np.array([v], dtype=np.float32) for k, v in settings.items()}
            front = MLBAFrontEnd(n_options).apply(theta32, {}, 1)
            got = np.array([front[f"v{i}"][0] for i in range(n_options)])

            target_cfg = ModelConfigBuilder.from_model(reduction["to"])
            target_theta = dict(zip(target_cfg["params"], target_cfg["default_params"]))
            want = np.array(
                [target_theta[f"v{i}"] for i in range(n_options)], dtype=np.float32
            )
            for source, target in reduction["map"].items():
                target_theta[target] = settings[source]

            np.random.seed(20140137)
            tgt = simulator(
                theta=target_theta,
                model=reduction["to"],
                n_samples=200_000,
                random_state=20140137,
            )
            np.random.seed(20140137)
            src = simulator(
                theta=dict(settings),
                model=model["name"],
                n_samples=200_000,
                random_state=20140137,
            )
            print(
                f"{model['name']} -> {reduction['to']} ({reduction['expect']}):\n"
                f"  drift target {want}, obtained {got}, max|d| = "
                f"{np.abs(got - want).max():.2e}\n"
                f"  max|rt difference| = {np.abs(src['rts'] - tgt['rts']).max():.2e}, "
                f"choice mismatch = {(src['choices'] != tgt['choices']).mean():.2e}\n"
                f"  choice_p source {src['choice_p'].ravel().round(5)} "
                f"target {tgt['choice_p'].ravel().round(5)}"
            )


if __name__ == "__main__":
    choice_shares()
    quadrature_convergence()
    drift_convention()
    reductions()
