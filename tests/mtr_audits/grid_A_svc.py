"""Grid of p(dont know) and p(correct) over unprinted A and sv_c (analytic). Run from the repository root: .venv/bin/python -m tests.mtr_audits.grid_A_svc. Output in grid_A_svc.txt."""

import itertools

import numpy as np

from tests import mtr_analytic as analytic
from tests.mtr_audits.exp1_params import DK, B, V


def cell(cost, emph, sim, side, A, svc):
    dkl, dkr = DK[(cost, emph)]
    bl, br = B[(cost, emph)]
    m, mm = V[(cost, emph, sim)]
    if side == "left":
        v0, v1, s0, s1 = m, mm, svc, 1.0
    else:
        v0, v1, s0, s1 = mm, m, 1.0, svc
    return {
        "A": A,
        "B0": bl,
        "B1": br,
        "D0": 1 - dkl,
        "D1": 1 - dkr,
        "v0": v0,
        "v1": v1,
        "sv0": s0,
        "sv1": s1,
    }


def summary(A, svc):
    res = {}
    for cost in ("low", "high"):
        P = []
        for emph, sim, side in itertools.product(
            ("accuracy", "speed"), ("lower", "higher"), ("left", "right")
        ):
            p = analytic.response_probabilities(
                cell(cost, emph, sim, side, A, svc), epsabs=1e-9, epsrel=1e-7
            )
            P.append((emph, sim, side, p[0], p[1], p[2]))
        res[cost] = P
    return res


for A in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]:
    for svc in [0.6, 0.8, 1.0, 1.2, 1.5]:
        r = summary(A, svc)
        dk = {c: np.mean([x[5] for x in r[c]]) for c in r}
        corr = {c: np.mean([x[3] if x[2] == "left" else x[4] for x in r[c]]) for c in r}
        print(
            f"A={A} svc={svc}  pDK low={dk['low']:.3f} high={dk['high']:.3f} diff={dk['high'] - dk['low']:.3f}  pCorrect low={corr['low']:.3f} high={corr['high']:.3f}",
            flush=True,
        )
