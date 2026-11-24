"""Mean decision times (t=0, RT<2.5) per cost x emphasis x response for A in {0.5,1,2}; used to calibrate t0. Run from the repository root: .venv/bin/python -m tests.mtr_audits.rt_by_condition."""

import itertools

import numpy as np

from ssms import Simulator
from tests.mtr_audits.exp1_params import DK, B, V

sim = Simulator("mtr")


def cell(cost, emph, sim_, side, A=1.0, svc=1.0, t=0.0):
    dkl, dkr = DK[(cost, emph)]
    bl, br = B[(cost, emph)]
    m, mm = V[(cost, emph, sim_)]
    v0, v1, s0, s1 = (m, mm, svc, 1.0) if side == "left" else (mm, m, 1.0, svc)
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
        "t": t,
    }


for A in [0.5, 1.0, 2.0]:
    for cost in ("low", "high"):
        for emph in ("speed", "accuracy"):
            rts = {0: [], 1: [], 2: []}
            for s_, side in itertools.product(("lower", "higher"), ("left", "right")):
                o = sim.simulate(
                    theta=cell(cost, emph, s_, side, A=A),
                    n_samples=100000,
                    random_state=3,
                )
                c = o["choices"].ravel()
                r = o["rts"].ravel()
                for k in range(3):
                    rts[k].append(r[(c == k) & (r < 2.5)])
            print(
                A,
                cost,
                emph,
                " ".join(f"{np.mean(np.concatenate(rts[k])):.3f}" for k in range(3)),
                "omit<2.5:",
                flush=True,
            )
