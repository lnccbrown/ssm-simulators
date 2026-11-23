"""Numerical error of the analytic choice probabilities used by the paper grid analysis.

* Gauss-Legendre (t = x/(1-x)) at 100/200/400/1600 nodes over the full 3,840
  grid x 10 Table 3 sets: max |P_n - P_1600| and stability of every count.
* Adaptive scipy.quad (epsabs 1e-13) at the Fig. 7 points, the grid point with
  the smallest drift, and 40 random grid points x 10 sets.
* Smallest non-tie decision margin, to compare with the quadrature error.
* Usher et al. analysis: which parameter sets have exact three-way ties.

Writes results/quadrature_audit.json next to this script. Run from the
repository root with ``.venv/bin/python -B tests/mlba/audits/quadrature_audit.py``.
"""

import itertools
import json
import os

import numpy as np

from ssms.basic_simulators.mlba_analytic import (
    choice_probabilities,
    choice_probabilities_quad,
)
from ssms.basic_simulators.mlba_frontend import mlba_drifts

HERE = os.path.dirname(os.path.abspath(__file__))


class R:
    """Grid analysis of pp. 191-192 (the contribution's checker-only reproduce.py)."""

    X = (4.0, 6.0)
    Y = (6.0, 4.0)
    SETS = {
        "R1": (3.4, 6.0),
        "R2": (6.0, 3.4),
        "F1": (4.0, 5.4),
        "F2": (5.4, 4.0),
        "RF1": (3.7, 5.7),
        "RF2": (5.7, 3.7),
        "S1": (3.8, 6.2),
        "S2": (6.2, 3.8),
        "C1": (8.0, 2.0),
        "C2": (2.0, 8.0),
    }
    choice_probabilities = staticmethod(choice_probabilities)

    @staticmethod
    def grid():
        ms = np.arange(1, 11) * 0.5
        ls = np.arange(1, 9) / 10.0
        i0s = np.arange(5, 11).astype(float)
        return np.array(list(itertools.product(ms, ls, ls, i0s)))

    @staticmethod
    def set_probs(G, options, n_nodes=400):
        opts = np.asarray(options, dtype=float)
        k = len(G)
        P = np.tile(opts[:, 0], (k, 1))
        Q = np.tile(opts[:, 1], (k, 1))
        D = mlba_drifts(P, Q, G[:, 0], G[:, 1], G[:, 2], G[:, 3])
        return choice_probabilities(D, 1.0, 2.0, 1.0, n_nodes)

    @staticmethod
    def grid_results(n_nodes=400):
        G = R.grid()
        P = {k: R.set_probs(G, [R.X, R.Y, d], n_nodes) for k, d in R.SETS.items()}
        # Effects (pp. 180-181 definitions). Set 1 decoys sit near X (attraction,
        # similarity) or beyond Y (compromise: Y is the middle option in Set 1).
        margins = {
            "R": P["R1"][:, 0] - P["R2"][:, 0],
            "F": P["F1"][:, 0] - P["F2"][:, 0],
            "RF": P["RF1"][:, 0] - P["RF2"][:, 0],
            "S": P["S1"][:, 1] - P["S2"][:, 1],  # Y is focal when the decoy is near X
            "C": P["C1"][:, 1] - P["C2"][:, 1],  # Y is the compromise in Set 1
        }
        # By the P<->Q mirror symmetry of the stimuli, the second inequality of
        # each effect definition is the same number as the first.
        all5 = np.all(np.stack([v > 0 for v in margins.values()]), axis=0)
        regularity = (P["R1"][:, 0] > 0.5) & (P["R2"][:, 1] > 0.5)
        # Usher et al. (2008): compromise Set 1 {X, Y, (8, 2)}; Y is the compromise.
        pc = P["C1"]
        first = np.argmax(pc, axis=1)  # ties -> lowest index (X, Y, decoy order)
        extreme_first = first != 1
        masked = pc.copy()
        masked[np.arange(len(pc)), first] = -np.inf
        second = np.argmax(masked, axis=1)
        comp_second = second[extreme_first] == 1
        return {
            "G": G,
            "P": P,
            "margins": margins,
            "all5": all5,
            "regularity": regularity,
            "extreme_first": extreme_first,
            "comp_second": comp_second,
        }


def main():
    res = {}
    ref = R.grid_results(n_nodes=1600)
    G = ref["G"]
    for n in (100, 200, 400):
        g = R.grid_results(n_nodes=n)
        err = max(np.max(np.abs(g["P"][k] - ref["P"][k])) for k in R.SETS)
        res[f"max_abs_err_n{n}_vs_n1600"] = float(err)
        res[f"counts_n{n}"] = [
            int(g["all5"].sum()),
            int(g["regularity"].sum()),
            int(g["extreme_first"].sum()),
            int(g["comp_second"].sum()),
        ]
    res["counts_n1600"] = [
        int(ref["all5"].sum()),
        int(ref["regularity"].sum()),
        int(ref["extreme_first"].sum()),
        int(ref["comp_second"].sum()),
    ]
    res["max_defective_mass_1_minus_sum"] = float(
        max(np.max(1 - ref["P"][k].sum(1)) for k in R.SETS)
    )

    # adaptive-quadrature spot checks
    rng = np.random.default_rng(0)
    idx = list(rng.choice(len(G), 40, replace=False))
    dmin = []
    for k, dec in R.SETS.items():
        opts = np.array([R.X, R.Y, dec])
        D = mlba_drifts(
            np.tile(opts[:, 0], (len(G), 1)),
            np.tile(opts[:, 1], (len(G), 1)),
            G[:, 0],
            G[:, 1],
            G[:, 2],
            G[:, 3],
        )
        dmin.append((D.min(), k, int(np.argmin(D.min(1)))))
    lo = min(dmin)
    res["smallest_mean_drift_on_grid"] = {
        "drift": float(lo[0]),
        "set": lo[1],
        "params": G[lo[2]].tolist(),
    }
    worst = 0.0
    checks = [(int(i), k) for i in idx for k in R.SETS] + [(lo[2], lo[1])]
    for i, k in checks:
        opts = np.array([R.X, R.Y, R.SETS[k]])
        d = mlba_drifts(opts[:, 0], opts[:, 1], *G[i])[0]
        q = choice_probabilities_quad(d)
        worst = max(worst, float(np.max(np.abs(q - R.choice_probabilities(d)))))
    res["max_abs_err_n400_vs_scipy_quad"] = worst
    res["n_quad_checks"] = len(checks)

    # margins
    allm = np.concatenate([np.abs(v) for v in ref["margins"].values()])
    res["n_exact_ties_margin_lt_1e-9"] = int((allm < 1e-9).sum())
    res["smallest_nontie_effect_margin"] = float(allm[allm >= 1e-9].min())
    reg = np.concatenate(
        [np.abs(ref["P"]["R1"][:, 0] - 0.5), np.abs(ref["P"]["R2"][:, 1] - 0.5)]
    )
    res["smallest_regularity_margin"] = float(reg.min())
    pc = ref["P"]["C1"]
    srt = np.sort(pc, 1)
    res["smallest_nontie_usher_rank_gap"] = float(
        np.min(np.where(np.diff(srt, axis=1) < 1e-9, np.inf, np.diff(srt, axis=1)))
    )

    # Usher ties: three-way equal probabilities
    tie3 = (srt[:, 2] - srt[:, 0]) < 1e-9
    ef = ref["extreme_first"]
    res["usher_three_way_ties"] = int(tie3.sum())
    res["usher_three_way_tie_params_unique_m"] = sorted(set(G[tie3, 0].tolist()))
    res["usher_three_way_ties_lambda1_eq_lambda2"] = bool(
        np.all(G[tie3, 1] == G[tie3, 2])
    )
    cs = ref["comp_second"]
    ef_idx = np.where(ef)[0]
    tie_in_ef = tie3[ef_idx]
    res["usher_compromise_second_if_ties_counted"] = [
        int(cs.sum()),
        int(ef.sum()),
        100 * cs.mean(),
    ]
    res["usher_compromise_second_if_ties_not_compromise"] = [
        int((cs & ~tie_in_ef).sum()),
        int(ef.sum()),
        100 * (cs & ~tie_in_ef).sum() / ef.sum(),
    ]
    res["usher_ties_excluded_from_both"] = [
        int((cs & ~tie_in_ef).sum()),
        int((~tie_in_ef).sum()),
        100 * (cs & ~tie_in_ef).sum() / (~tie_in_ef).sum(),
    ]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "quadrature_audit.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
