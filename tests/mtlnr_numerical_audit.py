"""Numerical audit for the MTLNR models (``mtlnr2_conf2`` / ``mtlnr2_conf3``).

Offline test-support program (not collected by pytest; slow: about 2e6
simulated trials per Figure 7 condition). Run from the repository root::

    python -m tests.mtlnr_numerical_audit
    python -m tests.mtlnr_numerical_audit --reproduce PATH/TO/reproduce.py

Section A needs the Table 2/3 per-participant estimates that lived in the
contribution's checker-only ``reproduce.py`` (not part of the library); it is
skipped unless that file is passed with ``--reproduce``. Sections B and C use
only the library and ``tests/mtlnr_reference.py``.

The simulator samples finishing times exactly (no time step), so its only
numerical error is Monte Carlo. This script measures, and writes to
``tests/mtlnr_numerical_audit_results.txt``:

A. reproduce.py quantities: spread over 10 seeds (MC SD) and the analytic
   value of the *same* quantity (participant average of sigma*sqrt(2(1-rho))
   and of log d), separating implementation/MC error from paper disagreement.
B. Figure 7 conditions (incl. rho = 0.9, the most demanding: rare categories
   at P ~ 1e-4): label probabilities from 2e6 simulated trials vs the closed
   form (SDT representation) vs quadrature of Eqs. 6-7; z-scores.
C. Quadrature accuracy of the Eq. 6-7 reference: closed form vs quad at two
   tolerances; and RT-quantile check: simulated conditional quantiles vs the
   quadrature CDF (Kolmogorov-type comparison at 10/50/90%).
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np

from ssms import OMISSION_SENTINEL
from ssms.basic_simulators.simulator import simulator
from tests import mtlnr_reference as R

HERE = os.path.dirname(os.path.abspath(__file__))

# Figure 7 of the paper (accumulator 0 = "new", 1 = "old"); t not stated -> 0.
PAPER_PARAMS = {"mu_old": -0.95, "mu_new": -0.63, "sigma": 0.55, "d1": 0.3, "d2": 0.6}
CONDITIONS = ["neg_corr", "independent", "pos_corr"]
CONDITION_RHO = {"neg_corr": -0.9, "independent": 0.0, "pos_corr": 0.9}
reproduce = None  # set by --reproduce (section A only)


def figure7_theta(condition):
    p = PAPER_PARAMS
    return {
        "mu0": p["mu_new"],
        "mu1": p["mu_old"],
        "sigma0": p["sigma"],
        "sigma1": p["sigma"],
        "rho": CONDITION_RHO[condition],
        "d1acc0": p["d1"],
        "d2acc0": p["d2"],
        "d1acc1": p["d1"],
        "d2acc1": p["d2"],
        "t": 0.0,
    }


def simulate_figure7(condition, n, seed):
    """rt, choice (+1 old / -1 new) and confidence (1 low .. 3 high)."""
    out = simulator(
        model="mtlnr2_conf3",
        theta=figure7_theta(condition),
        n_samples=n,
        random_state=seed,
    )
    labels = np.asarray(out["choices"]).ravel()
    winner, conf = labels % 2, labels // 2
    rts = np.asarray(out["rts"], dtype=np.float64).ravel()
    return {
        "rt": np.where(rts == OMISSION_SENTINEL, np.nan, rts),
        "choice": np.where(winner == 1, 1, -1),
        "confidence": conf + 1,
    }


PAPER = {
    "sdt_sd_new_low": 0.707,
    "sdt_sd_weak_low": 1.055,
    "sdt_sd_strong_low": 0.942,
    "sdt_sd_new_high": 0.700,
    "sdt_sd_weak_high": 0.908,
    "sdt_sd_strong_high": 0.890,
    "crit_new_high_medium": -0.647,
    "crit_new_medium_low": -0.172,
    "crit_old_high_medium": 0.55,
}


def analytic_table3():
    q = {}
    for cell, qid in reproduce.CELL_IDS.items():
        r = np.minimum(np.array(reproduce.RHO[cell]), reproduce.RHO_CAP)
        s = np.array(reproduce.SIGMA[cell])
        q[qid] = float(np.mean(s * np.sqrt(2 * (1 - r))))
    q["crit_new_high_medium"] = float(np.mean(np.log(reproduce.D["d1Old"])))
    q["crit_new_medium_low"] = float(np.mean(np.log(reproduce.D["d2Old"])))
    q["crit_old_high_medium"] = float(np.mean(-np.log(reproduce.D["d1New"])))
    return q


def main():
    lines = []
    out = lines.append

    if reproduce is None:
        out("A. Table 3 quantities: skipped (pass --reproduce PATH/reproduce.py)")
    else:
        section_a(out)
    _sections_bc(out, lines)


def section_a(out):
    out("A. Table 3 quantities: 10 seeds x 400k trials/cell x 66 cells")
    runs = [reproduce.quantities(seed) for seed in range(10)]
    ana = analytic_table3()
    out(
        f"{'id':24s} {'mean':>8s} {'MC sd':>8s} {'analytic':>9s} "
        f"{'sim-ana':>9s} {'paper':>7s} {'ana-paper':>9s}"
    )
    for k, paper_value in PAPER.items():
        v = np.array([r[k] for r in runs])
        out(
            f"{k:24s} {v.mean():8.4f} {v.std(ddof=1):8.4f} {ana[k]:9.4f} "
            f"{v.mean() - ana[k]:9.4f} {paper_value:7.3f} {ana[k] - paper_value:9.4f}"
        )
    # RHO cap sensitivity: participant 6 NewHigh at 0.999 instead of 0.995
    r = np.minimum(np.array(reproduce.RHO["NewHigh"]), 0.999)
    s = np.array(reproduce.SIGMA["NewHigh"])
    out(
        f"sdt_sd_new_high analytic with rho cap 0.999: "
        f"{np.mean(s * np.sqrt(2 * (1 - r))):.4f} (cap 0.995: {ana['sdt_sd_new_high']:.4f})"
    )


def _sections_bc(out, lines):
    out("")
    out("B. Figure 7 label probabilities (labels winner+2*conf; 0=new,1=old)")
    n = 2_000_000
    for c in CONDITIONS:
        th = figure7_theta(c)
        mu = [th["mu0"], th["mu1"]]
        sg = [th["sigma0"], th["sigma1"]]
        d = [[th["d1acc0"], th["d2acc0"]], [th["d1acc1"], th["d2acc1"]]]
        p_cf = R.label_probs_sdt(mu, sg, th["rho"], d)
        p_q = np.array(
            [
                R.joint_prob_quad(l % 2, l // 2, mu, sg, th["rho"], d)[0]
                for l in range(6)
            ]
        )
        o = simulate_figure7(c, n, 123)
        lab = (o["choice"] == 1).astype(int) + 2 * (o["confidence"] - 1)
        p_s = np.bincount(lab, minlength=6) / n
        se = np.sqrt(np.maximum(p_cf * (1 - p_cf), 1e-300) / n)
        zs = np.where(p_cf > 0, (p_s - p_cf) / se, 0.0)
        out(f"{c}: rho={th['rho']}")
        out(f"  closed form : {np.array2string(p_cf, precision=6)}")
        out(f"  |cf - quad| max: {np.abs(p_cf - p_q).max():.2e}")
        out(f"  simulated   : {np.array2string(p_s, precision=6)}")
        out(
            f"  z (sim vs cf): {np.array2string(zs, precision=2)}  max|z|={np.abs(zs).max():.2f}"
        )
        out(
            f"  P(old)={p_cf[1::2].sum():.4f}  P(high conf)={p_cf[4:].sum():.4f}"
            f"  P(low conf)={p_cf[:2].sum():.4f}"
        )

    out("")
    out("C. Quadrature reference accuracy and RT quantiles (pos_corr = hardest)")
    for c in CONDITIONS:
        th = figure7_theta(c)
        mu = [th["mu0"], th["mu1"]]
        sg = [th["sigma0"], th["sigma1"]]
        d = [[th["d1acc0"], th["d2acc0"]], [th["d1acc1"], th["d2acc1"]]]
        p_cf = R.label_probs_sdt(mu, sg, th["rho"], d)
        for eps in (1e-8, 1e-12):
            p_q = np.array(
                [
                    R.joint_prob_quad(l % 2, l // 2, mu, sg, th["rho"], d, epsabs=eps)[
                        0
                    ]
                    for l in range(6)
                ]
            )
            out(
                f"  {c} quad epsabs={eps:g}: max|quad - closed form| = "
                f"{np.abs(p_q - p_cf).max():.2e}"
            )
        o = simulate_figure7(c, n, 7)
        lab = (o["choice"] == 1).astype(int) + 2 * (o["confidence"] - 1)
        for l in range(6):
            sel = o["rt"][lab == l]
            if sel.size < 2000:
                out(f"  {c} label {l}: {sel.size} samples, quantiles skipped")
                continue
            qs = np.quantile(sel, [0.1, 0.5, 0.9])
            F = [R.conditional_cdf(x, l % 2, l // 2, mu, sg, th["rho"], d) for x in qs]
            dev = np.array(F) - np.array([0.1, 0.5, 0.9])
            se = np.sqrt(np.array([0.09, 0.25, 0.09]) / sel.size)
            out(
                f"  {c} label {l}: n={sel.size}, sim q10/50/90={np.round(qs, 4)}, "
                f"F_ref(q)-p={np.round(dev, 5)}, max|z|={np.abs(dev / se).max():.2f}"
            )

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "numerical_audit_results.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reproduce", help="path to the contribution's reproduce.py")
    args = parser.parse_args()
    if args.reproduce:
        spec = importlib.util.spec_from_file_location("reproduce", args.reproduce)
        reproduce = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reproduce)
    main()
