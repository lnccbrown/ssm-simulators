"""Audit of the Table 8 reproduction (response-signal SAT function).

1. Monte Carlo error: d'(lag) from the registered simulator (reproduce.sat_function)
   vs the exact d'(lag) from the closed-form race integrals (Eqs. 1-3,
   ssms.support_utils.trdm_analytic) at every lag.
2. Fit sensitivity: shifted-exponential parameters fitted to the *exact*
   d'(lag) on several lag grids (the paper states the range 0-1.3 s but not the
   spacing; Fig. 8's axis spans about .35-1.3 s), so the fit/grid contribution
   is separated from Monte Carlo error.

Run (from the repository root):
    TRDM_BUNDLE_DIR=<bundle>/contribution .venv/bin/python tests/trdm/audits/sat_audit.py
Writes tests/trdm/audits/sat_audit.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
# bridge.py and reproduce.py are the paper-reproduction scripts of the original
# contribution bundle; they are not part of the library. Point TRDM_BUNDLE_DIR at
# the bundle's contribution/ directory to run this audit.
_BUNDLE = os.environ.get("TRDM_BUNDLE_DIR")
if not _BUNDLE:
    raise SystemExit(
        "set TRDM_BUNDLE_DIR to the directory containing bridge.py and reproduce.py"
    )
sys.path.insert(0, str(pathlib.Path(_BUNDLE).resolve()))

import bridge
import reproduce

from ssms.support_utils.trdm_analytic import Wald, race_probabilities


def exact_dprime(paper, lags):
    gp = paper["gamma_E_mu"] + paper["gamma_E_zeta"] / 2
    gm = paper["gamma_E_mu"] - paper["gamma_E_zeta"] / 2
    a1 = paper["alpha_E_mu"] + paper["alpha_E_zeta"] / 2
    a2 = paper["alpha_E_mu"] - paper["alpha_E_zeta"] / 2
    sE, tE = paper["sigma_E"], paper["tau_E"]
    timer = Wald(paper["gamma_T"], paper["sigma_T"], paper["alpha_T"], paper["tau_T"])
    out = []
    for L in lags:
        p = []
        for v0, v1 in ((gp, gm), (gm, gp)):
            ev = [Wald(v0, sE, a1, tE), Wald(v1, sE, a2, tE)]
            if L <= min(tE, paper["tau_T"]):
                p.append(0.5)
                continue
            r = race_probabilities(ev, timer, upper=float(L))
            p.append(r["evidence"][0] + 0.5 * r["timer"] + 0.5 * r["none"])
        out.append(norm.ppf(p[0]) - norm.ppf(p[1]))
    return np.array(out)


GRIDS = {
    "0-1.3 step .05 (reproduce.py)": np.round(np.arange(0.0, 1.3 + 1e-9, 0.05), 4),
    "0-1.3 step .01": np.round(np.arange(0.0, 1.3 + 1e-9, 0.01), 4),
    "0-1.3 step .1": np.round(np.arange(0.0, 1.3 + 1e-9, 0.1), 4),
    ".35-1.3 step .05": np.round(np.arange(0.35, 1.3 + 1e-9, 0.05), 4),
    ".4-1.3 step .1": np.round(np.arange(0.4, 1.3 + 1e-9, 0.1), 4),
    ".5-1.3 step .1": np.round(np.arange(0.5, 1.3 + 1e-9, 0.1), 4),
}


def main():
    rep = {}
    for coh in ("5", "10", "40"):
        paper = bridge.PAPER_PARAMS[f"explicit_{coh}"]
        lags = reproduce.LAGS
        mc = reproduce.sat_function(paper, reproduce.N_SAT, 17)
        ex = exact_dprime(paper, lags)
        entry = {
            "lags": lags.tolist(),
            "dprime_exact": ex.tolist(),
            "dprime_mc": mc.tolist(),
            "max_abs_mc_minus_exact": float(np.max(np.abs(mc - ex))),
            "fit_mc_0-1.3_step.05": dict(
                zip(
                    ("delta", "lambda", "beta"),
                    map(float, reproduce.fit_shifted_exponential(lags, mc)),
                )
            ),
            "fits_exact": {},
        }
        for name, grid in GRIDS.items():
            exg = exact_dprime(paper, grid)
            entry["fits_exact"][name] = dict(
                zip(
                    ("delta", "lambda", "beta"),
                    map(float, reproduce.fit_shifted_exponential(grid, exg)),
                )
            )
        rep[coh] = entry
        print(
            f"coherence {coh}%: max|d'_MC - d'_exact| = {entry['max_abs_mc_minus_exact']:.4f}"
        )
        print(
            "   d' exact at lags .35,.4,.45,.5,.6,.8,1.3:",
            np.round(exact_dprime(paper, [0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.3]), 3),
        )
        print(
            "   MC fit (reproduce):",
            {k: round(v, 3) for k, v in entry["fit_mc_0-1.3_step.05"].items()},
        )
        for name, f in entry["fits_exact"].items():
            print(f"   exact fit {name:32s}", {k: round(v, 3) for k, v in f.items()})
    (HERE / "sat_audit.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
