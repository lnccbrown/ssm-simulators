"""Audit: timer shares from the registered simulator vs exact quadrature.

reproduce.py measures P(timer wins) with two same-seed runs of the registered
``trdm_2_guess`` (g = 1 vs g = 0). Here the same probabilities are computed by
adaptive quadrature of the closed-form race integral (Eqs. 1-3,
``ssms.support_utils.trdm_analytic``), which has no Monte Carlo error, so the MC error of the
reported numbers is measured, not assumed. Paper values are listed for
reference.

Run (from the repository root):
    TRDM_BUNDLE_DIR=<bundle>/contribution .venv/bin/python tests/trdm/audits/timer_share_audit.py
Writes tests/trdm/audits/timer_share_audit.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

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

PAPER = {
    "timer_share_perceptual_speed": 0.27,
    "timer_share_perceptual_neutral": 0.20,
    "timer_share_perceptual_accuracy": 0.15,
    "timer_share_lexical_speed": 0.21,
    "timer_share_lexical_accuracy": 0.07,
    "timer_share_explicit_0": 0.22,
    "timer_share_explicit_5": 0.21,
    "timer_share_explicit_10": 0.17,
    "timer_share_explicit_40": 0.09,
    "timer_share_implicit_0": 0.41,
    "timer_share_implicit_3.2": 0.39,
    "timer_share_implicit_6.4": 0.31,
    "timer_share_implicit_12.8": 0.17,
    "timer_share_implicit_25.6": 0.04,
    "timer_share_implicit_51.2": 0.005,
    "timer_share_gonogo": 0.21,
    "timer_share_mvm_speed": 0.36,
    "timer_share_mvm_accuracy": 0.19,
}


def exact_share_theta(th):
    """Exact P(timer wins) for trdm_2 parameters in library units (s = 1)."""
    ev = [
        Wald(th["v0"], 1.0, th["b0"], th["t"]),
        Wald(th["v1"], 1.0, th["b1"], th["t"]),
    ]
    tm = Wald(th["vT"], th["sT"], 1.0, th["tT"])
    return race_probabilities(ev, tm)["timer"]


def exact_share(paper, cond, both=True):
    vals = [exact_share_theta(reproduce._theta(paper, cond))]
    if both:
        vals.append(exact_share_theta(reproduce._theta(paper, cond, mirrored=True)))
    return float(np.mean(vals))


def exact_all():
    P = bridge.PAPER_PARAMS
    q = {}
    for e in ("speed", "neutral", "accuracy"):
        q[f"timer_share_perceptual_{e}"] = exact_share(
            P[f"perceptual_{e}"], f"perceptual_{e}"
        )
    for e in ("speed", "accuracy"):
        q[f"timer_share_lexical_{e}"] = float(
            np.mean(
                [
                    exact_share(P[f"lexical_{e}_{c}"], f"lexical_{e}_{c}")
                    for c in ("HF", "LF", "VLF", "NW")
                ]
            )
        )
    for c in ("0", "5", "10", "40"):
        q[f"timer_share_explicit_{c}"] = exact_share(
            P[f"explicit_{c}"], f"explicit_{c}"
        )
    for c in ("0", "3.2", "6.4", "12.8", "25.6", "51.2"):
        q[f"timer_share_implicit_{c}"] = exact_share(
            P[f"implicit_{c}"], f"implicit_{c}"
        )
    go = exact_share(P["gonogo_go"], "gonogo_go", both=False)
    nogo = exact_share(P["gonogo_nogo"], "gonogo_nogo", both=False)
    q["timer_share_gonogo"] = (8 * go + nogo) / 9
    q["timer_share_mvm_speed"] = exact_share(reproduce.MVM_SPEED, "mvm_speed")
    q["timer_share_mvm_accuracy"] = exact_share(P["mvm_accuracy"], "mvm_accuracy")
    return q


def main():
    exact = exact_all()
    mc = reproduce.quantities(seed=0)
    rep = {}
    print(f"{'quantity':34s} {'exact':>8s} {'MC':>8s} {'MC-exact':>9s} {'paper':>7s}")
    for k, v in exact.items():
        rep[k] = {
            "exact": v,
            "mc_reproduce": mc[k],
            "mc_minus_exact": mc[k] - v,
            "paper": PAPER[k],
        }
        print(f"{k:34s} {v:8.4f} {mc[k]:8.4f} {mc[k] - v:+9.4f} {PAPER[k]:7.3f}")
    (HERE / "timer_share_audit.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
