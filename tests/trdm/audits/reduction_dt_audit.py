"""Audit: trdm_3 with the timer off vs the library's racing_diffusion_3.

The reduction declared in declarations.yaml is

    trdm_3 --{vT: 0, sT: 0, tT: 0}--> racing_diffusion_3, map {v0,v1,v2,A,b,t} -> same names

trdm_3 samples first-passage times exactly; racing_diffusion_3 integrates an
Euler scheme on a ``delta_t`` grid. This script runs both registered public
simulators (``ssms.basic_simulators.simulator.simulator``) and reports, per
library ``delta_t``, the difference in choice probabilities, mean RT, RT
quantiles and the two-sample KS statistic, so the discretisation error of the
library kernel is separated from the (exact) reduction.

Run (from the repository root):
    .venv/bin/python tests/trdm/audits/reduction_dt_audit.py
Writes tests/trdm/audits/reduction_dt_audit.json (about a minute; offline audit)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

from ssms.basic_simulators.simulator import simulator

HERE = pathlib.Path(__file__).resolve().parent

SETTINGS = {"vT": 0.0, "sT": 0.0, "tT": 0.0}
MAP = {"v0": "v0", "v1": "v1", "v2": "v2", "A": "A", "b": "b", "t": "t"}

POINTS = {
    "rd3_default": {"v0": 1.0, "v1": 1.0, "v2": 1.0, "A": 0.5, "b": 1.5, "t": 1e-3},
    "asymmetric": {"v0": 2.0, "v1": 1.0, "v2": 0.5, "A": 0.3, "b": 1.0, "t": 0.3},
    "slow_high_b": {"v0": 0.6, "v1": 0.3, "v2": 0.1, "A": 0.0, "b": 2.5, "t": 0.2},
}
QS = [0.1, 0.3, 0.5, 0.7, 0.9]


def run(n=40000, seed=7, dts=(1e-3, 3e-4, 1e-4)):
    report = {}
    for label, lib_theta in POINTS.items():
        ours = {k: lib_theta[MAP[k]] for k in MAP}
        ours.update(SETTINGS)
        a = simulator(theta=ours, model="trdm_3", n_samples=n, random_state=seed)
        rt_all_a = np.asarray(a["rts"]).ravel()
        rt_a = rt_all_a[rt_all_a != -999]  # omissions (decision time > max_t) excluded
        np.asarray(a["choices"]).ravel()
        entry = {
            "trdm_3": {
                "choice_p": a["choice_p"].ravel().tolist(),
                "mean_rt": float(rt_a.mean()),
                "quantiles": np.quantile(rt_a, QS).tolist(),
                "omission_p": float(a["omission_p"].ravel()[0]),
            }
        }
        for dt in dts:
            b = simulator(
                theta=lib_theta,
                model="racing_diffusion_3",
                n_samples=n,
                random_state=seed + 1,
                delta_t=dt,
                smooth_unif=True,
            )
            rt_b = np.asarray(b["rts"]).ravel()
            ok = rt_b != -999
            ks = stats.ks_2samp(rt_a, rt_b[ok])
            entry[f"racing_diffusion_3_dt{dt:g}"] = {
                "choice_p": b["choice_p"].ravel().tolist(),
                "omission_p": float(b["omission_p"].ravel()[0]),
                "mean_rt": float(rt_b[ok].mean()),
                "mean_rt_diff_lib_minus_trdm": float(rt_b[ok].mean() - rt_a.mean()),
                "quantile_diff_lib_minus_trdm": (
                    np.quantile(rt_b[ok], QS) - np.quantile(rt_a, QS)
                ).tolist(),
                "max_abs_choice_p_diff": float(
                    np.max(np.abs(b["choice_p"].ravel() - a["choice_p"].ravel()))
                ),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
        report[label] = entry
    return report


if __name__ == "__main__":
    rep = run()
    out = HERE / "reduction_dt_audit.json"
    out.write_text(json.dumps(rep, indent=2))
    for label, e in rep.items():
        print(
            label,
            "trdm_3 mean RT {:.4f} choice_p {}".format(
                e["trdm_3"]["mean_rt"], np.round(e["trdm_3"]["choice_p"], 4)
            ),
        )
        for k, v in e.items():
            if k == "trdm_3":
                continue
            print(
                "   %-28s dMeanRT %+.4f  dQ %s  max|dP| %.4f  KS %.4f (p=%.3g)"  # noqa: UP031
                % (
                    k,
                    v["mean_rt_diff_lib_minus_trdm"],
                    np.round(v["quantile_diff_lib_minus_trdm"], 4),
                    v["max_abs_choice_p_diff"],
                    v["ks_stat"],
                    v["ks_p"],
                )
            )
