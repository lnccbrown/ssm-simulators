"""Monte Carlo audit of the registered mlba2/mlba3 simulators against the
analytic LBA (choice probabilities, omission rate, RT defective CDF).

Conditions
* the 11 bridge conditions (Table 3 sets + binary {X, Y}) at the Fig. 7
  parameters (m = 5, lambda1 = .2, lambda2 = .4, I0 = 5, A = 1, chi = 2, s = 1);
* the four Table 6 best fits with the Table 5 example stimuli of the matching
  experiment (attributes / 10 as in the paper). The similarity fit
  (m = 32.02, I0 = .31) is the most demanding paper condition: negative rates
  are frequent, so the non-folded back end matters and some trials never end.

For each condition the registered simulator is run through bridge-style
conversion (library units, default sigma_noise) and compared with analytic
values computed in paper units with 1600 quadrature nodes. Also reported: the
choice probabilities the library's folded-rate ``lba3`` gives at the same mean
drifts (to quantify the back-end difference).

Writes results/simulator_audit.json next to this script (expensive: 400,000
trials per condition). Run from the repository root with
``.venv/bin/python -B tests/mlba/audits/simulator_audit.py``.
"""

import json
import os

import numpy as np
from scipy.stats import norm

from ssms.basic_simulators.mlba_analytic import choice_probabilities, lba_cdf, rt_cdf
from ssms.basic_simulators.mlba_frontend import mlba_drifts
from ssms.basic_simulators.mlba_units import paper_to_library
from ssms.basic_simulators.simulator import simulator

HERE = os.path.dirname(os.path.abspath(__file__))


class bridge:
    """Paper parameters and Table 3 / p. 192 choice sets (0 = X, 1 = Y, 2 = decoy)."""

    PAPER_PARAMS = {
        "m": 5.0,
        "lambda1": 0.2,
        "lambda2": 0.4,
        "I0": 5.0,
        "A": 1.0,
        "chi": 2.0,
        "s": 1.0,
    }
    X = (4.0, 6.0)
    Y = (6.0, 4.0)
    DECOYS = {
        "attraction_range_set1": (3.4, 6.0),
        "attraction_range_set2": (6.0, 3.4),
        "attraction_frequency_set1": (4.0, 5.4),
        "attraction_frequency_set2": (5.4, 4.0),
        "attraction_range_frequency_set1": (3.7, 5.7),
        "attraction_range_frequency_set2": (5.7, 3.7),
        "similarity_set1": (3.8, 6.2),
        "similarity_set2": (6.2, 3.8),
        "compromise_set1": (8.0, 2.0),
        "compromise_set2": (2.0, 8.0),
    }
    CONDITIONS = [*DECOYS, "binary_xy"]

    @staticmethod
    def options_for(condition):
        if condition == "binary_xy":
            return [bridge.X, bridge.Y]
        return [bridge.X, bridge.Y, bridge.DECOYS[condition]]


N = 400_000
FIG7 = dict(bridge.PAPER_PARAMS)
BASE = {"A": 1.0, "chi": 2.0, "s": 1.0}
TABLE6 = {
    "table6_attraction": (
        {"m": 1.68, "lambda1": 0.024, "lambda2": 0.001, "I0": 7.73},
        [(3.5, 6.5), (6.5, 3.5), (2.8, 6.5)],
    ),
    "table6_similarity": (
        {"m": 32.02, "lambda1": 0.325, "lambda2": 0.897, "I0": 0.31},
        [(3.5, 7.5), (5.5, 5.5), (3.3, 7.7)],
    ),
    "table6_compromise": (
        {"m": 29.48, "lambda1": 0.234, "lambda2": 0.204, "I0": 15.21},
        [(7.5, 3.5), (6.5, 4.5), (5.5, 5.5)],
    ),
    "table6_combined": (
        {"m": 22.30, "lambda1": 0.193, "lambda2": 0.278, "I0": 13.39},
        [(2.5, 4.5), (4.5, 2.5), (1.7, 4.5)],
    ),
}


def run(params, opts, seed):
    theta = paper_to_library(params, opts)
    out = simulator(
        model=f"mlba{len(opts)}", theta=theta, n_samples=N, random_state=seed
    )
    rts = np.asarray(out["rts"]).reshape(-1).astype(float)
    ch = np.asarray(out["choices"]).reshape(-1)
    return rts, ch, theta


def audit(name, params, opts, seed):
    rts, ch, theta = run(params, opts, seed)
    o = np.asarray(opts, float)
    d = mlba_drifts(
        o[:, 0],
        o[:, 1],
        params["m"],
        params["lambda1"],
        params["lambda2"],
        params["I0"],
        params.get("beta", 1.0),
    )[0]
    A, b, s = params["A"], params["chi"], params["s"]
    p = choice_probabilities(d, A, b, s, n_nodes=1600)
    resp = rts != -999.0
    phat = np.array([np.mean((ch == i) & resp) for i in range(len(opts))])
    se = np.sqrt(np.maximum(p * (1 - p), 1e-300) / N)
    z = (phat - p) / se
    om_exact = float(np.prod(norm.cdf(-d / s)))
    om_hat = float(np.mean(~resp))
    # the simulator also censors decision times > max_t (= 20, same in both units)
    om_maxt = float(np.prod(1.0 - lba_cdf(20.0, d, A, b, s)))
    om_z = (om_hat - om_maxt) / np.sqrt(max(om_maxt * (1 - om_maxt), 1e-300) / N)
    # RT defective CDF for the most likely option at 9 quantiles (paper time units;
    # the bridge's library-unit RTs are identical by scale covariance)
    k = int(np.argmax(p))
    rk = rts[resp & (ch == k)]
    qs = np.quantile(rk, np.linspace(0.1, 0.9, 9))
    emp = np.array([np.mean(resp & (ch == k) & (rts <= q)) for q in qs])
    ana = rt_cdf(qs, d, A, b, s, k, n_nodes=800)
    dkw99 = float(np.sqrt(np.log(2 / 0.01) / (2 * N)))
    # library lba3 / lba2 (folded |v|) at the same mean drifts, library units
    c = 0.1 / s
    lt = {"A": theta["A"], "b": theta["b"]}
    for i, di in enumerate(d):
        lt[f"v{i}"] = float(di * c)
    lo = simulator(model=f"lba{len(opts)}", theta=lt, n_samples=N, random_state=seed)
    lch = np.asarray(lo["choices"]).reshape(-1)
    pf = np.array([np.mean(lch == i) for i in range(len(opts))])
    return {
        "condition": name,
        "mean_drifts_paper_units": d.round(4).tolist(),
        "analytic_p": p.round(5).tolist(),
        "simulated_p": phat.round(5).tolist(),
        "max_abs_z": float(np.max(np.abs(z))),
        "omission_exact": om_exact,
        "omission_simulated": om_hat,
        "omission_exact_with_max_t_20": om_maxt,
        "omission_z": float(om_z) if om_maxt > 1e-12 else None,
        "rt_cdf_max_abs_diff": float(np.max(np.abs(emp - ana))),
        "dkw_99_bound": dkw99,
        "folded_lba_p_same_drifts": pf.round(5).tolist(),
        "max_abs_diff_mlba_vs_folded": float(np.max(np.abs(pf - p))),
    }


def main():
    out = []
    for j, c in enumerate(bridge.CONDITIONS):
        out.append(audit(c, FIG7, bridge.options_for(c), 100 + j))
    for j, (c, (pp, opts)) in enumerate(TABLE6.items()):
        out.append(audit(c, dict(BASE, **pp), opts, 200 + j))
    for r in out:
        print(json.dumps(r))
    z = max(r["max_abs_z"] for r in out)
    print("max |z| over", len(out), "conditions x options:", z)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "simulator_audit.json"), "w") as fh:
        json.dump(
            {"n_samples_per_condition": N, "conditions": out, "max_abs_z_overall": z},
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
