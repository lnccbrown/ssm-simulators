"""Numerical-error audit for the MTR simulator and its semi-analytic reference.

1. Quadrature error of the reference: residual 1 - sum(p) and change between
   default (1e-10/1e-8) and tight (1e-13/1e-12) tolerances, plus an
   independent fixed-grid (trapezoid, log-spaced 200k points) integration.
2. Monte Carlo error of the registered simulator: |MC - reference| in units of
   the binomial SE for every bridge cell (n = 400k each), including the most
   demanding cell (smallest DK, low-cost speed: DK_right = 0.055).
3. Deadline censoring: P(no response within the 2.5 s window) MC vs reference.
4. Truncated-normal sampler: sample mean vs closed form for v/sv = 0 .. 6.
Run from the repository root:
    MTR_BUNDLE_DIR=<bundle>/contribution .venv/bin/python -m tests.mtr_audits.numerical_error
(writes numerical_error.txt)
"""

import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
# ``bridge.py``/``reproduce.py`` are checker-only artifacts of the generated
# contribution bundle and are not part of the library; point MTR_BUNDLE_DIR at
# the bundle's ``contribution/`` directory to run this audit.
_BUNDLE = os.environ.get("MTR_BUNDLE_DIR")
if not _BUNDLE:
    raise SystemExit("Set MTR_BUNDLE_DIR to the bundle's contribution/ directory.")
sys.path.insert(0, _BUNDLE)
import bridge

from ssms import Simulator
from ssms.basic_simulators.mtr import sample_truncated_normal_rates
from tests import mtr_analytic as analytic

lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


PP = bridge.PAPER_PARAMS
hard = "exp1_low_cost_speed_lower_similarity_left_target"

log("1. Reference quadrature error")
for cond in (bridge.CONDITIONS[0], hard):
    th = bridge.mtr_theta(PP, cond)
    pd = analytic.response_probabilities(th, epsabs=1e-10, epsrel=1e-8)
    pt = analytic.response_probabilities(th, epsabs=1e-13, epsrel=1e-12)
    tg = np.concatenate([[0.0], np.logspace(-4, 4, 200_000)])
    dens = analytic.joint_densities(np.maximum(tg, 1e-12), th)
    pg = [np.trapezoid(dens[k], tg) for k in range(4)]
    pg = [pg[0], pg[1], pg[2] + pg[3]]
    log(
        f"  {cond}: residual(default)={pd['residual']:.2e} residual(tight)={pt['residual']:.2e}"
    )
    log(
        f"    max|default-tight|={max(abs(pd[k] - pt[k]) for k in range(3)):.2e}  max|default-grid|={max(abs(pd[k] - pg[k]) for k in range(3)):.2e}"
    )

log("\n2. Monte Carlo vs reference, n=400000 per cell (z = diff / binomial SE)")
sim = Simulator("mtr")
zs = []
for i, cond in enumerate(bridge.CONDITIONS):
    th = bridge.mtr_theta(PP, cond)
    ref = analytic.response_probabilities(th, epsabs=1e-10, epsrel=1e-8)
    out = sim.simulate(theta=th, n_samples=400_000, random_state=1000 + i)
    ch = np.asarray(out["choices"]).ravel()
    row = []
    for k in range(3):
        p = np.mean(ch == k)
        se = np.sqrt(ref[k] * (1 - ref[k]) / ch.size)
        z = (p - ref[k]) / se
        zs.append(z)
        row.append(f"p{k}: mc={p:.4f} ref={ref[k]:.4f} z={z:+.2f}")
    log(f"  {cond}: " + " | ".join(row))
zs = np.array(zs)
log(
    f"  max|z| = {np.max(np.abs(zs)):.2f} over {zs.size} comparisons; mean z^2 = {np.mean(zs**2):.2f}"
)

log("\n3. Response-window censoring (deadline 2.5 s incl. t0)")
simd = Simulator("mtr_deadline")
for cond in (
    bridge.CONDITIONS[0],
    hard,
    "exp1_low_cost_accuracy_higher_similarity_right_target",
):
    th = bridge.mtr_theta(PP, cond)
    ref = analytic.response_probabilities(
        th, upper=2.5 - th["t"], epsabs=1e-10, epsrel=1e-8
    )
    p_resp = ref[0] + ref[1] + ref[2]
    th["deadline"] = 2.5
    out = simd.simulate(theta=th, n_samples=400_000, random_state=7)
    om = np.mean(np.asarray(out["rts"]).ravel() == -999)
    se = np.sqrt((1 - p_resp) * p_resp / 400_000)
    log(
        f"  {cond}: omission mc={om:.5f} ref={1 - p_resp:.5f} z={(om - (1 - p_resp)) / se:+.2f}"
    )

log(
    "\n4. Truncated-normal sampler mean (n=1e6), closed form mean = v + sv*phi(v/sv)/Phi(v/sv)"
)
rng = np.random.default_rng(0)
for v in (0.0, 0.5, 1.0, 3.0, 6.0):
    u = rng.random(1_000_000)
    r = sample_truncated_normal_rates(u, np.array(v), np.array(1.0))
    exact = v + norm.pdf(v) / norm.cdf(v)
    sd_exact = np.sqrt(
        1 - v * norm.pdf(v) / norm.cdf(v) - (norm.pdf(v) / norm.cdf(v)) ** 2
    )
    log(
        f"  v={v}: mean={r.mean():.5f} exact={exact:.5f} z={(r.mean() - exact) / (sd_exact / 1e3):+.2f} min={r.min():.2e}"
    )

with open(os.path.join(HERE, "numerical_error.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
