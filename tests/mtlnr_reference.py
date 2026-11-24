"""Analytic reference for the MTLNR (verification only, not used to simulate).

Test support for ``tests/test_mtlnr.py`` and ``tests/mtlnr_numerical_audit.py``.

Two independent routes to the model's predictions:

1. Paper Eqs. 4, 6, 7: defective density of (winner, confidence) at decision
   time ``dt``: ``f(dt | mu_w, sigma_w) * [S(lo) - S(hi)]`` where ``S`` is the
   conditional Lognormal survivor of the loser given the winner finished at
   ``dt`` (conditional log-mean ``mu_l + rho * sigma_l/sigma_w * (ln dt - mu_w)``,
   log-SD ``sqrt(1 - rho^2) * sigma_l``), evaluated at ``dt / d`` boundaries.
2. The paper's signal-detection representation (p. 11):
   ``X = log T_new - log T_old ~ N(mu_new - mu_old,
   sigma_new^2 + sigma_old^2 - 2 rho sigma_new sigma_old)``; old response iff
   ``X > 0``; criteria at ``log d_k(old acc)`` (new side) and
   ``-log d_k(new acc)`` (old side).

Accumulator 0 = "new", 1 = "old" throughout; labels = winner + 2 * conf.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm


def _bins(d_loser):
    """Loser-time boundaries (as multiples of dt) per confidence level.

    conf = number of d with d * T > dt.  Level c (0 = lowest) is
    dt/d_(K-c+1) ... ; returned as list of (lo_mult, hi_mult) for T/dt.
    """
    d = np.sort(np.asarray(d_loser, dtype=float))[::-1]  # descending: d_K..d_1
    # multiples m = 1/d ascending; T/dt in (1, 1/d_K] -> conf 0, ...
    mult = [1.0] + [np.inf if x == 0 else 1.0 / x for x in d] + [np.inf]
    return [(mult[c], mult[c + 1]) for c in range(len(d) + 1)]


def joint_density(dt, winner, conf, mu, sigma, rho, d):
    """Defective density of decision time ``dt`` for (winner, conf), Eqs. 6-7."""
    dt = np.asarray(dt, dtype=float)
    w, l = winner, 1 - winner
    lo, hi = _bins(d[l])[conf]
    ldt = np.log(dt)
    f = norm.pdf((ldt - mu[w]) / sigma[w]) / (dt * sigma[w])
    cm = mu[l] + rho * sigma[l] / sigma[w] * (ldt - mu[w])
    cs = np.sqrt(1.0 - rho**2) * sigma[l]

    def surv(mult):
        if np.isinf(mult):
            return 0.0
        return norm.sf((ldt + np.log(mult) - cm) / cs)

    return f * (surv(lo) - surv(hi))


def joint_prob_quad(winner, conf, mu, sigma, rho, d, epsabs=1e-12):
    """P(winner, conf) by quadrature of ``joint_density`` over log dt."""

    def g(u):
        x = np.exp(u)
        return joint_density(x, winner, conf, mu, sigma, rho, d) * x

    centre = mu[winner]
    span = 12.0 * max(sigma)
    val, err = integrate.quad(
        g, centre - span, centre + span, limit=400, epsabs=epsabs, epsrel=1e-12
    )
    return val, err


def label_probs_sdt(mu, sigma, rho, d):
    """Closed-form P(label) via the SDT representation; labels winner+2*conf."""
    m = mu[0] - mu[1]
    s = np.sqrt(sigma[0] ** 2 + sigma[1] ** 2 - 2 * rho * sigma[0] * sigma[1])
    K = len(d[0])
    probs = np.zeros(2 * (K + 1))
    # new response (winner 0) iff X < 0; loser = old acc (1); conf counts
    # d1(old) with X < log d.
    for w in (0, 1):
        l = 1 - w
        for c, (lo, hi) in enumerate(_bins(d[l])):
            # T_l/T_w in (lo, hi]  <=>  |X| in (log lo, log hi]
            a, b = np.log(lo), np.log(hi)
            if w == 1:  # X = log T_new - log T_old > 0
                p = norm.cdf((b - m) / s) - norm.cdf((a - m) / s)
            else:  # -X in (a, b]
                p = norm.cdf((-a - m) / s) - norm.cdf((-b - m) / s)
            probs[w + 2 * c] = p
    return probs


def conditional_cdf(x, winner, conf, mu, sigma, rho, d):
    """P(dt <= x | winner, conf) by quadrature (used for quantile checks)."""
    tot, _ = joint_prob_quad(winner, conf, mu, sigma, rho, d)

    def g(u):
        e = np.exp(u)
        return joint_density(e, winner, conf, mu, sigma, rho, d) * e

    lo = mu[winner] - 12.0 * max(sigma)
    val, _ = integrate.quad(g, lo, np.log(x), limit=400, epsabs=1e-12, epsrel=1e-12)
    return val / tot
