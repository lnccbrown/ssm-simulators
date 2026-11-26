"""Semi-analytic MTR response probabilities and densities (independent reference).

Derived from the model definition rather than from the simulator. For
accumulator ``j`` with start ``K ~ U(0, A)`` and rate ``R ~ N(v, sv)``
truncated to ``R > 0``, the probability that its evidence is still below a
level ``c`` at time ``t`` is

    G_j(t; c) = P(K + R t < c)
              = 1/(A P(R>0)) * int_0^{min(c,A)} [Phi_R((c-u)/t) - Phi_R(0)] du,

whose integral is closed form (``int Phi(z) dz = z Phi(z) + phi(z)``). With
``f_i(t) = -dG_i(t; b_i)/dt`` the finishing-time density of accumulator ``i``
(standard LBA density divided by ``P(R>0)``; valid because ``b_i > A``),

    p(definitive i, t) = f_i(t) * G_j(t; d_j)
    p(don't know via i, t) = f_i(t) * [G_j(t; b_j) - G_j(t; d_j)].

Time integrals are done with adaptive quadrature (``scipy.integrate.quad``);
the residual ``1 - sum(p)`` is reported so the quadrature error is visible.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm


def _zPhi_phi(z):
    return z * norm.cdf(z) + norm.pdf(z)


def below_level(t, c, A, v, sv):
    """``P(K + R t < c)``, ``K~U(0,A)``, ``R~N(v,sv)`` truncated to ``R>0``."""
    t = np.asarray(t, dtype=float)
    pos = norm.cdf(v / sv)
    if c <= 0:
        return np.zeros_like(t)
    if A == 0:
        # point start at 0: P(R < c/t)
        return (norm.cdf((c / t - v) / sv) - norm.cdf(-v / sv)) / pos
    m = min(c, A)
    s = t * sv
    z0 = (c - v * t) / s  # u = 0
    zm = (c - m - v * t) / s  # u = m
    integral = s * (_zPhi_phi(z0) - _zPhi_phi(zm))  # int_0^m Phi(((c-u)/t-v)/sv) du
    integral -= m * norm.cdf(-v / sv)
    return integral / (A * pos)


def finish_density(t, b, A, v, sv):
    """Density of the time at which an accumulator reaches ``b`` (``b > A``)."""
    t = np.asarray(t, dtype=float)
    pos = norm.cdf(v / sv)
    if A == 0:
        x = b / t
        return norm.pdf((x - v) / sv) / sv * b / t**2 / pos
    s = t * sv
    z1 = (b - A - t * v) / s
    z2 = (b - t * v) / s
    dens = (
        -v * norm.cdf(z1) + sv * norm.pdf(z1) + v * norm.cdf(z2) - sv * norm.pdf(z2)
    ) / A
    return dens / pos


def _unpack(params):
    A = float(params["A"])
    b = [A + float(params["B0"]), A + float(params["B1"])]
    d = [float(params["D0"]) * b[0], float(params["D1"]) * b[1]]
    v = [float(params["v0"]), float(params["v1"])]
    sv = [float(params["sv0"]), float(params["sv1"])]
    return A, b, d, v, sv


def joint_densities(t, params):
    """Defective decision-time densities for (def0, def1, dk via 0, dk via 1)."""
    A, b, d, v, sv = _unpack(params)
    t = np.asarray(t, dtype=float)
    out = []
    f = [finish_density(t, b[i], A, v[i], sv[i]) for i in range(2)]
    for i in range(2):
        j = 1 - i
        g_d = below_level(t, d[j], A, v[j], sv[j])
        out.append(f[i] * g_d)
    for i in range(2):
        j = 1 - i
        g_b = below_level(t, b[j], A, v[j], sv[j])
        g_d = below_level(t, d[j], A, v[j], sv[j])
        out.append(f[i] * (g_b - g_d))
    return out


def response_probabilities(params, upper=np.inf, epsabs=1e-12, epsrel=1e-10):
    """Return dict with p(0), p(1), p(2 = don't know), latent split, residual.

    ``upper`` truncates the decision-time integral (e.g. to compute the
    probability of responding before a deadline, ``upper = deadline - t``).
    """
    vals = []
    for k in range(4):

        def fk(tt, k=k):
            return float(joint_densities(np.array([tt]), params)[k][0])

        val, _ = integrate.quad(fk, 0.0, upper, limit=500, epsabs=epsabs, epsrel=epsrel)
        vals.append(val)
    p = {
        0: vals[0],
        1: vals[1],
        2: vals[2] + vals[3],
        "dk_via_0": vals[2],
        "dk_via_1": vals[3],
    }
    p["residual"] = 1.0 - (p[0] + p[1] + p[2]) if np.isinf(upper) else np.nan
    return p


def mean_decision_times(params, upper, epsabs=1e-12, epsrel=1e-10):
    """Mean decision time per response (0, 1, 2) among decisions before ``upper``.

    ``upper`` must be finite: with truncated-normal rates the rate density is
    positive at 0, so the finishing-time tail decays like ``1/t**2`` and the
    unconditional mean decision time is infinite.
    """
    if not np.isfinite(upper):
        raise ValueError("upper must be finite: the unconditional mean is infinite.")
    out = {}
    probs = response_probabilities(params, upper=upper, epsabs=epsabs, epsrel=epsrel)
    for resp, ks in ((0, (0,)), (1, (1,)), (2, (2, 3))):

        def g(tt, ks=ks):
            dens = joint_densities(np.array([tt]), params)
            return tt * sum(float(dens[k][0]) for k in ks)

        m, _ = integrate.quad(g, 0.0, upper, limit=500, epsabs=epsabs, epsrel=epsrel)
        out[resp] = m / probs[resp] if probs[resp] > 0 else np.nan
    return out
