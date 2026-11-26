"""Analytic LBA reference quantities used to check the MLBA simulator.

Single-accumulator defective first-passage density and CDF of the LBA with
start ``U(0, A)``, threshold ``b`` and rate ``N(v, s)`` (Brown & Heathcote,
2008, Eqs. 1-2). They are defective: ``F(inf) = Phi(v / s)``, the probability of
a positive rate. Choice probabilities are

    P(i) = int_0^inf f_i(t) prod_{j != i} (1 - F_j(t)) dt,

computed (as in the paper, "numerically integrating over decision times") by
Gauss-Legendre quadrature after the substitution ``t = x / (1 - x)``. They sum
to ``1 - prod_i Phi(-v_i / s)``, the probability that some accumulator
finishes; the simulator records the remainder as omissions.

``choice_probabilities_quad`` is a slow adaptive (``scipy.integrate.quad``)
reference used only to measure the quadrature error.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .mlba_frontend import mlba_drifts


def lba_cdf(t, v, A, b, s):
    """Defective CDF of one LBA accumulator's finishing time (A > 0)."""
    t = np.asarray(t, dtype=np.float64)
    ts = t * s
    z1 = (b - A - t * v) / ts
    z2 = (b - t * v) / ts
    return (
        1.0
        + (b - A - t * v) / A * norm.cdf(z1)
        - (b - t * v) / A * norm.cdf(z2)
        + ts / A * norm.pdf(z1)
        - ts / A * norm.pdf(z2)
    )


def lba_pdf(t, v, A, b, s):
    """Defective density of one LBA accumulator's finishing time (A > 0)."""
    t = np.asarray(t, dtype=np.float64)
    ts = t * s
    z1 = (b - A - t * v) / ts
    z2 = (b - t * v) / ts
    return (
        -v * norm.cdf(z1) + s * norm.pdf(z1) + v * norm.cdf(z2) - s * norm.pdf(z2)
    ) / A


def _nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    x = (x + 1.0) / 2.0
    w = w / 2.0
    t = x / (1.0 - x)
    return t, w / (1.0 - x) ** 2


def choice_probabilities(drifts, A=1.0, b=2.0, s=1.0, n_nodes=400):
    """Choice probabilities of a standard LBA race, vectorised over rows.

    Parameters
    ----------
    drifts : array (n_sets, n_acc) or (n_acc,)
    A, b, s : scalars (common to all accumulators)
    n_nodes : Gauss-Legendre nodes on x in (0, 1), t = x / (1 - x)
    """
    D = np.atleast_2d(np.asarray(drifts, dtype=np.float64))
    t, w = _nodes(n_nodes)
    T = t[None, None, :]
    V = D[:, :, None]
    surv = 1.0 - lba_cdf(T, V, A, b, s)
    dens = lba_pdf(T, V, A, b, s)
    out = np.empty(D.shape)
    for i in range(D.shape[1]):
        others = np.prod(np.delete(surv, i, axis=1), axis=1)
        out[:, i] = (dens[:, i] * others * w[None]).sum(-1)
    return out if np.ndim(drifts) > 1 else out[0]


def choice_probabilities_quad(drifts, A=1.0, b=2.0, s=1.0):
    """Adaptive-quadrature reference for ``choice_probabilities`` (slow)."""
    d = np.asarray(drifts, dtype=np.float64)
    out = []
    for i in range(d.size):

        def g(tt, i=i):
            val = lba_pdf(tt, d[i], A, b, s)
            for j in range(d.size):
                if j != i:
                    val = val * (1.0 - lba_cdf(tt, d[j], A, b, s))
            return float(val)

        out.append(quad(g, 0.0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-11)[0])
    return np.array(out)


def rt_cdf(tgrid, drifts, A, b, s, choice, n_nodes=400):
    """Defective joint CDF P(choice, decision time <= t) on ``tgrid``."""
    d = np.asarray(drifts, dtype=np.float64)
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    out = []
    for tt in np.atleast_1d(tgrid):
        u = (x + 1) / 2 * tt
        ww = w / 2 * tt
        val = lba_pdf(u, d[choice], A, b, s)
        for j in range(d.size):
            if j != choice:
                val = val * (1 - lba_cdf(u, d[j], A, b, s))
        out.append(np.sum(val * ww))
    return np.array(out)


def mlba_choice_probabilities(
    options, m, lambda1, lambda2, I0, beta=1.0, A=1.0, b=2.0, s=1.0, n_nodes=400
):
    """MLBA choice probabilities for one choice set ``options = [(P, Q), ...]``."""
    opts = np.asarray(options, dtype=np.float64)
    d = mlba_drifts(opts[:, 0], opts[:, 1], m, lambda1, lambda2, I0, beta)[0]
    return choice_probabilities(d, A, b, s, n_nodes)
