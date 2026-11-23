"""Closed-form TRDM densities (paper Eqs. 1-3) and race probabilities.

Used as an independent reference for the sampler and to compute
model-implied quantities (e.g. the share of timer-terminated trials) without
Monte Carlo error. Every process is a shifted one-boundary Wiener process with
drift ``g``, diffusion coefficient ``s > 0``, threshold ``a`` and onset ``tau``:

    f(t) = a / (s sqrt(2 pi x^3)) exp(-(a - g x)^2 / (2 s^2 x)),  x = t - tau
    F(t) = Phi((g x - a)/(s sqrt x)) + exp(2 a g / s^2) Phi(-(g x + a)/(s sqrt x))

valid for any sign of ``g`` (for ``g < 0`` the distribution is defective,
``F(inf) = exp(2 a g / s^2)``). The exponential term is evaluated in log space
so that small ``s`` (e.g. the paper's interval-production sigma_T = .14) does
not overflow.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.special import log_ndtr, ndtr


@dataclass(frozen=True)
class Wald:
    """Shifted one-boundary Wiener first-passage law."""

    drift: float
    s: float
    a: float
    tau: float

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        x = t - self.tau
        out = np.zeros_like(x)
        m = x > 0
        xm = x[m]
        out[m] = (
            self.a
            / (self.s * np.sqrt(2 * np.pi * xm**3))
            * np.exp(-((self.a - self.drift * xm) ** 2) / (2 * self.s**2 * xm))
        )
        return out

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        x = t - self.tau
        out = np.zeros_like(x)
        m = x > 0
        xm = x[m]
        sq = self.s * np.sqrt(xm)
        term1 = ndtr((self.drift * xm - self.a) / sq)
        log_term2 = 2 * self.a * self.drift / self.s**2 + log_ndtr(
            -(self.drift * xm + self.a) / sq
        )
        out[m] = term1 + np.exp(log_term2)
        return np.clip(out, 0.0, 1.0)

    def sf(self, t):
        return 1.0 - self.cdf(t)

    def total_mass(self) -> float:
        """P(the threshold is ever reached)."""
        if self.drift >= 0:
            return 1.0
        return float(np.exp(2 * self.a * self.drift / self.s**2))


def race_probabilities(
    evidence: list[Wald], timer: Wald, upper: float = np.inf
) -> dict[str, np.ndarray | float]:
    """Probability that each process wins the race (finishes first by ``upper``).

    Returns ``{'evidence': array of P(accumulator i wins), 'timer': P(timer
    wins), 'none': P(nothing finishes by upper)}``. The integrals
    ``int f_k(t) prod_{j != k} S_j(t) dt`` are evaluated with adaptive
    quadrature, split at the onsets.
    """
    procs = list(evidence) + [timer]
    breaks = sorted({p.tau for p in procs})

    def integrand(t, k):
        val = procs[k].pdf(np.array([t]))[0]
        if val == 0.0:
            return 0.0
        for j, p in enumerate(procs):
            if j != k:
                val *= p.sf(np.array([t]))[0]
        return val

    def integral(k):
        total = 0.0
        edges = breaks + ([upper] if np.isfinite(upper) else [])
        for left, right in itertools.pairwise(edges):
            if right > left:
                total += integrate.quad(
                    integrand,
                    left,
                    right,
                    args=(k,),
                    limit=400,
                    epsabs=1e-12,
                    epsrel=1e-10,
                )[0]
        if not np.isfinite(upper):
            total += integrate.quad(
                integrand,
                edges[-1],
                np.inf,
                args=(k,),
                limit=400,
                epsabs=1e-12,
                epsrel=1e-10,
            )[0]
        return total

    probs = np.array([integral(k) for k in range(len(procs))])
    return {
        "evidence": probs[:-1],
        "timer": float(probs[-1]),
        "none": float(max(0.0, 1.0 - probs.sum())),
    }


def response_probabilities(
    evidence: list[Wald], timer: Wald, upper: float = np.inf
) -> np.ndarray:
    """P(response i) with the paper's unbiased timer guess, Eq. 1 integrated."""
    r = race_probabilities(evidence, timer, upper)
    n = len(evidence)
    return r["evidence"] + r["timer"] / n


def response_density(evidence: list[Wald], timer: Wald, i: int, t) -> np.ndarray:
    """Eq. 1: defective density of response ``i`` at RT ``t``."""
    t = np.asarray(t, dtype=float)
    n = len(evidence)
    surv_e = np.ones_like(t)
    for p in evidence:
        surv_e = surv_e * p.sf(t)
    own = evidence[i].pdf(t)
    others = np.ones_like(t)
    for j, p in enumerate(evidence):
        if j != i:
            others = others * p.sf(t)
    return own * others * timer.sf(t) + timer.pdf(t) * surv_e / n
