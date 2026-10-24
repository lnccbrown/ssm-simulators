"""Isolated research experiments; not part of the production API.

Run from the repository root:
  python -m notebooks.volterra_accuracy.paper_method_prototypes

Research code supporting the Volterra accuracy notebook.
"""
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from scipy.stats import binomtest
from scipy.integrate import quad_vec
from scipy.interpolate import BarycentricInterpolator
from scipy.special import ndtr, log_ndtr

from .race import RaceAccumulator, MultiStageRaceModel


class CenteredAccumulator(RaceAccumulator):
    """Finite Gauss–Legendre grid following the unabsorbed Gaussian envelope."""

    tail_width = 8.0

    def _nodes_and_weights(self, stage):
        durations = np.diff(self.breaks[:stage + 1])
        mean = self.x0 + self.mu[:stage] @ durations
        sd = np.sqrt(self.sigma[:stage] ** 2 @ durations)
        upper = min(self._boundary_starts[stage], mean + self.tail_width * sd)
        lower = min(mean - self.tail_width * sd, upper - np.finfo(float).eps * sd)
        z, w = np.polynomial.legendre.leggauss(self.quadrature_order)
        half = (upper - lower) / 2
        return lower + half * (1 + z), half * w


class VolterraAccumulator(RaceAccumulator):
    """Research extension of the neural paper's Appendix B to variance time.

    Drift and boundary movement become a piecewise linear Brownian boundary.
    The Volterra kernel vanishes within each linear segment. Only completed
    segments are integrated; square-root changes of variables handle the
    history endpoint singularity and density cusp after a stage change.
    Chebyshev interpolation represents the FPT density in square-root elapsed
    variance time. This is not an error-certified solver.
    """

    analytic_first_stage = False

    def _first_stage_marginal(self, u):
        """Exact first-stage density and CDF in cumulative-variance time."""
        u = np.asarray(u, dtype=float)
        positive = u > 0
        safe = np.where(positive, u, 1.0)
        gap = self.boundary0 - self.x0
        drift = (self.mu[0] - self.boundary_slopes[0]) / self.sigma[0]**2
        density = np.exp(np.log(gap) - .5*np.log(2*np.pi) - 1.5*np.log(safe)
                         - (gap-drift*safe)**2/(2*safe))
        cdf = ndtr((drift*safe-gap)/np.sqrt(safe)) + np.exp(
            2*drift*gap + log_ndtr((-drift*safe-gap)/np.sqrt(safe)))
        return np.where(positive, density, 0.0), np.where(positive, cdf, 0.0)

    @cached_property
    def _volterra(self):
        durations = np.diff(self.breaks)
        ends = np.r_[0.0, np.cumsum(self.sigma**2 * durations)]
        slopes = (self.boundary_slopes - self.mu) / self.sigma**2
        heights = np.r_[self.boundary0 - self.x0,
                        self.boundary0 - self.x0 + np.cumsum((self.boundary_slopes - self.mu) * durations)]
        # Explicit weights keep interpolation deterministic (no randomized
        # product ordering used by the generic barycentric constructor).
        z = -np.cos(np.pi * np.arange(self.quadrature_order + 1) / self.quadrature_order)
        bary_weights = (-1.0)**np.arange(z.size)
        bary_weights[[0, -1]] *= .5
        segments, cumulative = [], [0.0]
        root2pi = np.sqrt(2 * np.pi)
        for stage in range(self.n_stages):
            lo, hi = ends[stage:stage + 2]
            if stage == 0 and self.analytic_first_stage:
                # Preserve the whole first-stage law, not just its plotting values.
                segments.append((None, None))
                cumulative.append(float(self._first_stage_marginal(hi)[1]))
                continue
            u = lo + (hi - lo) * ((z + 1) / 2)**2
            h = heights[stage] + slopes[stage] * (u - lo)
            safe_u = np.maximum(u, np.finfo(float).tiny)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                values = (h / safe_u - slopes[stage]) * np.exp(-h*h / (2*safe_u)) / (root2pi*np.sqrt(safe_u))
            values[u == 0] = 0
            for previous in range(stage):
                left, right = ends[previous:previous + 2]
                ylo, yhi = np.sqrt(np.maximum(u - right, 0)), np.sqrt(u - left)
                span = yhi - ylo
                previous_interpolator = segments[previous][0]
                # h(u)-h(s) = c_previous*y^2 + offset. Form offset
                # from slope differences, avoiding subtraction of nearly
                # equal boundary heights immediately after a breakpoint.
                offset = (slopes[stage] - slopes[previous]) * (u - lo)
                for middle in range(previous + 1, stage):
                    offset += (slopes[middle] - slopes[previous]) * (ends[middle + 1] - ends[middle])

                def integrand(v):
                    y = ylo + span * v
                    s = np.clip(u - y*y, left, right)
                    slope_difference = slopes[stage] - slopes[previous] - offset / (y*y)
                    gaussian = np.exp(-.5 * (slopes[previous] * y + offset/y)**2) / root2pi
                    previous_z = 2 * np.sqrt(np.clip((s - left) / (right - left), 0, 1)) - 1
                    if previous == 0 and self.analytic_first_stage:
                        history_density = self._first_stage_marginal(s)[0]
                    else:
                        history_density = previous_interpolator(previous_z)
                    return 2 * span * history_density * slope_difference * gaussian

                integral, error, info = quad_vec(integrand, 0, 1, epsabs=self.passage_atol,
                                                epsrel=self.passage_rtol, full_output=True)
                if not info.success:
                    raise RuntimeError("Volterra history integration did not converge")
                values += integral
            interpolator = BarycentricInterpolator(z, values, wi=bary_weights)
            coefficients = np.polynomial.chebyshev.chebfit(z, values, self.quadrature_order)
            weighted_coefficients = np.polynomial.chebyshev.chebmul(coefficients, [(hi - lo) / 2] * 2)
            primitive = np.polynomial.chebyshev.chebint(weighted_coefficients)
            mass = (np.polynomial.chebyshev.chebval(1, primitive)
                    - np.polynomial.chebyshev.chebval(-1, primitive))
            segments.append((interpolator, primitive))
            cumulative.append(cumulative[-1] + mass)
        return ends, segments, np.asarray(cumulative)

    def marginal(self, t):
        t = np.asarray(t, dtype=float)
        if np.any((t < 0) | (t > self.deadline)):
            raise ValueError("Prototype expects times in [0, deadline]")
        ends, segments, cumulative = self._volterra
        density, cdf = np.zeros_like(t), np.zeros_like(t)
        stages = np.minimum(np.searchsorted(self.breaks[1:], t, side="left"), self.n_stages - 1)
        for stage, (interpolator, primitive) in enumerate(segments):
            mask = stages == stage
            u = ends[stage] + self.sigma[stage]**2 * (t[mask] - self.breaks[stage])
            if stage == 0 and self.analytic_first_stage:
                first_f, first_F = self._first_stage_marginal(u)
                density[mask] = self.sigma[0]**2 * first_f
                cdf[mask] = first_F
                continue
            z = 2 * np.sqrt(np.clip((u - ends[stage]) / (ends[stage + 1] - ends[stage]), 0, 1)) - 1
            density[mask] = self.sigma[stage]**2 * interpolator(z)
            cdf[mask] = cumulative[stage] + (
                np.polynomial.chebyshev.chebval(z, primitive)
                - np.polynomial.chebyshev.chebval(-1, primitive))
        # Deliberately leave raw values unclipped to expose interpolation error.
        return density, cdf


class AnalyticFirstVolterraAccumulator(VolterraAccumulator):
    """Keep stage one exact, including its density in later history integrals.

    Later stages retain the original Volterra interpolation. Keeping this as
    a separate backend makes the before/after comparison reproducible.
    """

    analytic_first_stage = True


def sample_runner(acc, n, rng):
    """Stage-exact FPT sampling in ideal arithmetic, with rejection survivors.

    A survivor endpoint is sampled from the killed transition density,
    conditional on no crossing. It is NOT an unconstrained Gaussian endpoint.
    No finite-dt trajectory is constructed. The rejection guard raises rather
    than returning a biased sample in extreme low-survival regimes.
    """
    hits = np.full(n, np.inf)
    positions = np.full(n, acc.x0, dtype=float)
    active = np.arange(n)
    proposals = 0
    for stage, duration in enumerate(np.diff(acc.breaks)):
        if not active.size:
            break
        x = positions[active].copy()
        gap = acc._boundary_starts[stage] - x
        sigma = acc.sigma[stage]
        relative_drift = acc.mu[stage] - acc.boundary_slopes[stage]
        if relative_drift == 0:
            tau = (gap / (sigma * rng.standard_normal(active.size))) ** 2
        else:
            tau = rng.wald(gap / abs(relative_drift), (gap / sigma) ** 2)
            if relative_drift < 0:
                hit_probability = np.exp(2 * relative_drift * gap / sigma**2)
                tau[rng.random(active.size) >= hit_probability] = np.inf
        crossed = tau <= duration
        hits[active[crossed]] = acc.breaks[stage] + tau[crossed]
        active = active[~crossed]
        if stage == acc.n_stages - 1:
            break
        x, gap = x[~crossed], gap[~crossed]
        pending = np.arange(active.size)
        boundary_end = acc.boundary(acc.breaks[stage + 1])
        for _ in range(10000):
            if not pending.size:
                break
            proposals += pending.size
            candidate = rng.normal(x[pending] + acc.mu[stage] * duration,
                                   sigma * np.sqrt(duration))
            end_gap = np.maximum(boundary_end - candidate, 0.0)
            acceptance = -np.expm1(-2 * gap[pending] * end_gap / (sigma**2 * duration))
            accepted = rng.random(pending.size) < acceptance
            positions[active[pending[accepted]]] = candidate[accepted]
            pending = pending[~accepted]
        else:
            raise RuntimeError("Survivor rejection exceeded 10000 rounds")
    return hits, proposals


def sample_race(model, n, seed):
    rng = np.random.default_rng(seed)
    results = [sample_runner(a, n, rng) for a in model.accumulators]
    hits = np.stack([r[0] for r in results])
    rt = hits.min(axis=0)
    choices = hits.argmin(axis=0)
    choices[~np.isfinite(rt)] = -1
    rt[~np.isfinite(rt)] = model.deadline
    return rt, choices, sum(r[1] for r in results)


def specifications(kind):
    if kind == "B":
        return [dict(breaks=[0, .18, .54, .87, 1.31, 1.8],
                     mu=[.2, .85, .35, .7, .25], sigma=[.75, .85, .8, .9, .75],
                     boundary_slopes=[-.03, .04, -.08, .02, -.04], boundary0=1.35),
                dict(breaks=[0, .31, .63, 1.06, 1.46, 1.8],
                     mu=[.4, .15, .9, .3, .6], sigma=[.85, .75, .95, .8, .9],
                     boundary_slopes=[.02, -.05, .03, -.07, .01], boundary0=1.38, x0=-.05)]
    return [dict(breaks=[0, .02 if kind == "short" else .22, .57, .95, 1.42, 2],
                 mu=[-.3 if kind == "negative" else .55] * 5, sigma=[.8] * 5,
                 boundary_slopes=[-.12] * 5, boundary0=1.45, x0=.1),
            dict(breaks=[0, .31, .66, 1.08, 1.61, 2], mu=[.3] * 5,
                 sigma=[.95] * 5, boundary_slopes=[.06] * 5, boundary0=1.3, x0=-.05)]


def make_model(kind, cls=RaceAccumulator, order=40):
    return MultiStageRaceModel([cls(**s, quadrature_order=order) for s in specifications(kind)])


def analytic_model(kind):
    specs = specifications(kind)
    return MultiStageRaceModel([RaceAccumulator(**{
        **s, "breaks": [0, 2], "mu": s["mu"][:1], "sigma": s["sigma"][:1],
        "boundary_slopes": s["boundary_slopes"][:1]}) for s in specs])


def bin_probabilities(model):
    edges = np.linspace(0, model.deadline, 13)
    all_breaks = np.concatenate([a.breaks for a in model.accumulators])
    z, w = np.polynomial.legendre.leggauss(24)
    nodes, weights, indices = [], [], []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        cuts = np.unique(np.r_[left, all_breaks[(all_breaks > left) & (all_breaks < right)], right])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            nodes.extend(lo + (hi - lo) * (1 + z) / 2)
            weights.extend(w * (hi - lo) / 2)
            indices.extend([i] * len(z))
    weighted = model.joint_density(nodes) * np.asarray(weights)[:, None]
    bins = np.zeros((12, model.n_choices))
    np.add.at(bins, indices, weighted)
    omission = model.likelihood(-1, model.deadline)
    return edges, bins, omission


def main():
    results = {"quadrature": [], "sampler_checks": []}
    for kind in ("A", "short", "negative"):
        specs = specifications(kind)
        breaks = np.unique(np.concatenate([s["breaks"] for s in specs]))
        offsets = np.r_[0, np.logspace(-8, np.log10(.02), 25)]
        grid = np.unique(np.r_[np.linspace(.001, 2, 2001),
                               (breaks[:, None] + offsets).ravel(),
                               (breaks[:, None] - offsets).ravel()])
        grid = grid[(grid > 0) & (grid <= 2)]
        exact = analytic_model(kind)
        true_density, true_cdf = exact.joint_density(grid), exact.marginals(grid)[1]
        for cls in (RaceAccumulator, CenteredAccumulator):
            for order in (20, 40, 80):
                model = make_model(kind, cls, order)
                start = perf_counter()
                density, cdf = model.joint_density(grid), model.marginals(grid)[1]
                row = dict(case=kind, method=cls.__name__, order=order,
                           max_joint_density_error=float(abs(density - true_density).max()),
                           max_marginal_cdf_error=float(abs(cdf - true_cdf).max()),
                           seconds=perf_counter() - start)
                results["quadrature"].append(row)
                print(row, flush=True)
    n = 100000
    # Positive, negative and zero relative drift, with artificial stage splits.
    for k, relative_drift in enumerate((.67, -.18, 0.0)):
        acc = RaceAccumulator([0, .02, .57, .95, 1.42, 2],
                              [relative_drift - .12] * 5, [.8] * 5,
                              [-.12] * 5, 1.45, .1)
        exact = replace(acc, breaks=[0, 2], mu=acc.mu[:1], sigma=acc.sigma[:1],
                        boundary_slopes=acc.boundary_slopes[:1], quadrature_scale=None)
        start = perf_counter()
        hits, proposals = sample_runner(acc, n, np.random.default_rng(20260928 + k))
        grid = np.linspace(0, 2, 2001)
        ecdf = np.searchsorted(np.sort(hits), grid, side="right") / n
        max_error = float(abs(ecdf - exact.marginal(grid)[1]).max())
        # DKW applies after mapping all omissions to an atom beyond the deadline.
        bound = float(np.sqrt(np.log(2 * 3 / .05) / (2 * n)))
        row = dict(relative_drift=relative_drift, max_cdf_error=max_error,
                   simultaneous_95pct_DKW_bound=bound, passed=max_error <= bound,
                   seconds=perf_counter() - start, rejection_proposals=proposals)
        results["sampler_checks"].append(row)
        print(row, flush=True)
    model = make_model("B", CenteredAccumulator, 40)
    edges, bins, omission = bin_probabilities(model)
    _, refined, omission80 = bin_probabilities(make_model("B", CenteredAccumulator, 80))
    _, laguerre, omission_lag = bin_probabilities(make_model("B", RaceAccumulator, 80))
    start = perf_counter()
    rt, choice, proposals = sample_race(model, n, 20261001)
    seconds = perf_counter() - start
    counts = np.stack([np.histogram(rt[choice == i], edges)[0] for i in range(2)], axis=1)
    probabilities = np.r_[bins.ravel(), omission]
    observed_counts = np.r_[counts.ravel(), (choice == -1).sum()]
    pvalues = [binomtest(int(c), n, float(p)).pvalue for c, p in zip(observed_counts, probabilities)]
    results["B"] = dict(n_trials=n, stage_sampler_seconds=seconds,
                        rejection_proposals=proposals,
                        probabilities=probabilities.tolist(), observed=(observed_counts / n).tolist(),
                        omission=omission, observed_omission=float((choice == -1).mean()),
                        omission_mc_standard_error=float(np.sqrt(omission * (1 - omission) / n)),
                        max_bin_error=float(abs(counts / n - bins).max()),
                        max_bin_order40_vs80=float(abs(refined - bins).max()),
                        max_bin_vs_laguerre80=float(abs(laguerre - bins).max()),
                        omission_order40_vs80=abs(omission80 - omission),
                        omission_vs_laguerre80=abs(omission_lag - omission),
                        mass=float(probabilities.sum()),
                        bonferroni_rejected_bins=int(np.sum(np.asarray(pvalues) < .05 / len(pvalues))))
    # Same machine, model, trial count; one timing run, not a general benchmark.
    start = perf_counter()
    model.simulate(n, dt=.00025, rng=20261002)
    results["B"]["euler_seconds_dt_00025"] = perf_counter() - start
    results["B"]["speed_ratio_this_run"] = results["B"]["euler_seconds_dt_00025"] / seconds
    print(results["B"], flush=True)
    output = Path(__file__).with_name("paper_method_results.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    assert all(row["passed"] for row in results["sampler_checks"])
    assert abs(probabilities.sum() - 1) < 1e-6
    assert results["B"]["max_bin_order40_vs80"] < 1e-6
    assert results["B"]["max_bin_vs_laguerre80"] < 1e-6


if __name__ == "__main__":
    main()
