"""Shared calculations for the new notebooks; model algorithms live in the prototypes.

Keep this module about measurements and case definitions. The notebooks explain
why each comparison is useful and show the settings that control it.
"""
from copy import deepcopy
from pathlib import Path
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr, log_ndtr
from scipy.stats import binomtest, norm

from .paper_method_prototypes import (
    RaceAccumulator, MultiStageRaceModel, CenteredAccumulator,
    VolterraAccumulator, AnalyticFirstVolterraAccumulator, specifications, sample_runner, sample_race,
)

METHODS = {"Laguerre": RaceAccumulator, "Centered": CenteredAccumulator,
           "Volterra": VolterraAccumulator,
           "Volterra analytic first": AnalyticFirstVolterraAccumulator}
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def build_model(specs, method="Volterra", order=80, **settings):
    accumulators = [METHODS[method](**deepcopy(s), quadrature_order=order, **settings)
                    for s in specs]
    # A prepared Volterra representation belongs to one parameter setting.
    # Prevent accidental array edits after it has been cached.
    for acc in accumulators:
        for name in ("breaks", "mu", "sigma", "boundary_slopes", "quadrature_scale", "_boundary_starts"):
            getattr(acc, name).setflags(write=False)
    return MultiStageRaceModel(accumulators)


def analytic_cases():
    cases = {}
    for stages in (1, 5, 10, 20):
        specs = deepcopy(specifications("A"))
        if stages != 5:
            for s in specs:
                s["breaks"] = np.linspace(0, 2, stages + 1)
                for name in ("mu", "sigma", "boundary_slopes"):
                    s[name] = [s[name][0]] * stages
        label = "1 stage" if stages == 1 else f"{stages} stages"
        cases[label] = specs
    cases["Short first"] = specifications("short")
    cases["Short middle"] = deepcopy(specifications("A"))
    cases["Short middle"][0]["breaks"] = [0, .22, .2201, .95, 1.42, 2]
    cases["Negative drift"] = specifications("negative")
    cases["Near boundary"] = deepcopy(specifications("A"))
    cases["Near boundary"][0]["x0"] = 1.40
    cases["Varying parameters"] = specifications("B")
    for s, drift_in_clock in zip(cases["Varying parameters"], (.6, -.2)):
        s["mu"] = np.asarray(s["boundary_slopes"]) + drift_in_clock * np.asarray(s["sigma"])**2
    return cases


def parameter_table(specs):
    rows = []
    for runner, s in enumerate(specs):
        for k, duration in enumerate(np.diff(s["breaks"])):
            rows.append(dict(runner=runner, stage=k + 1, start=s["breaks"][k],
                             duration=duration, mu=s["mu"][k], sigma=s["sigma"][k],
                             boundary_slope=s["boundary_slopes"][k],
                             initial_boundary=s["boundary0"], x0=s.get("x0", 0)))
    return pd.DataFrame(rows)


def time_grid(specs, points=1001):
    breaks = np.unique(np.concatenate([s["breaks"] for s in specs]))
    deadline = breaks[-1]
    transitions = breaks[(breaks > 0) & (breaks < deadline)]
    offsets = np.logspace(-8, np.log10(.02), 12)
    grid = np.r_[np.linspace(0, deadline, points), np.geomspace(1e-8, .02, 120),
                 breaks, np.nextafter(transitions, -np.inf), np.nextafter(transitions, np.inf),
                 (breaks[:, None] - offsets).ravel(), (breaks[:, None] + offsets).ravel()]
    return np.unique(grid[(grid >= 0) & (grid <= deadline)])


def exact_marginals(specs, times):
    """Wald laws in cumulative-variance time, independent of either solver."""
    t = np.asarray(times, dtype=float)
    densities, cdfs = [], []
    for s in specs:
        breaks = np.asarray(s["breaks"])
        sigma = np.asarray(s["sigma"])
        relative = (np.asarray(s["mu"]) - s["boundary_slopes"]) / sigma**2
        if not np.allclose(relative, relative[0], atol=1e-13, rtol=1e-13):
            raise ValueError("This schedule does not have the variance-clock reference")
        drift = relative[0]
        elapsed = np.clip(t[..., None] - breaks[:-1], 0, np.diff(breaks))
        variance_time = np.sum(elapsed * sigma**2, axis=-1)
        gap = s["boundary0"] - s.get("x0", 0)
        stage = np.minimum(np.searchsorted(breaks[1:], t, side="left"), len(sigma) - 1)
        u = np.maximum(variance_time, np.finfo(float).tiny)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            density = np.exp(2*np.log(sigma[stage]) + np.log(gap) - .5*np.log(2*np.pi)
                             - 1.5*np.log(u) - (gap-drift*u)**2/(2*u))
            cdf = ndtr((drift*u-gap)/np.sqrt(u)) + np.exp(2*drift*gap + log_ndtr((-drift*u-gap)/np.sqrt(u)))
        densities.append(np.where(t == 0, 0, density))
        cdfs.append(np.where(t == 0, 0, cdf))
    return np.stack(densities, axis=-1), np.stack(cdfs, axis=-1)


def race_density(density, cdf):
    # No clipping here: negative interpolation artifacts must remain visible.
    return np.stack([density[..., i] * np.prod(np.delete(1-cdf, i, axis=-1), axis=-1)
                     for i in range(density.shape[-1])], axis=-1)


def integration_rule(specs, edges, order=48):
    """Split every RT bin at stage changes and integrate in square-root time."""
    breaks = np.unique(np.concatenate([s["breaks"] for s in specs]))
    z, w = np.polynomial.legendre.leggauss(order)
    r, weights_r = (1+z)/2, w/2
    nodes, weights, bins = [], [], []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        cuts = np.unique(np.r_[left, breaks[(breaks > left) & (breaks < right)], right])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            nodes.extend(lo + (hi-lo)*r**2)
            weights.extend(weights_r * 2*(hi-lo)*r)
            bins.extend([i] * order)
    return np.asarray(nodes), np.asarray(weights), np.asarray(bins)


def probability_bins(specs, model=None, order=48, edges=None):
    deadline = specs[0]["breaks"][-1]
    edges = np.linspace(0, deadline, 13) if edges is None else np.asarray(edges)
    t, w, index = integration_rule(specs, edges, order)
    evaluate = model.marginals if model is not None else lambda x: exact_marginals(specs, x)
    f, F = evaluate(t)
    bins = np.zeros((len(edges)-1, len(specs)))
    np.add.at(bins, index, race_density(f, F)*w[:, None])
    omission = float(np.prod(1-evaluate(deadline)[1]))
    return edges, bins, omission


def accuracy_row(name, specs, method, order, bin_order=48):
    t = time_grid(specs)
    ft, Ft = exact_marginals(specs, t)
    joint_true = race_density(ft, Ft)
    start = perf_counter()
    model = build_model(specs, method, order)
    f, F = model.marginals(t)
    seconds = perf_counter() - start
    joint = race_density(f, F)
    difference = abs(joint-joint_true)
    worst_time, worst_choice = np.unravel_index(np.argmax(difference), difference.shape)
    edges, bins, omission = probability_bins(specs, model, order=bin_order)
    _, true_bins, true_omission = probability_bins(specs, order=bin_order)
    nodes, weights, _ = integration_rule(specs, edges, order=bin_order)
    integrated_error = abs(race_density(*model.marginals(nodes)) - race_density(*exact_marginals(specs, nodes)))
    above_floor = joint_true > 1e-5
    max_density_error = float(difference.max())
    max_cdf_error = float(abs(F-Ft).max())
    row = dict(case=name, method=method, order=order, bin_order=bin_order, grid_points=len(t),
               max_density_error=max_density_error, max_cdf_error=max_cdf_error,
               max_relative_error=float((difference[above_floor]/joint_true[above_floor]).max()),
               density_L1=float(np.sum(integrated_error*weights[:, None])),
               worst_time=float(t[worst_time]), worst_choice=int(worst_choice),
               choice_probability_error=float(abs(bins.sum(axis=0)-true_bins.sum(axis=0)).max()),
               max_bin_error=float(abs(bins-true_bins).max()), omission_error=abs(omission-true_omission),
               mass_error=abs(float(bins.sum()+omission)-1),
               min_density=float(f.min()), min_cdf=float(F.min()), max_cdf=float(F.max()),
               largest_cdf_drop=float(max(0, -np.diff(F, axis=0).min())),
               setup_and_grid_seconds=seconds)
    row["meets_targets"] = bool(max_density_error < 1e-6 and max_cdf_error < 1e-7
                                and row["max_bin_error"] < 1e-7 and row["mass_error"] < 1e-7
                                and row["largest_cdf_drop"] < 1e-10 and row["min_density"] > -1e-10)
    return row


def wilson_interval(count, total, alpha=.05):
    count = np.asarray(count)
    p = count / total
    z = norm.ppf(1-alpha/2)
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total**2)) / (1+z*z/total)
    return center-half, center+half


def compare_sample(rt, choice, edges, bins, omission, alpha=.05):
    n = len(rt)
    counts = np.stack([np.histogram(rt[choice == i], edges)[0] for i in range(bins.shape[1])], axis=1)
    probabilities = np.r_[bins.ravel(), omission]
    all_counts = np.r_[counts.ravel(), (choice == -1).sum()]
    # Correct across this run's 24 choice/RT bins and its omission category.
    low, high = wilson_interval(all_counts, n, alpha/len(probabilities))
    pvalues = np.array([binomtest(int(k), n, float(p)).pvalue for k, p in zip(all_counts, probabilities)])
    labels = [f"bin {j+1}, choice {i}" for j in range(bins.shape[0]) for i in range(bins.shape[1])] + ["omission"]
    detail = pd.DataFrame(dict(category=labels, predicted=probabilities,
                               observed=all_counts/n, lower=low, upper=high,
                               adjusted_p=np.minimum(1, pvalues*len(probabilities))))
    lo, hi = wilson_interval((choice == -1).sum(), n)
    summary = dict(trials=n, max_bin_discrepancy=float(abs(counts/n-bins).max()),
                   predicted_omission=omission, observed_omission=float((choice == -1).mean()),
                   omission_lower=float(lo), omission_upper=float(hi),
                   rejected_categories=int(np.sum(pvalues < alpha/len(probabilities))))
    return summary, detail


def empirical_cdf(rt, choice, grid, winning_choice=None):
    included = choice >= 0 if winning_choice is None else choice == winning_choice
    return np.searchsorted(np.sort(rt[included]), grid, side="right") / len(rt)


def refine_reference(specs):
    model = build_model(specs, order=80)
    edges, bins, omission = probability_bins(specs, model, order=48)
    _, finer_bins, finer_omission = probability_bins(specs, build_model(specs, order=120), order=48)
    _, finer_time_bins, _ = probability_bins(specs, model, order=96)
    diagnostics = dict(order_change=float(abs(finer_bins-bins).max()),
                       time_integration_change=float(abs(finer_time_bins-bins).max()),
                       omission_change=abs(finer_omission-omission),
                       mass_error=abs(float(bins.sum()+omission)-1))
    if max(diagnostics.values()) > 1e-7:
        raise RuntimeError(f"Refine this probability reference before simulating: {diagnostics}")
    return model, edges, bins, omission, diagnostics


def evidence_conditions():
    def condition(swap=None, moving_boundary=False):
        breaks = sorted(set([0, 1.8] + ([] if swap is None else [swap])
                            + ([.6] if moving_boundary else [])))
        specs = []
        for runner in range(2):
            initial = (.7, .2)[runner]
            later = (.2, .7)[runner]
            specs.append(dict(breaks=breaks, mu=[later if swap is not None and t >= swap else initial for t in breaks[:-1]],
                              sigma=[.8]*(len(breaks)-1),
                              boundary_slopes=[-.1 if moving_boundary and t >= .6 else 0 for t in breaks[:-1]],
                              boundary0=1.35, x0=0))
        return specs
    conditions = {"Constant": condition(), "Immediate reversal": condition(.6),
                  "Delayed reversal": condition(.75), "Boundary change": condition(None, True),
                  "Reversal + boundary": condition(.6, True),
                  "Early favors 0": condition(.9)}
    conditions["Early favors 1"] = deepcopy(conditions["Early favors 0"])
    for s in conditions["Early favors 1"]:
        s["mu"] = list(reversed(s["mu"]))
    return conditions


def split_without_changing(specs, stages):
    """Add splits to the longest pieces; retain every real parameter change."""
    result = deepcopy(specs)
    for s in result:
        old = np.asarray(s["breaks"])
        breaks = old.copy()
        if stages < len(old)-1:
            raise ValueError("Cannot remove a real change to reach this stage count")
        while len(breaks)-1 < stages:
            i = np.argmax(np.diff(breaks))
            breaks = np.insert(breaks, i+1, (breaks[i]+breaks[i+1])/2)
        index = np.minimum(np.searchsorted(old[1:], breaks[:-1], side="right"), len(old)-2)
        for name in ("mu", "sigma", "boundary_slopes"):
            s[name] = np.asarray(s[name])[index]
        s["breaks"] = breaks
    return result


def batch_log_likelihood(model, choices, rts):
    f, F = model.marginals(rts)
    joint = race_density(f, F)
    p = np.prod(1-F, axis=-1)
    responded = choices >= 0
    p[responded] = joint[np.flatnonzero(responded), choices[responded]]
    if np.any(p <= 0):
        raise ValueError("Nonpositive likelihood: inspect approximation accuracy")
    return float(np.log(p).sum())


def save_results(stem, tables, settings):
    folder = RESULTS / stem
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    metadata = dict(settings=settings, python=platform.python_version(),
                    platform=platform.platform(), machine=platform.machine(),
                    numpy=np.__version__, scipy=scipy.__version__)
    (folder / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder
