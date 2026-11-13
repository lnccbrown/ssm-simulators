"""Numerical likelihoods and simulation for multi-stage race models.

This module implements the full model described in the standard one-boundary first-passage model.  A
race consists of conditionally independent accumulators, each with a single
absorbing upper boundary.  An accumulator can have a native (different from
the other accumulators) piecewise-linear boundary and piecewise-constant
drift/noise schedule.

Stage-to-stage propagation uses Gauss--Laguerre quadrature.  The final
first-passage integral uses time-scaled adaptive quadrature to resolve
crossings immediately after a stage transition.  The public Python
convention is zero-based choices; ``-1`` denotes an omitted/no-response trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import log_ndtr, ndtr

from .race_single_stage import big_F, q, small_f


def _as_1d(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


@dataclass(frozen=True)
class RaceAccumulator:
    """Parameters for one independently racing, multi-stage accumulator.

    ``breaks`` includes 0 and the common deadline.  ``mu``, ``sigma``, and
    ``boundary_slopes`` have one value for every interval in ``breaks``.
    ``boundary0`` is the initial upper-boundary height and ``x0`` is a
    deterministic starting position.

    ``quadrature_order`` and ``quadrature_scale`` control propagation between
    stages. Within a stage, density and crossing probability are integrated
    adaptively after scaling distance from the boundary by ``sigma*sqrt(t)``.
    ``passage_atol`` and ``passage_rtol`` control that final adaptive integral;
    they do not bound earlier propagation error or the Gaussian tail cutoff.
    """

    breaks: Sequence[float]
    mu: Sequence[float]
    sigma: Sequence[float]
    boundary_slopes: Sequence[float]
    boundary0: float
    x0: float = 0.0
    quadrature_order: int = 20
    quadrature_scale: float | Sequence[float] | None = None
    passage_atol: float = 1e-10
    passage_rtol: float = 1e-8

    def __post_init__(self):
        breaks = _as_1d("breaks", self.breaks)
        mu, sigma, slopes = (
            _as_1d("mu", self.mu),
            _as_1d("sigma", self.sigma),
            _as_1d("boundary_slopes", self.boundary_slopes),
        )
        stages = breaks.size - 1
        if stages < 1 or any(x.size != stages for x in (mu, sigma, slopes)):
            raise ValueError("breaks needs one more element than each stage parameter")
        if not np.isclose(breaks[0], 0.0) or np.any(np.diff(breaks) <= 0.0):
            raise ValueError("breaks must start at 0 and be strictly increasing")
        if np.any(sigma <= 0.0):
            raise ValueError("all sigma values must be positive")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be at least 2")
        if any(not np.isfinite(value) or value <= 0.0
               for value in (self.passage_atol, self.passage_rtol)):
            raise ValueError("passage_atol and passage_rtol must be finite and positive")
        if self.quadrature_scale is None:
            # The surviving density is normally concentrated much more tightly
            # than its full Gaussian support.  Half its unconstrained standard
            # deviation is a useful default Laguerre scale; callers can and
            # should vary it in numerical-accuracy studies.
            scales = 0.5 * np.sqrt(np.cumsum(sigma**2 * np.diff(breaks)))
        else:
            scales = np.broadcast_to(
                np.asarray(self.quadrature_scale, dtype=float), (stages,)
            )
        if np.any(scales <= 0.0):
            raise ValueError("quadrature_scale values must be positive")
        boundary_starts = self.boundary0 + np.r_[0.0, np.cumsum(slopes[:-1] * np.diff(breaks)[:-1])]
        if self.x0 >= boundary_starts[0]:
            raise ValueError("x0 must be strictly below boundary0")
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "boundary_slopes", slopes)
        object.__setattr__(self, "quadrature_scale", scales)
        object.__setattr__(self, "_boundary_starts", boundary_starts)

    @property
    def n_stages(self) -> int:
        return len(self.mu)

    @property
    def deadline(self) -> float:
        return float(self.breaks[-1])

    def boundary(self, t):
        """Upper boundary at absolute time(s) ``t`` within the deadline."""
        t = np.asarray(t, dtype=float)
        if np.any((t < 0.0) | (t > self.deadline)):
            raise ValueError("t must lie within [0, deadline]")
        stage = np.minimum(np.searchsorted(self.breaks[1:], t, side="left"), self.n_stages - 1)
        return self._boundary_starts[stage] + self.boundary_slopes[stage] * (t - self.breaks[stage])

    def _nodes_and_weights(self, stage: int):
        nodes, weights = np.polynomial.laguerre.laggauss(self.quadrature_order)
        # Integral on (-inf, a): lambda * sum(w exp(y) h(a-lambda*y)).
        weights = self.quadrature_scale[stage] * weights * np.exp(nodes)
        positions = self._boundary_starts[stage] - self.quadrature_scale[stage] * nodes
        return positions, weights

    def _incoming_state(self, target_stage: int):
        """Return quadrature positions, weights, q-values for target-stage input."""
        if target_stage == 0:
            return None
        # q-values represented on the input quadrature grid of each next stage.
        positions, weights = self._nodes_and_weights(1)
        duration = self.breaks[1] - self.breaks[0]
        q_values = q(positions, self.mu[0], self.sigma[0], self._boundary_starts[0], self.boundary_slopes[0], duration, self.x0)
        for stage in range(1, target_stage):
            out_positions, _ = self._nodes_and_weights(stage + 1)
            duration = self.breaks[stage + 1] - self.breaks[stage]
            kernel = q(
                out_positions[:, None], self.mu[stage], self.sigma[stage],
                self._boundary_starts[stage], self.boundary_slopes[stage], duration,
                positions[None, :],
            )
            q_values = kernel @ (weights * q_values)
            positions, weights = out_positions, self._nodes_and_weights(stage + 1)[1]
        return positions, weights, np.maximum(q_values, 0.0)

    def _stage_passage(self, stage: int, elapsed: np.ndarray):
        """Density and additional crossing mass since a noninitial stage began.

        Write x = a - sigma*sqrt(s)*z, where s is elapsed time. This resolves
        the O(sqrt(s)) boundary region even as s approaches zero. Evaluate the
        incoming subdensity at these moving positions through the preceding
        killed kernel, rather than interpolating the fixed Laguerre nodes.
        """
        previous = stage - 1
        duration = self.breaks[stage] - self.breaks[previous]
        state = self._incoming_state(previous)
        sigma = self.sigma[stage]
        root_elapsed = np.sqrt(elapsed)
        width = sigma * root_elapsed
        eta = (self.mu[stage] - self.boundary_slopes[stage]) * root_elapsed / sigma

        # Intersect the crossing-kernel region with the incoming diffusion's
        # Gaussian envelope. Otherwise even adaptive quadrature can miss a
        # narrow incoming distribution far from z=0 (e.g. a very short first
        # stage with small noise). Twelve standard deviations make the
        # discarded Gaussian tails negligible at the integration tolerance.
        durations = np.diff(self.breaks[:stage + 1])
        mean = self.x0 + self.mu[:stage] @ durations
        spread = np.sqrt(self.sigma[:stage]**2 @ durations)
        distance = self._boundary_starts[stage] - mean
        lower = max(0.0, distance - 12.0 * spread) / width
        upper = max(0.0, distance + 12.0 * spread) / width
        # Density is localized around eta; the CDF also includes all smaller
        # positive distances. Map their separate intervals onto [0, 1].
        starts = np.stack((np.maximum(lower, eta - 12.0), lower))
        stops = np.minimum(upper, np.maximum(0.0, eta + 12.0))
        spans = np.maximum(stops - starts, 0.0)
        variance = self.sigma[previous]**2 * duration
        drift_distance = (self.mu[previous] - self.boundary_slopes[previous]) * duration
        if state is None:
            starting_gap = self._boundary_starts[previous] - self.x0
        else:
            nodes, weights, values = state
            starting_gap = self._boundary_starts[previous] - nodes
            incoming_weights = weights * values

        def integrand(unit_position):
            z = np.where(spans > 0.0, starts + spans * unit_position, 0.0)
            distance = width * z
            if state is not None:
                distance = distance[..., None]
            # The same killed kernel as q, expressed in boundary distances.
            # Forming x = boundary - distance and then subtracting x from the
            # boundary loses precision for times nextafter a breakpoint.
            incoming = np.exp(-0.5 * (starting_gap - drift_distance - distance)**2 / variance)
            incoming *= -np.expm1(-2.0 * starting_gap * distance / variance)
            incoming /= np.sqrt(2.0 * np.pi * variance)
            if state is not None:
                incoming = incoming @ incoming_weights

            # Include dx = sigma*sqrt(s) dz in both integrands. Express the
            # conditional kernels in z directly to avoid subtracting nearby
            # boundary/position values when s is very small.
            density = incoming[0] * sigma / root_elapsed * z[0] * np.exp(
                -0.5 * (z[0] - eta)**2
            ) / np.sqrt(2.0 * np.pi)
            conditional_cdf = ndtr(eta - z[1]) + np.exp(
                np.minimum(2.0 * eta * z[1] + log_ndtr(-eta - z[1]), 0.0)
            )
            passage = incoming[1] * width * conditional_cdf
            return np.stack((density, passage)) * spans

        result, _, info = quad_vec(
            integrand, 0.0, 1.0, epsabs=self.passage_atol, epsrel=self.passage_rtol,
            norm="max", full_output=True,
        )
        if not info.success:
            raise RuntimeError(f"Stage {stage} passage integration failed: {info.message}")
        return result[0], result[1]

    def marginal(self, t):
        """Return marginal FPT density and CDF at absolute time(s) ``t``.

        The result is ``(density, cdf)`` with the same shape as ``t``. The CDF
        accumulates crossing mass across stages, avoiding a discontinuity
        from re-estimating total surviving mass on each stage's node grid.
        """
        times = np.asarray(t, dtype=float)
        if np.any((times < 0.0) | (times > self.deadline)):
            raise ValueError("t must lie within [0, deadline]")
        density = np.zeros_like(times, dtype=float)
        cdf = np.zeros_like(times, dtype=float)
        flat_times, flat_density, flat_cdf = times.ravel(), density.ravel(), cdf.ravel()
        stage_start_cdf = 0.0
        for stage in range(self.n_stages):
            left, right = self.breaks[stage], self.breaks[stage + 1]
            # At an exact breakpoint, use the preceding stage's endpoint.
            mask = (flat_times > left) & (flat_times <= right)
            needs_endpoint = np.any(flat_times > right)
            if not np.any(mask) and not needs_endpoint:
                break
            elapsed = flat_times[mask] - left
            duration = right - left
            # Carry all earlier crossing mass forward even when callers only
            # request times in a later stage. Do not depend on query ordering.
            evaluation_times = np.r_[elapsed, duration] if needs_endpoint else elapsed
            if stage == 0:
                stage_density = small_f(evaluation_times, self.mu[stage], self.sigma[stage], self._boundary_starts[stage], self.boundary_slopes[stage], duration, self.x0)
                passage = big_F(evaluation_times, self.mu[stage], self.sigma[stage], self._boundary_starts[stage], self.boundary_slopes[stage], duration, self.x0)
            else:
                stage_density, passage = self._stage_passage(stage, evaluation_times)
            flat_density[mask] = stage_density[:elapsed.size]
            flat_cdf[mask] = stage_start_cdf + passage[:elapsed.size]
            if needs_endpoint:
                stage_start_cdf += float(passage[-1])
        return density, np.clip(cdf, 0.0, 1.0)


class MultiStageRaceModel:
    """A race of independent :class:`RaceAccumulator` instances."""

    def __init__(self, accumulators: Sequence[RaceAccumulator]):
        self.accumulators = tuple(accumulators)
        if len(self.accumulators) < 2:
            raise ValueError("a race model needs at least two accumulators")
        deadlines = np.array([a.deadline for a in self.accumulators])
        if not np.allclose(deadlines, deadlines[0]):
            raise ValueError("all accumulators must have a common deadline")
        self.deadline = float(deadlines[0])

    @property
    def n_choices(self) -> int:
        return len(self.accumulators)

    def marginals(self, t):
        """Return ``(f, F)`` arrays with final dimension ``n_choices``."""
        values = [accumulator.marginal(t) for accumulator in self.accumulators]
        return np.stack([v[0] for v in values], axis=-1), np.stack([v[1] for v in values], axis=-1)

    def joint_density(self, t):
        """Joint density of response time and each zero-based winning choice."""
        density, cdf = self.marginals(t)
        survival = np.clip(1.0 - cdf, 0.0, 1.0)
        result = np.empty_like(density)
        for choice in range(self.n_choices):
            competitors = np.prod(np.delete(survival, choice, axis=-1), axis=-1)
            result[..., choice] = density[..., choice] * competitors
        return result

    def likelihood(self, choice: int, rt: float) -> float:
        """Likelihood for one trial; use choice ``-1`` for no response."""
        if not 0.0 <= rt <= self.deadline:
            raise ValueError("rt must lie within [0, deadline]")
        density, cdf = self.marginals(rt)
        survival = np.clip(1.0 - cdf, 0.0, 1.0)
        if choice == -1:
            return float(np.prod(survival))
        if not 0 <= choice < self.n_choices:
            raise ValueError("choice must be in [0, n_choices) or -1 for no response")
        return float(density[choice] * np.prod(np.delete(survival, choice)))

    def log_likelihood(self, choices, rts) -> float:
        """Total log likelihood for zero-based choices; ``-1`` means omission."""
        choices = np.asarray(choices, dtype=int)
        rts = np.asarray(rts, dtype=float)
        if choices.shape != rts.shape:
            raise ValueError("choices and rts must have matching shapes")
        likelihoods = np.array([self.likelihood(int(c), float(t)) for c, t in zip(choices.flat, rts.flat)])
        return float(np.sum(np.log(likelihoods)))

    def simulate(self, n_trials: int, dt: float = 1e-3, rng=None):
        """Forward Euler simulation returning ``(rt, choice)``.

        ``choice`` is zero-based, with ``-1`` for a trial not terminating by
        the deadline.  This is intended for validation against the numerical
        density, not as a replacement for the analytical likelihood.
        ``dt`` is the maximum step; steps end at every accumulator's stage
        breakpoints so a step never uses outdated drift or noise parameters.
        """
        if n_trials < 1 or dt <= 0.0:
            raise ValueError("n_trials and dt must be positive")
        generator = np.random.default_rng(rng)
        positions = np.array([a.x0 for a in self.accumulators], dtype=float)[:, None] * np.ones((self.n_choices, n_trials))
        rt = np.full(n_trials, self.deadline, dtype=float)
        choice = np.full(n_trials, -1, dtype=int)
        active = np.ones(n_trials, dtype=bool)
        stage_ends = np.unique(np.concatenate([a.breaks[1:] for a in self.accumulators]))
        t = 0.0
        while t < self.deadline and np.any(active):
            next_break = stage_ends[np.searchsorted(stage_ends, t, side="right")]
            end = min(t + dt, next_break, self.deadline)
            step = end - t
            active_index = np.flatnonzero(active)
            crossed = np.zeros((self.n_choices, active_index.size), dtype=bool)
            for i, accumulator in enumerate(self.accumulators):
                stage = min(np.searchsorted(accumulator.breaks[1:], t, side="right"), accumulator.n_stages - 1)
                positions[i, active_index] += accumulator.mu[stage] * step + accumulator.sigma[stage] * np.sqrt(step) * generator.standard_normal(active_index.size)
                crossed[i] = positions[i, active_index] >= accumulator.boundary(end)
            any_crossed = np.any(crossed, axis=0)
            if np.any(any_crossed):
                local = np.flatnonzero(any_crossed)
                winners = np.argmax(crossed[:, local], axis=0)
                trial_indices = active_index[local]
                # Midpoint time has smaller bias than assigning all crossings to end.
                rt[trial_indices] = t + step / 2.0
                choice[trial_indices] = winners
                active[trial_indices] = False
            t = end
        return rt, choice


__all__ = ["RaceAccumulator", "MultiStageRaceModel"]
