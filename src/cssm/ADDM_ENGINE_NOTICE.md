# Vendored aDDM simulation engine — provenance & license

The simulation engine inside [`addm_models.pyx`](addm_models.pyx) — the inline
xoshiro256++/SplitMix64/Box-Muller PRNG and the stage-indexed
`_run_heterog_trial` / `_simulate_heterog_multistage` Euler–Maruyama kernel — is
**vendored verbatim from the efficient-fpt (efpt) project** (`src/efficient_fpt/
cython/simulator.pyx`). The aDDM covariate → per-stage drift construction
(`_build_addm_mu_array_data`) is ported from `src/efficient_fpt/addm_helpers.py`.
efpt's gamma-only fixation self-sampler (`_generate_sacc_array_data`) has been
generalized to any positive `scipy.stats` distribution and now lives in
`ssms/basic_simulators/fixation_continuation.py::generate_schedule` (pure-Python
glue, not vendored engine), shared by Mode-1 self-sampling and the `resample_all_fixations`
posterior-predictive continuation strategy.

This is the **same in-house source** as HSSM's JAX aDDM likelihood, so the
ssm-simulators aDDM simulator and the HSSM likelihood share one engine — the
motivation for vendoring rather than re-deriving.

> **Do not edit the vendored engine in place.** To update, re-vendor from upstream
> and re-apply the ssm-simulators glue (the `addm()` entry point, param contract,
> `setup_simulation`/`build_return_dict` wiring, float32 output cast).

## License & attribution

The vendored code is © 2025 Sicheng Liu, released under the MIT License — see
[`LICENSE.efpt`](LICENSE.efpt) in this directory, reproduced verbatim as MIT
requires. efficient-fpt is an in-house ecosystem project; the intended end-state
is to **absorb & relicense** this engine under ssm-simulators' own MIT license.

- **TODO (relicense):** obtain the author's sign-off to relicense under
  ssm-simulators' license; until then this MIT notice governs the vendored engine.

## Upstream

| | |
|---|---|
| Project | efficient-fpt |
| Commit | `d97a451479141acef845195610f0f9d85824844e` |
| Source | `src/efficient_fpt/cython/simulator.pyx`, `src/efficient_fpt/addm_helpers.py` |

## ssm-simulators-authored glue (re-apply on re-vendor)

1. Public `addm()` entry point with the canonical param contract
   `[eta, kappa, a, b, x0, t]` (+ `sigma`), dual self-sampled / covariate modes,
   and the collapsing `addm_collapse` boundary.
2. `float64` internals (efpt's dtype), cast to ssm-simulators' `float32` on output.
3. Output via `setup_simulation` / `build_return_dict` with the standard
   `(n_samples, n_trials, 1)` shape and the `-999.0` omission sentinel.
4. Inline xoshiro PRNG kept (not ssm-simulators' GSL RNG) so results are identical
   to efpt on the same per-trial seeds and independent of `n_threads`.

## Sanctioned in-place modification (re-apply on re-vendor)

The "do not edit in place" rule has **one** deliberate exception, needed so the
model-cartoon plotter can read a representative evidence path from simulator
metadata. It is opt-in and inert unless requested, so it does not change any efpt
result. Each edit site is tagged `# ssm-sim MOD` in `addm_models.pyx`:

- `_run_heterog_trial` gains a trailing `float *traj_out` parameter and, in the
  Euler loop, `if traj_out != NULL: traj_out[step] = <float>y`. When `traj_out` is
  `NULL` the loop is byte-for-byte the efpt original.
- `_simulate_heterog_multistage` gains `float[:, ::1] traj_out=None,
  int record_trial=-1`; it hands the sink to exactly one row (`record_trial`) and
  `NULL` to all others (`record_trial=-1` → records nothing, the efpt default).
- `addm()` passes `traj` (from `setup_simulation`, already `>= max_steps` rows) and
  `record_trial=0`, so row 0's path lands in `metadata['trajectory']`. `addm()` also
  emits `metadata['boundary']` (`+(a - b·t)` over the grid) and a relative
  `metadata['z']` start alias — both pure ssm-simulators glue, no engine change.

To re-vendor: drop in the upstream engine, then re-apply the two `# ssm-sim MOD`
sites above (the `addm()` glue is regenerated as part of items 1–4).
