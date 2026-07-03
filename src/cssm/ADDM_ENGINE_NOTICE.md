# Vendored aDDM simulation engine — provenance & license

The simulation engine inside [`addm_models.pyx`](addm_models.pyx) — the inline
xoshiro256++/SplitMix64/Box-Muller PRNG and the stage-indexed
`_run_heterog_trial` / `_simulate_heterog_multistage` Euler–Maruyama kernel — is
**vendored verbatim from the efficient-fpt (efpt) project** (`src/efficient_fpt/
cython/simulator.pyx`). The aDDM covariate → per-stage array construction
(`_build_addm_mu_array_data`, `_generate_sacc_array_data`) is ported from
`src/efficient_fpt/addm_helpers.py`.

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
