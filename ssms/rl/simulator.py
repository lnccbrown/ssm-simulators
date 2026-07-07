"""Simulator — interleaved learning + SSM decision simulation."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ssms.basic_simulators import OMISSION_SENTINEL
from ssms.basic_simulators.simulator import simulator as ssm_simulator

from .config import ModelConfig
from .env import TaskEnvironment
from .validation import validate_rlssm_data


MISSING_RESPONSE_SENTINEL = -999


class Simulator:
    """RLSSM simulator composing a learning process with an SSM decision process.

    Runs the interleaved trial-by-trial loop:
    compute SSM params -> simulate SSM -> observe choice -> generate reward -> update learning.

    Reuses the existing ssm-simulators ``simulator()`` function with ``n_samples=1``
    for each trial. No Cython modifications needed — all 40+ SSM models work as
    decision processes out of the box.

    Parameters
    ----------
    config : ModelConfig
        Structural model configuration. Validated on construction.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        config.validate()

    def simulate(
        self,
        theta: dict[str, Any],
        n_trials: int = 200,
        n_participants: int | None = None,
        random_state: int | None = None,
        mode: Literal["generative", "ppc"] = "generative",
        observed_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run full RLSSM simulation.

        Parameters
        ----------
        theta : dict[str, Any]
            Concrete parameter values. Must contain all params required by the
            learning process and fixed SSM parameters. Each value can be a scalar
            shared by all participants or a one-dimensional list/array with one
            value per participant.
        n_trials : int
            Number of trials per participant. Default 200.
        n_participants : int | None
            Number of participants to simulate. If None, inferred from
            participant-wise theta values when present; otherwise defaults to 20.
        random_state : int | None
            Seed for reproducibility. If None, non-deterministic.
        mode : {"generative", "ppc"}
            Simulation mode. ``"generative"`` runs the unconstrained simulator
            loop. ``"ppc"`` runs observed-history-conditioned posterior
            predictive simulation.
        observed_data : pd.DataFrame | None
            Observed participant history required for ``mode="ppc"``.

        Returns
        -------
        pd.DataFrame
            Balanced panel with columns: participant_id, trial_id, rt, response,
            configured context fields, and optional derived choice.
        """
        self._validate_mode(mode)
        if mode == "ppc":
            return self._simulate_ppc(
                theta=theta,
                observed_data=observed_data,
                n_participants=n_participants,
                random_state=random_state,
            )

        participant_theta, resolved_n_participants = self._expand_participant_theta(
            theta, n_participants
        )

        rng = np.random.default_rng(random_state)
        child_rngs = rng.spawn(resolved_n_participants)

        all_rows = []
        for p in range(resolved_n_participants):
            rows = self._simulate_subject(
                p, participant_theta[p], n_trials, child_rngs[p]
            )
            all_rows.extend(rows)

        return pd.DataFrame(all_rows).reset_index(drop=True)

    def _validate_mode(self, mode: str) -> None:
        """Validate the public simulation mode."""
        if mode not in {"generative", "ppc"}:
            raise ValueError(
                f"mode must be one of 'generative' or 'ppc'. Got {mode!r}."
            )

    def _simulate_ppc(
        self,
        theta: dict[str, Any],
        observed_data: pd.DataFrame | None,
        n_participants: int | None,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Run observed-history-conditioned posterior predictive simulation."""
        observed, participant_ids = self._validate_observed_data(observed_data)
        observed_n_participants = len(participant_ids)
        if (
            n_participants is not None
            and int(n_participants) != observed_n_participants
        ):
            raise ValueError(
                f"n_participants={n_participants} does not match observed_data "
                f"participant count {observed_n_participants}."
            )

        participant_theta, _ = self._expand_participant_theta(
            theta, observed_n_participants
        )
        rng = np.random.default_rng(random_state)
        child_rngs = rng.spawn(observed_n_participants)

        all_rows = []
        for participant_idx, participant_id in enumerate(participant_ids):
            observed_subject = observed[observed["participant_id"] == participant_id]
            rows = self._simulate_subject_ppc(
                participant_id,
                participant_theta[participant_idx],
                observed_subject,
                child_rngs[participant_idx],
            )
            all_rows.extend(rows)

        return pd.DataFrame(all_rows).reset_index(drop=True)

    def _validate_observed_data(
        self, observed_data: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, list[Any]]:
        """Validate observed participant history for PPC mode."""
        if observed_data is None:
            raise ValueError("observed_data is required when mode='ppc'.")
        report = validate_rlssm_data(self.config, observed_data)
        report.raise_for_errors()

        observed = observed_data.reset_index(drop=True)
        participant_ids = observed["participant_id"].unique().tolist()
        return observed, participant_ids

    def _validate_theta_keys(self, theta: dict[str, Any]) -> None:
        """Check that theta contains exactly the required params."""
        required_params = self.config.required_params
        missing = [p for p in required_params if p not in theta]
        if missing:
            raise ValueError(
                f"theta is missing required params: {missing}. "
                f"Expected all of: {required_params}"
            )
        unknown = sorted(set(theta) - set(required_params))
        if unknown:
            raise ValueError(
                f"theta contains unknown params: {unknown}. "
                f"Expected only: {required_params}"
            )

    def _expand_participant_theta(
        self,
        theta: dict[str, Any],
        n_participants: int | None,
    ) -> tuple[list[dict[str, float]], int]:
        """Expand scalar or participant-wise theta into one dict per participant.

        Downstream learning backends and SSM simulators receive scalar parameter
        values for a single participant. This method accepts the public API's
        scalar-or-vector theta form and tiles it into one scalar dict per
        participant.
        """
        self._validate_theta_keys(theta)
        if n_participants is not None:
            if not isinstance(n_participants, Integral) or n_participants <= 0:
                raise ValueError(
                    "n_participants must be a positive integer or None. "
                    f"Got {n_participants!r}."
                )
            n_participants = int(n_participants)

        scalar_values: dict[str, float] = {}
        participant_values: dict[str, list[float]] = {}
        participant_lengths: dict[str, int] = {}

        for param in self.config.required_params:
            values = self._theta_value_to_array(param, theta[param])
            if values.ndim == 0:
                scalar_values[param] = float(values.item())
                continue
            if values.ndim != 1:
                raise ValueError(
                    f"theta[{param!r}] must be a scalar or a one-dimensional "
                    f"participant-wise value. Got shape {values.shape}."
                )
            if values.size == 0:
                raise ValueError(
                    f"theta[{param!r}] participant-wise value must not be empty."
                )
            participant_values[param] = [float(value) for value in values]
            participant_lengths[param] = int(values.size)

        unique_lengths = sorted(set(participant_lengths.values()))
        if len(unique_lengths) > 1:
            raise ValueError(
                "participant-wise theta values must all have the same length. "
                f"Got lengths by param: {participant_lengths}."
            )

        inferred_n_participants = unique_lengths[0] if unique_lengths else None
        if n_participants is None:
            resolved_n_participants = inferred_n_participants or 20
        elif (
            inferred_n_participants is not None
            and n_participants != inferred_n_participants
        ):
            raise ValueError(
                f"n_participants={n_participants} does not match the "
                f"participant-wise theta length {inferred_n_participants}."
            )
        else:
            resolved_n_participants = n_participants

        participant_theta = []
        for participant_idx in range(resolved_n_participants):
            values_for_participant = {}
            for param in self.config.required_params:
                if param in participant_values:
                    values_for_participant[param] = participant_values[param][
                        participant_idx
                    ]
                else:
                    values_for_participant[param] = scalar_values[param]
            participant_theta.append(values_for_participant)

        return participant_theta, resolved_n_participants

    def _theta_value_to_array(self, param: str, value: Any) -> np.ndarray:
        """Convert a public theta value to a numeric ndarray for validation."""
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"theta[{param!r}] must be numeric and scalar or one-dimensional."
            ) from err

    def _simulate_subject(
        self,
        subject_id: int,
        theta: dict[str, float],
        n_trials: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Simulate one participant's trial sequence."""
        config = self.config
        lp = config.learning_process
        env = cast(TaskEnvironment, config.task_environment)

        # Split theta into RL params and fixed SSM params
        rl_params = {k: theta[k] for k in lp.free_params}
        fixed_ssm_params = {k: theta[k] for k in config._fixed_ssm_params}

        # Build the computed_param_mapping (learning output -> SSM param name)
        mapping = config.computed_param_mapping or {}

        # Initialize explicit learning state and task environment.
        learning_state = self._init_learning_state()
        env.reset(rng=rng)

        rows = []
        for t in range(n_trials):
            pre_context = dict(env.get_trial_context(t))
            # COMPUTE: learning process produces SSM params from current state
            computed_raw = self._compute_learning_params(
                learning_state, rl_params, pre_context
            )

            # Apply mapping: learning output name -> SSM param name
            computed_ssm = {}
            for output_name, value in computed_raw.items():
                ssm_name = mapping.get(output_name, output_name)
                computed_ssm[ssm_name] = float(value)

            # MERGE: fixed SSM params + computed SSM params
            full_theta = {**fixed_ssm_params, **computed_ssm}

            # SIMULATE: one SSM trial
            rt, ssm_choice = self._simulate_decision_trial(full_theta, rng)

            # OMISSION CHECK
            if rt == OMISSION_SENTINEL:
                row = {
                    "participant_id": subject_id,
                    "trial_id": t,
                    "rt": OMISSION_SENTINEL,
                    "response": MISSING_RESPONSE_SENTINEL,
                }
                row.update(self._context_fields_for_output(pre_context))
                if config.include_choice:
                    row["choice"] = MISSING_RESPONSE_SENTINEL
                rows.append(row)
                continue

            # Use the SSM choice label as the recorded response, but convert it
            # to a zero-based choice index for the task environment and learning rule.
            response = ssm_choice
            choice = self._response_to_choice_index(response)

            context = {
                **pre_context,
                "rt": rt,
                "response": response,
                "choice": choice,
            }
            context.update(env.sample_context(context, t))

            # UPDATE learning process
            learning_state = self._update_learning_state(
                learning_state, rl_params, context
            )
            self._store_learning_state(learning_state)

            # RECORD
            row = {
                "participant_id": subject_id,
                "trial_id": t,
                "rt": rt,
                "response": response,
            }
            row.update(self._context_fields_for_output(context))
            if config.include_choice:
                row["choice"] = choice
            rows.append(row)

        return rows

    def _simulate_subject_ppc(
        self,
        subject_id: Any,
        theta: dict[str, float],
        observed_subject: pd.DataFrame,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Simulate one participant conditioned on observed trial history."""
        config = self.config
        lp = config.learning_process
        env = cast(TaskEnvironment, config.task_environment)

        rl_params = {k: theta[k] for k in lp.free_params}
        fixed_ssm_params = {k: theta[k] for k in config._fixed_ssm_params}
        mapping = config.computed_param_mapping or {}

        learning_state = self._init_learning_state()
        env.reset(rng=rng)

        rows = []
        for t, observed_trial in enumerate(observed_subject.itertuples(index=False)):
            observed_context = self._observed_context(observed_trial)
            computed_raw = self._compute_learning_params(
                learning_state, rl_params, observed_context
            )

            computed_ssm = {}
            for output_name, value in computed_raw.items():
                ssm_name = mapping.get(output_name, output_name)
                computed_ssm[ssm_name] = float(value)

            full_theta = {**fixed_ssm_params, **computed_ssm}
            rt, ssm_choice = self._simulate_decision_trial(full_theta, rng)

            if rt == OMISSION_SENTINEL:
                response = MISSING_RESPONSE_SENTINEL
                simulated_choice = MISSING_RESPONSE_SENTINEL
            else:
                response = ssm_choice
                simulated_choice = self._response_to_choice_index(response)

            observed_response = int(getattr(observed_trial, "response"))
            observed_choice = self._response_to_choice_index(observed_response)
            update_context = {
                **observed_context,
                "response": observed_response,
                "choice": observed_choice,
            }
            if hasattr(observed_trial, "rt"):
                update_context["rt"] = float(getattr(observed_trial, "rt"))

            learning_state = self._update_learning_state(
                learning_state, rl_params, update_context
            )
            self._store_learning_state(learning_state)

            row = {
                "participant_id": subject_id,
                "trial_id": getattr(observed_trial, "trial_id"),
                "rt": rt,
                "response": response,
            }
            row.update(self._context_fields_for_output(observed_context))
            if config.include_choice:
                row["choice"] = simulated_choice
            rows.append(row)

        return rows

    def _simulate_decision_trial(
        self, theta: dict[str, float], rng: np.random.Generator
    ) -> tuple[float, int]:
        """Simulate one SSM decision trial and unpack scalar RT/choice."""
        trial_seed = int(rng.integers(0, 2**31))
        result = ssm_simulator(
            theta=theta,
            model=self.config.decision_process,
            n_samples=1,
            random_state=trial_seed,
            **self.config.ssm_kwargs,
        )
        return float(result["rts"].item()), int(result["choices"].item())

    def _init_learning_state(self):
        """Initialize participant learning state for the configured backend."""
        lp = self.config.learning_process
        backend = self.config.resolved_learning_backend
        if backend == "jax" and hasattr(lp, "init_jax_state"):
            state = lp.init_jax_state()
        elif hasattr(lp, "init_state"):
            state = lp.init_state()
        else:
            lp.reset()
            state = None
        self._store_learning_state(state)
        return state

    def _compute_learning_params(
        self, state, rl_params: dict[str, float], context: dict
    ):
        """Compute SSM params from the current explicit learning state."""
        lp = self.config.learning_process
        backend = self.config.resolved_learning_backend
        if backend == "jax" and hasattr(lp, "compute_jax"):
            return lp.compute_jax(state, rl_params, context)
        if hasattr(lp, "compute_python"):
            return lp.compute_python(state, rl_params, context)
        return lp.compute_ssm_params(rl_params)

    def _update_learning_state(self, state, rl_params: dict[str, float], context: dict):
        """Return the next learning state after observing one trial context."""
        lp = self.config.learning_process
        backend = self.config.resolved_learning_backend
        if backend == "jax" and hasattr(lp, "update_jax"):
            return lp.update_jax(state, rl_params, context)
        if hasattr(lp, "update_python"):
            return lp.update_python(state, rl_params, context)
        lp.update(context["choice"], context.get("feedback", 0.0), rl_params)
        return state

    def _store_learning_state(self, state) -> None:
        """Keep built-in mutable inspection properties in sync during transition."""
        if hasattr(self.config.learning_process, "_state"):
            self.config.learning_process._state = state

    def _response_to_choice_index(self, response: int) -> int:
        """Map an SSM response label to the learning process choice index."""
        response_to_choice = self.config.response_to_choice
        if response not in response_to_choice:
            raise ValueError(
                f"SSM response {response} is not in response_to_choice. "
                f"Expected one of: {sorted(response_to_choice)}."
            )
        return int(response_to_choice[response])

    def _context_fields_for_output(self, context: dict) -> dict:
        """Return configured observable context fields in stable config order."""
        return {
            field_name: context.get(field_name, 0.0)
            for field_name in (self.config.context_fields or [])
        }

    def _observed_context(self, observed_trial) -> dict:
        """Extract configured context fields from an observed trial row."""
        return {
            field_name: getattr(observed_trial, field_name)
            for field_name in (self.config.context_fields or [])
        }
