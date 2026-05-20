"""Simulator — interleaved learning + SSM decision simulation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ssms.basic_simulators import OMISSION_SENTINEL
from ssms.basic_simulators.simulator import simulator as ssm_simulator

from .config import ModelConfig
from .env import TaskEnvironment


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
        theta: dict[str, float],
        n_trials: int = 200,
        n_participants: int = 20,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Run full RLSSM simulation.

        Parameters
        ----------
        theta : dict[str, float]
            Concrete parameter values. Must contain all params required by the
            learning process and fixed SSM parameters.
        n_trials : int
            Number of trials per participant. Default 200.
        n_participants : int
            Number of participants to simulate. Default 20.
        random_state : int | None
            Seed for reproducibility. If None, non-deterministic.

        Returns
        -------
        pd.DataFrame
            Balanced panel with columns: participant_id, trial_id, rt, response,
            feedback, plus any extra_fields from the task environment.
        """
        self._validate_theta(theta)

        rng = np.random.default_rng(random_state)
        child_rngs = rng.spawn(n_participants)

        all_rows = []
        for p in range(n_participants):
            rows = self._simulate_subject(p, theta, n_trials, child_rngs[p])
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df = df.sort_values(["participant_id", "trial_id"]).reset_index(drop=True)
        return df

    def _validate_theta(self, theta: dict[str, float]) -> None:
        """Check that theta contains all required params."""
        required_params = self.config.required_params
        missing = [p for p in required_params if p not in theta]
        if missing:
            raise ValueError(
                f"theta is missing required params: {missing}. "
                f"Expected all of: {required_params}"
            )

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

        # Reset learning process and task environment
        lp.reset()
        env.reset(rng=rng)

        rows = []
        for t in range(n_trials):
            # COMPUTE: learning process produces SSM params from current state
            computed_raw = lp.compute_ssm_params(rl_params)

            # Apply mapping: learning output name -> SSM param name
            computed_ssm = {}
            for output_name, value in computed_raw.items():
                ssm_name = mapping.get(output_name, output_name)
                computed_ssm[ssm_name] = value

            # MERGE: fixed SSM params + computed SSM params
            full_theta = {**fixed_ssm_params, **computed_ssm}

            # SIMULATE: one SSM trial
            trial_seed = int(rng.integers(0, 2**31))
            result = ssm_simulator(
                theta=full_theta,
                model=config.decision_process,
                n_samples=1,
                random_state=trial_seed,
                **config.ssm_kwargs,
            )

            rt = float(result["rts"].item())
            ssm_choice = int(result["choices"].item())

            # OMISSION CHECK
            if rt == OMISSION_SENTINEL:
                row = {
                    "participant_id": subject_id,
                    "trial_id": t,
                    "rt": OMISSION_SENTINEL,
                    "response": MISSING_RESPONSE_SENTINEL,
                    "feedback": 0.0,
                }
                if config.include_action:
                    row["action"] = MISSING_RESPONSE_SENTINEL
                row.update(env.get_extra_data(t))
                rows.append(row)
                continue

            # Use the SSM choice label as the recorded response, but convert it
            # to a zero-based action index for the task environment and learning rule.
            response = ssm_choice
            action = self._response_to_action_index(response)

            # REWARD
            reward = env.sample_reward(action, t)

            # UPDATE learning process
            lp.update(action, reward, rl_params)

            # RECORD
            row = {
                "participant_id": subject_id,
                "trial_id": t,
                "rt": rt,
                "response": response,
                "feedback": reward,
            }
            if config.include_action:
                row["action"] = action
            row.update(env.get_extra_data(t))
            rows.append(row)

        return rows

    def _response_to_action_index(self, response: int) -> int:
        """Map an SSM response label to the learning process action index."""
        response_to_action = self.config.response_to_action
        if response not in response_to_action:
            raise ValueError(
                f"SSM response {response} is not in response_mapping. "
                f"Expected one of: {sorted(response_to_action)}."
            )
        return int(response_to_action[response])
