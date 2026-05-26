import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ssms import OMISSION_SENTINEL
    import ssms.rl as rl

    return OMISSION_SENTINEL, mo, np, pd, plt, rl


@app.cell
def _(mo):
    mo.md("""
    # RLSSM Simulator Backend API Demo

    This notebook shows the newer `ssms.rl` interface for RLSSM simulation: explicit learning state, backend policy fields, simulator output with response-to-action mapping, and HSSM configuration export metadata.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Settings

    The simulator uses an angle decision process with a Rescorla-Wagner learning rule and a two-armed Bernoulli bandit. The SSM response labels are `[-1, 1]`, while learning updates use zero-based actions `[0, 1]`.
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        0.01,
        1.0,
        value=0.25,
        step=0.01,
        label="Learning rate (rl_alpha)",
    )
    scaler_slider = mo.ui.slider(
        0.1,
        6.0,
        value=2.0,
        step=0.1,
        label="Drift scaler",
    )
    trials_slider = mo.ui.slider(
        20,
        200,
        value=80,
        step=10,
        label="Trials",
    )
    seed_input = mo.ui.number(value=2026, start=0, stop=9999, label="Random seed")
    mo.vstack([alpha_slider, scaler_slider, trials_slider, seed_input])
    return alpha_slider, scaler_slider, seed_input, trials_slider


@app.cell
def _(mo):
    mo.md("""
    ## Compose the RLSSM
    """)
    return


@app.cell
def _(rl):
    learning_rule = rl.learning.RescorlaWagnerDeltaRule(
        n_actions=2,
        initial_q=0.5,
    )
    bandit_env = rl.env.Bandit.bernoulli(
        probabilities=[0.7, 0.3],
        response_labels=[-1, 1],
    )
    model_config = rl.ModelConfig(
        model_name="rlssm_angle_backend_api_demo",
        description="RW delta rule + angle SSM + Bernoulli bandit",
        decision_process="angle",
        learning_process=learning_rule,
        task_environment=bandit_env,
        response_mapping="auto",
        learning_backend="python",
        gradient="auto",
        include_action=True,
    )
    model_config.validate()
    return (model_config,)


@app.cell
def _(mo, model_config):
    mo.md(
        "The resolved learning backend is "
        f"**`{model_config.resolved_learning_backend}`** and the resolved "
        f"gradient policy is **`{model_config.resolved_gradient}`**.\n\n"
        "The automatic response mapping is:\n\n"
        "```python\n"
        f"{model_config.response_to_action}\n"
        "```"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Explicit Learning State
    """)
    return


@app.cell
def _(alpha_slider, model_config, np, pd, scaler_slider):
    backend_params = {
        "rl_alpha": float(alpha_slider.value),
        "scaler": float(scaler_slider.value),
    }
    state_initial = model_config.learning_process.init_state()
    computed_initial = model_config.learning_process.compute_python(
        state_initial,
        backend_params,
    )
    state_after_reward = model_config.learning_process.update_python(
        state_initial,
        action=0,
        reward=1.0,
        trial_params=backend_params,
    )
    explicit_state_table = pd.DataFrame(
        [
            {
                "step": "initial",
                "q_left": state_initial["q_values"][0],
                "q_right": state_initial["q_values"][1],
                "computed_v": computed_initial["v"],
            },
            {
                "step": "after action=0, reward=1",
                "q_left": state_after_reward["q_values"][0],
                "q_right": state_after_reward["q_values"][1],
                "computed_v": model_config.learning_process.compute_python(
                    state_after_reward,
                    backend_params,
                )["v"],
            },
        ]
    )
    explicit_state_table[["q_left", "q_right", "computed_v"]] = explicit_state_table[
        ["q_left", "q_right", "computed_v"]
    ].astype(float)
    np.asarray(state_initial["q_values"])
    explicit_state_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Simulate Behavioral Data
    """)
    return


@app.cell
def _(
    alpha_slider,
    model_config,
    rl,
    scaler_slider,
    seed_input,
    trials_slider,
):
    simulator = rl.Simulator(model_config)
    theta = {
        "rl_alpha": float(alpha_slider.value),
        "scaler": float(scaler_slider.value),
        "a": 1.5,
        "z": 0.5,
        "t": 0.3,
        "theta": 0.2,
    }
    simulated_data = simulator.simulate(
        theta=theta,
        n_trials=int(trials_slider.value),
        n_participants=3,
        random_state=int(seed_input.value),
    )
    simulated_data.head(10)
    return simulated_data, theta


@app.cell
def _(mo, simulated_data):
    mo.md(
        f"The simulated panel has **{len(simulated_data)} rows**. It includes "
        "both the raw SSM `response` label and the derived zero-based `action` "
        "column because `include_action=True`."
    )
    return


@app.cell
def _(OMISSION_SENTINEL, model_config, pd, simulated_data, theta):
    participant_zero = simulated_data[simulated_data["participant_id"] == 0].copy()
    replay_state = model_config.learning_process.init_state()
    replay_rows = []
    learning_params = {
        "rl_alpha": theta["rl_alpha"],
        "scaler": theta["scaler"],
    }
    for trial in participant_zero.itertuples(index=False):
        if trial.rt == OMISSION_SENTINEL:
            replay_rows.append(
                {
                    "trial_id": trial.trial_id,
                    "q_left": replay_state["q_values"][0],
                    "q_right": replay_state["q_values"][1],
                    "computed_v": None,
                    "action": None,
                    "feedback": trial.feedback,
                }
            )
            continue

        replay_computed = model_config.learning_process.compute_python(
            replay_state,
            learning_params,
        )
        replay_rows.append(
            {
                "trial_id": trial.trial_id,
                "q_left": replay_state["q_values"][0],
                "q_right": replay_state["q_values"][1],
                "computed_v": replay_computed["v"],
                "action": trial.action,
                "feedback": trial.feedback,
            }
        )
        replay_state = model_config.learning_process.update_python(
            replay_state,
            action=int(trial.action),
            reward=float(trial.feedback),
            trial_params=learning_params,
        )

    replay_table = pd.DataFrame(replay_rows)
    replay_table.head(10)
    return participant_zero, replay_table


@app.cell
def _(mo):
    mo.md("""
    ## Replay Q-Values and Drift
    """)
    return


@app.cell
def _(participant_zero, plt, replay_table):
    replay_fig, replay_axes = plt.subplots(1, 2, figsize=(10, 3.5))
    replay_axes[0].plot(
        replay_table["trial_id"], replay_table["q_left"], label="Q(action 0)"
    )
    replay_axes[0].plot(
        replay_table["trial_id"], replay_table["q_right"], label="Q(action 1)"
    )
    replay_axes[0].set_xlabel("Trial")
    replay_axes[0].set_ylabel("Q-value")
    replay_axes[0].legend()
    replay_axes[0].set_title("Explicit learning state")

    replay_axes[1].scatter(
        participant_zero["rt"],
        participant_zero["response"],
        c=participant_zero["action"],
        cmap="viridis",
        alpha=0.75,
    )
    replay_axes[1].set_xlabel("RT")
    replay_axes[1].set_ylabel("SSM response label")
    replay_axes[1].set_title("Response labels mapped to actions")
    replay_fig.tight_layout()
    replay_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## HSSM Configuration Bridge
    """)
    return


@app.cell
def _(mo, model_config, pd):
    hssm_config_dict = model_config.to_hssm_config_dict()
    hssm_metadata = pd.DataFrame(
        [
            {
                "field": "learning_backend",
                "value": hssm_config_dict["learning_backend"],
            },
            {"field": "gradient", "value": hssm_config_dict["gradient"]},
            {
                "field": "learning_process_kind",
                "value": hssm_config_dict["learning_process_kind"],
            },
            {
                "field": "decision_process_loglik_kind",
                "value": hssm_config_dict["decision_process_loglik_kind"],
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                "`to_hssm_config_dict()` now exports backend metadata alongside "
                "the structural RLSSM fields. Python-only learning processes "
                "are treated as black-box/no-gradient by default; JAX-backed "
                "learning processes can advertise differentiable support when "
                "available."
            ),
            hssm_metadata,
        ]
    )
    return


@app.cell
def _(mo, model_config):
    mo.md(
        "## Optional JAX Backend\n\n"
        "This learning rule declares available backends as "
        f"`{model_config.learning_process.available_backends}`. In an environment "
        'with JAX installed, `learning_backend="auto"` can resolve to `"jax"`. '
        'Requesting `learning_backend="jax"` without JAX installed raises a clear '
        "error instead of silently falling back to Python."
    )
    return


if __name__ == "__main__":
    app.run()
