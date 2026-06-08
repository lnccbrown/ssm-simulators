"""Tests for RLSSM data contract validation."""

import io
import contextlib

import numpy as np
import pandas as pd
import pytest

from ssms import OMISSION_SENTINEL
import ssms.rl as rl
from ssms.rl.simulator import MISSING_RESPONSE_SENTINEL
from ssms.rl.validation import DataValidationReport, validate_rlssm_data


def _make_config(**overrides):
    defaults = dict(
        model_name="test_rlssm",
        description="Test RLSSM",
        decision_process="angle",
        learning_process=rl.learning.RescorlaWagnerDeltaRule(
            n_actions=2, initial_q=0.5
        ),
        task_environment=rl.env.Bandit.bernoulli(
            probabilities=[0.7, 0.3], response_labels=[-1, 1]
        ),
    )
    defaults.update(overrides)
    return rl.ModelConfig(**defaults)


class _NoFeedbackEnvironment:
    n_arms = 2
    context_fields = []

    @property
    def response_labels(self):
        return [-1, 1]

    def reset(self, rng=None):
        pass

    def get_trial_context(self, trial_idx):
        return {}

    def sample_context(self, context, trial_idx):
        return {}


def _valid_panel(*, n_participants: int = 2, n_trials: int = 3) -> pd.DataFrame:
    rows = []
    for participant_id in range(n_participants):
        for trial_id in range(n_trials):
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_id": trial_id,
                    "rt": 0.5 + 0.01 * trial_id,
                    "response": -1 if trial_id % 2 == 0 else 1,
                    "feedback": float(trial_id % 2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def config():
    return _make_config()


class TestHappyPath:
    def test_manual_panel_passes(self, config):
        data = _valid_panel()
        report = config.validate_data(data)

        assert report.ok
        assert report.n_participants == 2
        assert report.n_trials == 3
        assert not any(issue.level == "error" for issue in report.issues)

    def test_simulated_output_passes_without_omissions(self):
        config = rl.preset.get("2AB_RW_Angle")
        sim = rl.Simulator(config)
        data = sim.simulate(
            theta={
                "rl_alpha": 0.2,
                "scaler": 2.0,
                "a": 1.5,
                "z": 0.5,
                "t": 0.3,
                "theta": 0.2,
            },
            n_trials=20,
            n_participants=3,
            random_state=42,
        )
        clean = data[
            (data["rt"] != OMISSION_SENTINEL)
            & (data["response"] != MISSING_RESPONSE_SENTINEL)
        ].reset_index(drop=True)

        # Reindex to contiguous trial ids after dropping omissions.
        reindexed = []
        for participant_id, group in clean.groupby("participant_id", sort=True):
            block = group.sort_values("trial_id").reset_index(drop=True)
            block["trial_id"] = np.arange(len(block))
            reindexed.append(block)
        clean = pd.concat(reindexed, ignore_index=True)

        report = config.validate_data(clean)
        assert report.ok
        assert report.n_participants == 3

    def test_validate_rlssm_data_matches_wrapper(self, config):
        data = _valid_panel()
        assert validate_rlssm_data(config, data).ok
        assert config.validate_data(data).ok


class TestColumns:
    def test_empty_dataframe(self, config):
        report = config.validate_data(pd.DataFrame())

        assert not report.ok
        assert any(issue.code == "empty_data" for issue in report.issues)

    def test_missing_rt(self, config):
        data = _valid_panel().drop(columns=["rt"])
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "missing_column" for issue in report.issues)

    def test_missing_feedback_with_hint(self, config):
        data = _valid_panel().rename(columns={"feedback": "reward"})
        report = config.validate_data(data)

        assert not report.ok
        missing = [i for i in report.issues if i.code == "missing_column"]
        assert any("feedback" in issue.message for issue in missing)
        assert any(issue.hint and "context_fields" in issue.hint for issue in missing)

    def test_missing_participant_id(self, config):
        data = _valid_panel().drop(columns=["participant_id"])
        report = config.validate_data(data)

        assert not report.ok

    def test_extra_columns_warn_only(self, config):
        data = _valid_panel()
        data["notes"] = "x"
        report = config.validate_data(data)

        assert report.ok
        assert any(issue.code == "extra_columns" for issue in report.issues)


class TestPanelStructure:
    def test_null_participant_id(self, config):
        data = _valid_panel()
        data.loc[0, "participant_id"] = np.nan
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "null_participant_id" for issue in report.issues)

    def test_duplicate_keys(self, config):
        data = _valid_panel()
        duplicate = data.iloc[[0]].copy()
        data = pd.concat([data, duplicate], ignore_index=True)
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "duplicate_keys" for issue in report.issues)

    def test_unbalanced_panel(self, config):
        participant_0 = _valid_panel(n_participants=1, n_trials=2)
        participant_1 = _valid_panel(n_participants=1, n_trials=4)
        participant_1["participant_id"] = 1
        data = pd.concat([participant_0, participant_1], ignore_index=True)
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "unbalanced_panel" for issue in report.issues)

    def test_invalid_trial_ids(self, config):
        data = _valid_panel()
        data.loc[data["participant_id"] == 0, "trial_id"] = [0, 2, 0]
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "invalid_trial_ids" for issue in report.issues)

    def test_interleaved_participants(self, config):
        data = pd.DataFrame(
            {
                "participant_id": [0, 1, 0, 1],
                "trial_id": [0, 0, 1, 1],
                "rt": [0.5, 0.6, 0.7, 0.8],
                "response": [-1, 1, 1, -1],
                "feedback": [1.0, 0.0, 0.0, 1.0],
            }
        )
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "interleaved_participants" for issue in report.issues)

    def test_unsorted_rows(self, config):
        data = _valid_panel().sort_values("rt").reset_index(drop=True)
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "unsorted_rows" for issue in report.issues)


class TestValues:
    def test_nan_in_required_column(self, config):
        data = _valid_panel()
        data.loc[0, "rt"] = np.nan
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "missing_values" for issue in report.issues)

    def test_invalid_response_label(self, config):
        data = _valid_panel()
        data.loc[0, "response"] = 99
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "invalid_response_labels" for issue in report.issues)

    def test_unmapped_response_label(self, config):
        data = _valid_panel()
        config.response_to_choice = {-1: 0}

        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "unmapped_response_labels" for issue in report.issues)

    def test_omission_rt(self, config):
        data = _valid_panel()
        data.loc[0, "rt"] = OMISSION_SENTINEL
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "omission_rt" for issue in report.issues)

    def test_omission_response(self, config):
        data = _valid_panel()
        data.loc[0, "response"] = MISSING_RESPONSE_SENTINEL
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "omission_response" for issue in report.issues)

    def test_all_omissions_skip_response_and_rt_value_checks(self, config):
        data = _valid_panel()
        data["rt"] = OMISSION_SENTINEL
        data["response"] = MISSING_RESPONSE_SENTINEL

        report = config.validate_data(data)

        codes = {issue.code for issue in report.issues}
        assert {"omission_rt", "omission_response"}.issubset(codes)
        assert "invalid_response_labels" not in codes
        assert "non_positive_rt" not in codes

    def test_non_finite_rt(self, config):
        data = _valid_panel()
        data.loc[0, "rt"] = np.inf

        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "invalid_rt" for issue in report.issues)

    def test_non_positive_rt(self, config):
        data = _valid_panel()
        data.loc[0, "rt"] = 0.0
        report = config.validate_data(data)

        assert not report.ok
        assert any(issue.code == "non_positive_rt" for issue in report.issues)


class TestReportAPI:
    def test_raise_for_errors(self, config):
        report = config.validate_data(_valid_panel().drop(columns=["rt"]))
        with pytest.raises(ValueError, match="missing_column"):
            report.raise_for_errors()

    def test_warnings_do_not_raise(self, config):
        data = _valid_panel()
        data["extra_col"] = 1
        report = config.validate_data(data)
        report.raise_for_errors()

    def test_print_smoke(self, config):
        report = DataValidationReport()
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            report.print()
        assert "OK" in captured.getvalue()

        report = config.validate_data(_valid_panel().drop(columns=["rt"]))
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            report.print()
        assert "missing_column" in captured.getvalue()

    def test_print_includes_panel_shape_for_valid_report(self, config):
        report = config.validate_data(_valid_panel(n_participants=2, n_trials=3))

        with contextlib.redirect_stdout(io.StringIO()) as captured:
            report.print()

        output = captured.getvalue()
        assert "participants=2" in output
        assert "trials_per_participant=3" in output

    def test_invalid_input_type(self, config):
        report = validate_rlssm_data(config, [[1, 2]])
        assert not report.ok
        assert any(issue.code == "invalid_type" for issue in report.issues)


class TestOutcomeFreeConfig:
    def test_outcome_free_panel(self):
        class ChoiceOnlyLearning:
            computed_params = ["v"]
            free_params = ["bias"]
            param_bounds = {"bias": (-5.0, 5.0)}
            default_params = {"bias": 0.0}
            available_backends = ("python",)
            supports_gradient = False
            n_actions = 2
            required_context_fields = ["choice"]

            def init_state(self):
                return {}

            def compute_python(self, state, trial_params, context):
                return {"v": trial_params["bias"]}

            def update_python(self, state, trial_params, context):
                return state

        config = _make_config(
            learning_process=ChoiceOnlyLearning(),
            task_environment=_NoFeedbackEnvironment(),
            context_fields=[],
        )
        data = _valid_panel().drop(columns=["feedback"])
        report = config.validate_data(data)

        assert report.ok
