"""Data contract validation for RLSSM panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ssms.basic_simulators import OMISSION_SENTINEL

from .config import ModelConfig

MISSING_RESPONSE_SENTINEL = -999

PARTICIPANT_COL = "participant_id"
TRIAL_COL = "trial_id"
RESPONSE_COL = "response"
RT_COL = "rt"

# Common outcome-like column names suggested when a required context field is missing.
_OUTCOME_NAME_HINTS = ("feedback", "reward", "outcome")


@dataclass(frozen=True)
class DataValidationIssue:
    """A single validation finding."""

    level: Literal["error", "warning"]
    code: str
    message: str
    hint: str | None = None


@dataclass
class DataValidationReport:
    """Aggregated validation results for an RLSSM data panel."""

    issues: list[DataValidationIssue] = field(default_factory=list)
    n_participants: int | None = None
    n_trials: int | None = None

    @property
    def ok(self) -> bool:
        """True when there are no error-level issues."""
        return not any(issue.level == "error" for issue in self.issues)

    def print(self) -> None:
        """Print a human-readable summary to stdout."""
        if not self.issues:
            print("RLSSM data validation: OK")
            if self.n_participants is not None and self.n_trials is not None:
                print(
                    f"  participants={self.n_participants}, "
                    f"trials_per_participant={self.n_trials}"
                )
            return

        print("RLSSM data validation report:")
        for issue in self.issues:
            prefix = issue.level.upper()
            print(f"  [{prefix}] {issue.code}: {issue.message}")
            if issue.hint:
                print(f"         hint: {issue.hint}")
        if self.n_participants is not None and self.n_trials is not None:
            print(
                f"  panel shape: participants={self.n_participants}, "
                f"trials_per_participant={self.n_trials}"
            )

    def raise_for_errors(self) -> None:
        """Raise ValueError if any error-level issues were recorded."""
        errors = [issue for issue in self.issues if issue.level == "error"]
        if not errors:
            return
        lines = [f"{issue.code}: {issue.message}" for issue in errors]
        if errors[0].hint:
            lines.append(f"hint: {errors[0].hint}")
        raise ValueError("\n".join(lines))


def _required_columns(config: ModelConfig) -> list[str]:
    """Return participant, response, and configured context columns."""
    columns = [PARTICIPANT_COL, *config.response]
    context_fields = config.context_fields or []
    for name in context_fields:
        if name not in columns:
            columns.append(name)
    return columns


def _add_issue(
    report: DataValidationReport,
    *,
    level: Literal["error", "warning"],
    code: str,
    message: str,
    hint: str | None = None,
) -> None:
    """Append a validation issue to ``report``."""
    report.issues.append(
        DataValidationIssue(level=level, code=code, message=message, hint=hint)
    )


def _check_input_type(data: object, report: DataValidationReport) -> bool:
    """Validate that ``data`` is a non-empty DataFrame."""
    if not isinstance(data, pd.DataFrame):
        _add_issue(
            report,
            level="error",
            code="invalid_type",
            message="data must be a pandas DataFrame.",
        )
        return False
    if data.empty:
        _add_issue(
            report,
            level="error",
            code="empty_data",
            message="data must contain at least one trial.",
        )
        return False
    return True


def _check_required_columns(
    config: ModelConfig, data: pd.DataFrame, report: DataValidationReport
) -> None:
    """Report missing required columns, with hints for common outcome names."""
    required = _required_columns(config)
    missing = [col for col in required if col not in data.columns]
    if not missing:
        return

    for col in missing:
        hint = None
        if col in (config.context_fields or []):
            candidates = [c for c in _OUTCOME_NAME_HINTS if c in data.columns]
            if candidates:
                hint = (
                    f"Rename column {candidates[0]!r} to {col!r}, or include "
                    f"{candidates[0]!r} in ModelConfig(context_fields=...)."
                )
        _add_issue(
            report,
            level="error",
            code="missing_column",
            message=f"Required column {col!r} is missing.",
            hint=hint or f"Expected columns include: {required}.",
        )


def _check_extra_columns(
    config: ModelConfig, data: pd.DataFrame, report: DataValidationReport
) -> None:
    """Warn when the panel includes columns not required by the model."""
    required = set(_required_columns(config))
    extra = sorted(set(data.columns) - required)
    if extra:
        _add_issue(
            report,
            level="warning",
            code="extra_columns",
            message=f"data has extra columns not required by this model: {extra}.",
        )


def _check_null_participants(data: pd.DataFrame, report: DataValidationReport) -> None:
    """Report null values in the participant identifier column."""
    if PARTICIPANT_COL not in data.columns:
        return
    n_null = int(data[PARTICIPANT_COL].isna().sum())
    if n_null > 0:
        _add_issue(
            report,
            level="error",
            code="null_participant_id",
            message=(
                f"Column {PARTICIPANT_COL!r} contains {n_null} missing value(s). "
                "All rows must have a valid participant identifier."
            ),
        )


def _check_row_contiguity(data: pd.DataFrame, report: DataValidationReport) -> None:
    """Require participant blocks to be contiguous in row order."""
    if PARTICIPANT_COL not in data.columns:
        return

    n_runs = int((data[PARTICIPANT_COL] != data[PARTICIPANT_COL].shift()).sum())
    n_participants = data[PARTICIPANT_COL].nunique(dropna=True)
    if n_runs != n_participants:
        _add_issue(
            report,
            level="error",
            code="interleaved_participants",
            message=(
                "data rows must be contiguous per participant. "
                "Interleaved participant blocks corrupt per-participant trial "
                "sequences for RLSSM inference and PPC."
            ),
            hint=(
                'Sort the data with data.sort_values(["participant_id", "trial_id"]) '
                "before passing it to RLSSM."
            ),
        )


def _check_balanced_panel(
    data: pd.DataFrame, report: DataValidationReport
) -> tuple[int | None, int | None]:
    """Validate balanced panels and return participant/trial counts when valid."""
    if PARTICIPANT_COL not in data.columns:
        return None, None

    counts = data.groupby(PARTICIPANT_COL, sort=True).size()
    if counts.empty:
        return None, None

    if counts.nunique() != 1:
        _add_issue(
            report,
            level="error",
            code="unbalanced_panel",
            message=(
                "data must form a balanced panel: all participants must have the "
                f"same number of trials. Observed trial counts: {dict(counts)}."
            ),
            hint="Filter or pad participants so every participant has the same trial count.",
        )
        return int(len(counts)), None

    return int(len(counts)), int(counts.iloc[0])


def _check_nan_values(
    data: pd.DataFrame, required_columns: list[str], report: DataValidationReport
) -> None:
    """Report NaN values in required non-omission columns."""
    present = [col for col in required_columns if col in data.columns]
    for col in present:
        n_nan = int(data[col].isna().sum())
        if n_nan > 0:
            _add_issue(
                report,
                level="error",
                code="missing_values",
                message=f"Column {col!r} contains {n_nan} missing value(s).",
                hint="Drop or impute missing values before validation.",
            )


def _check_omissions(data: pd.DataFrame, report: DataValidationReport) -> None:
    if RT_COL in data.columns:
        rt_omissions = data[RT_COL] == OMISSION_SENTINEL
        if rt_omissions.any():
            n = int(rt_omissions.sum())
            _add_issue(
                report,
                level="error",
                code="omission_rt",
                message=(
                    f"data contains {n} trial(s) with rt={OMISSION_SENTINEL} "
                    "(omission sentinel)."
                ),
                hint=(
                    f"Filter omissions with data[data['rt'] != {OMISSION_SENTINEL}] "
                    "before passing data to RLSSM."
                ),
            )

    if RESPONSE_COL in data.columns:
        response_omissions = data[RESPONSE_COL] == MISSING_RESPONSE_SENTINEL
        if response_omissions.any():
            n = int(response_omissions.sum())
            _add_issue(
                report,
                level="error",
                code="omission_response",
                message=(
                    f"data contains {n} trial(s) with response="
                    f"{MISSING_RESPONSE_SENTINEL} (omission sentinel)."
                ),
                hint=(
                    f"Filter omissions with data[data['response'] != "
                    f"{MISSING_RESPONSE_SENTINEL}] before passing data to RLSSM."
                ),
            )


def _check_response_values(
    config: ModelConfig, data: pd.DataFrame, report: DataValidationReport
) -> None:
    if RESPONSE_COL not in data.columns or RT_COL not in data.columns:
        return

    valid_mask = (data[RT_COL] != OMISSION_SENTINEL) & (
        data[RESPONSE_COL] != MISSING_RESPONSE_SENTINEL
    )
    if not valid_mask.any():
        return

    allowed = set(config.choices)
    mapping_keys = set(config.resolved_response_to_choice.keys())
    subset = data.loc[valid_mask, RESPONSE_COL]
    numeric_subset = pd.to_numeric(subset, errors="coerce")
    if numeric_subset.isna().any():
        _add_issue(
            report,
            level="error",
            code="invalid_response_dtype",
            message="response contains non-numeric values on non-omission trials.",
        )
        return

    unique_vals = {int(v) for v in numeric_subset.unique()}
    invalid_choices = sorted(v for v in unique_vals if v not in allowed)
    if invalid_choices:
        _add_issue(
            report,
            level="error",
            code="invalid_response_labels",
            message=(
                f"response contains values outside config.choices={tuple(config.choices)}: "
                f"{invalid_choices}."
            ),
            hint="Use SSM response labels consistent with the task environment.",
        )

    unmapped = sorted(v for v in unique_vals if v not in mapping_keys)
    if unmapped:
        _add_issue(
            report,
            level="error",
            code="unmapped_response_labels",
            message=(
                f"response contains values not covered by response_to_choice: {unmapped}."
            ),
            hint=(
                "Set ModelConfig.response_to_choice explicitly or align "
                "task_environment.response_labels with the data."
            ),
        )


def _check_rt_values(data: pd.DataFrame, report: DataValidationReport) -> None:
    if RT_COL not in data.columns or RESPONSE_COL not in data.columns:
        return

    valid_mask = (data[RT_COL] != OMISSION_SENTINEL) & (
        data[RESPONSE_COL] != MISSING_RESPONSE_SENTINEL
    )
    if not valid_mask.any():
        return

    rt_values = data.loc[valid_mask, RT_COL]
    numeric_rt = pd.to_numeric(rt_values, errors="coerce")
    if numeric_rt.isna().any():
        _add_issue(
            report,
            level="error",
            code="invalid_rt_dtype",
            message="rt contains non-numeric values on non-omission trials.",
        )
        return

    non_finite = ~np.isfinite(numeric_rt.to_numpy())
    if non_finite.any():
        _add_issue(
            report,
            level="error",
            code="invalid_rt",
            message="rt contains non-finite values on non-omission trials.",
        )

    non_positive = numeric_rt <= 0
    if non_positive.any():
        _add_issue(
            report,
            level="error",
            code="non_positive_rt",
            message="rt must be positive on non-omission trials.",
            hint="Remove deadline/omission trials or invalid RT entries.",
        )


def validate_rlssm_data(
    config: ModelConfig, data: pd.DataFrame
) -> DataValidationReport:
    """Validate a data panel against the RLSSM model contract.

    Parameters
    ----------
    config : ModelConfig
        Structural RLSSM configuration. Should already pass ``config.validate()``.
    data : pd.DataFrame
        Empirical or simulated trial-level panel.

    Returns
    -------
    DataValidationReport
        Validation findings. Call ``raise_for_errors()`` to fail fast on errors.
    """
    report = DataValidationReport()
    if not _check_input_type(data, report):
        return report

    required_columns = _required_columns(config)
    _check_required_columns(config, data, report)
    if report.ok:
        _check_extra_columns(config, data, report)
        _check_null_participants(data, report)
        _check_row_contiguity(data, report)
        n_participants, n_trials = _check_balanced_panel(data, report)
        _check_nan_values(data, required_columns, report)
        _check_omissions(data, report)
        _check_response_values(config, data, report)
        _check_rt_values(data, report)

        if report.ok and n_participants is not None and n_trials is not None:
            report.n_participants = n_participants
            report.n_trials = n_trials

    return report
