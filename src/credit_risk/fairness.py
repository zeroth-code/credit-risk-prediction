from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import brier_score_loss, roc_auc_score

FAIRNESS_COLUMNS = [
    "group",
    "count",
    "bad_rate",
    "selection_rate",
    "true_positive_rate",
    "false_positive_rate",
    "roc_auc",
    "brier_score",
    "suppressed",
]

STATE_TO_REGION = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IN": "Midwest",
    "IL": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "DC": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}

_ALLOWED_ACTIONS = {"approve", "manual_review", "decline"}
_ATTRIBUTE_METADATA = {
    "income": {
        "output_file": "fairness_income.csv",
        "group_definition": (
            "five qcut annual_inc quantile bands computed on the frozen test partition with "
            "duplicate edges dropped; missing is Unknown"
        ),
    },
    "home_ownership": {
        "output_file": "fairness_home_ownership.csv",
        "group_definition": "home_ownership stripped strings; missing or blank is Unknown",
    },
    "region": {
        "output_file": "fairness_region.csv",
        "group_definition": (
            "addr_state mapped to US Census regions after stripping and uppercasing; "
            "missing or unrecognized is Unknown"
        ),
    },
    "employment": {
        "output_file": "fairness_employment.csv",
        "group_definition": "emp_length stripped strings; missing or blank is Unknown",
    },
}


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def income_band(values: pd.Series, quantiles: int = 5) -> pd.Series:
    """Assign real, finite income values to deterministic ordered quantile bands."""
    quantile_count = _validate_positive_integer(quantiles, "quantiles")
    if not isinstance(values, pd.Series):
        raise ValueError("income values must be a pandas Series")

    raw_values = values.to_numpy(dtype=object, copy=True)
    numeric_values = np.full(len(values), np.nan, dtype=float)
    for position, value in enumerate(raw_values):
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                "income values must contain only real numeric values or missing values"
            )
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("income values must contain only finite values")
        numeric_values[position] = parsed

    nonmissing_positions = np.flatnonzero(~np.isnan(numeric_values))
    nonmissing = pd.Series(numeric_values[nonmissing_positions], dtype=float)
    assigned = np.full(len(values), "Unknown", dtype=object)
    band_count = 0
    if not nonmissing.empty:
        if nonmissing.nunique() == 1:
            assigned[nonmissing_positions] = "Income Q1"
            band_count = 1
        else:
            raw_codes = pd.qcut(
                nonmissing,
                q=quantile_count,
                labels=False,
                duplicates="drop",
            )
            observed_codes = sorted(int(code) for code in raw_codes.dropna().unique())
            if not observed_codes:
                assigned[nonmissing_positions] = "Income Q1"
                band_count = 1
            else:
                compressed_codes = {
                    code: f"Income Q{position + 1}" for position, code in enumerate(observed_codes)
                }
                assigned[nonmissing_positions] = raw_codes.map(compressed_codes).to_numpy()
                band_count = len(observed_codes)

    categories = [f"Income Q{position}" for position in range(1, band_count + 1)]
    categories.append("Unknown")
    categorical = pd.Categorical(assigned, categories=categories, ordered=True)
    return pd.Series(categorical, index=values.index, name=values.name)


def census_region(states: pd.Series) -> pd.Series:
    """Map normalized US state abbreviations and DC to Census regions."""
    if not isinstance(states, pd.Series):
        raise ValueError("states must be a pandas Series")
    regions: list[str] = []
    for value in states.to_numpy(dtype=object, copy=True):
        if pd.isna(value):
            regions.append("Unknown")
            continue
        if not isinstance(value, str):
            raise ValueError("addr_state values must be strings or missing")
        state = value.strip().upper()
        regions.append(STATE_TO_REGION.get(state, "Unknown"))
    return pd.Series(regions, index=states.index, name=states.name, dtype=object)


def _normalize_text_groups(values: pd.Series, *, column_name: str) -> pd.Series:
    normalized: list[str] = []
    for value in values.to_numpy(dtype=object, copy=True):
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            normalized.append("Unknown")
        elif not isinstance(value, str):
            raise ValueError(f"{column_name} values must be strings or missing")
        else:
            normalized.append(value.strip())
    return pd.Series(normalized, index=values.index, name=values.name, dtype=object)


def suppress_small_groups(metrics: pd.DataFrame, minimum_size: int) -> pd.DataFrame:
    """Null metrics for small groups without mutating group identity or counts."""
    validated_minimum = _validate_positive_integer(minimum_size, "minimum_size")
    missing_columns = [column for column in ("group", "count") if column not in metrics.columns]
    if missing_columns:
        raise ValueError(f"metrics missing required column: {missing_columns[0]}")

    groups = metrics["group"].to_numpy(dtype=object, copy=True)
    if any(pd.isna(group) or not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("group values must be non-empty strings")
    if len(set(groups.tolist())) != len(groups):
        raise ValueError("group values must be unique")

    counts = metrics["count"].to_numpy(dtype=object, copy=True)
    if any(
        isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0
        for count in counts
    ):
        raise ValueError("count values must be nonnegative integers")

    result = metrics.copy(deep=True)
    result["suppressed"] = result["count"] < validated_minimum
    metric_columns = [column for column in result if column not in {"group", "count", "suppressed"}]
    result.loc[result["suppressed"], metric_columns] = np.nan
    return result


def _selection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    del y_true
    return float(np.mean(y_pred))


def _conditional_selection_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    favorable_value: int,
) -> float:
    members = np.asarray(y_true) == favorable_value
    if not np.any(members):
        return float("nan")
    return float(np.mean(np.asarray(y_pred)[members]))


def _true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _conditional_selection_rate(y_true, y_pred, favorable_value=1)


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _conditional_selection_rate(y_true, y_pred, favorable_value=0)


def _bad_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    del y_pred
    return float(np.mean(1 - np.asarray(y_true)))


def _validated_metric_inputs(
    y_bad: object,
    probabilities: object,
    actions: object,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    target = np.asarray(y_bad).copy()
    scores = np.asarray(probabilities).copy()
    decisions = np.asarray(actions, dtype=object).copy()
    if target.ndim != 1 or scores.ndim != 1 or decisions.ndim != 1:
        raise ValueError("target, probabilities, actions, and groups must be one-dimensional")
    if not isinstance(groups, pd.Series):
        raise ValueError("groups must be a pandas Series")
    if not len(target) == len(scores) == len(decisions) == len(groups):
        raise ValueError("target, probabilities, actions, and group rows must align")
    if len(target) == 0:
        raise ValueError("fairness metrics require at least one row")

    contains_target_boolean = np.issubdtype(target.dtype, np.bool_) or (
        target.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in target)
    )
    if contains_target_boolean:
        raise ValueError("bad target must not contain boolean values")
    try:
        valid_target = bool(np.isin(target, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("bad target must contain only 0 and 1") from exc
    if not valid_target:
        raise ValueError("bad target must contain only 0 and 1")
    validated_target = target.astype(int, copy=False)

    contains_probability_boolean = np.issubdtype(scores.dtype, np.bool_) or (
        scores.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in scores)
    )
    if contains_probability_boolean:
        raise ValueError("default probabilities must not contain boolean values")
    try:
        validated_scores = scores.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("default probabilities must be numeric") from exc
    if not np.isfinite(validated_scores).all():
        raise ValueError("default probabilities must be finite")
    if not ((validated_scores >= 0.0) & (validated_scores <= 1.0)).all():
        raise ValueError("default probabilities must be between 0 and 1")

    if any(not isinstance(action, str) or action not in _ALLOWED_ACTIONS for action in decisions):
        raise ValueError("actions must contain only approve, manual_review, or decline")

    normalized_groups: list[str] = []
    for group in groups.to_numpy(dtype=object, copy=True):
        if pd.isna(group) or not isinstance(group, str) or not group.strip():
            raise ValueError("group values must be non-empty strings")
        normalized_groups.append(group.strip())
    validated_groups = pd.Series(normalized_groups, name="group", dtype=object)
    return validated_target, validated_scores, decisions, validated_groups


def subgroup_metrics(
    y_bad: object,
    probabilities: object,
    actions: object,
    groups: pd.Series,
    *,
    minimum_size: int,
) -> pd.DataFrame:
    """Calculate suppression-ready policy and reliability metrics by subgroup."""
    validated_minimum = _validate_positive_integer(minimum_size, "minimum_size")
    target, scores, decisions, validated_groups = _validated_metric_inputs(
        y_bad,
        probabilities,
        actions,
        groups,
    )
    good_outcome = 1 - target
    selected = (decisions == "approve").astype(int)
    metric_frame = MetricFrame(
        metrics={
            "bad_rate": _bad_rate,
            "selection_rate": _selection_rate,
            "true_positive_rate": _true_positive_rate,
            "false_positive_rate": _false_positive_rate,
        },
        y_true=good_outcome,
        y_pred=selected,
        sensitive_features=validated_groups,
    )
    policy_metrics = metric_frame.by_group.reset_index()

    rows: list[dict[str, Any]] = []
    for group in sorted(validated_groups.unique().tolist()):
        members = validated_groups.to_numpy() == group
        group_target = target[members]
        group_probabilities = scores[members]
        policy_row = policy_metrics.loc[policy_metrics["group"] == group].iloc[0]
        roc_auc = (
            float(roc_auc_score(group_target, group_probabilities))
            if len(np.unique(group_target)) == 2
            else float("nan")
        )
        rows.append(
            {
                "group": group,
                "count": int(np.sum(members)),
                "bad_rate": float(policy_row["bad_rate"]),
                "selection_rate": float(policy_row["selection_rate"]),
                "true_positive_rate": float(policy_row["true_positive_rate"]),
                "false_positive_rate": float(policy_row["false_positive_rate"]),
                "roc_auc": roc_auc,
                "brier_score": float(brier_score_loss(group_target, group_probabilities)),
            }
        )
    table = pd.DataFrame.from_records(rows)
    suppressed = suppress_small_groups(table, validated_minimum)
    return suppressed.loc[:, FAIRNESS_COLUMNS]


def _disparity(
    table: pd.DataFrame,
    metric: str,
    *,
    ratio: bool,
) -> dict[str, object]:
    usable = table.loc[~table["suppressed"], metric].dropna()
    if len(usable) < 2:
        return {
            "status": "undefined",
            "value": None,
            "reason": "fewer_than_two_usable_unsuppressed_groups",
        }
    maximum = float(usable.max())
    minimum = float(usable.min())
    if ratio and maximum == 0.0:
        return {
            "status": "undefined",
            "value": None,
            "reason": "maximum_selection_rate_is_zero",
        }
    value = minimum / maximum if ratio else maximum - minimum
    return {"status": "defined", "value": float(value), "reason": None}


def _attribute_summary(name: str, table: pd.DataFrame) -> dict[str, object]:
    suppressed_count = int(table["suppressed"].sum())
    return {
        **_ATTRIBUTE_METADATA[name],
        "total_group_count": int(len(table)),
        "evaluated_group_count": int(len(table) - suppressed_count),
        "suppressed_group_count": suppressed_count,
        "equal_opportunity_difference": _disparity(
            table,
            "true_positive_rate",
            ratio=False,
        ),
        "selection_rate_ratio": _disparity(table, "selection_rate", ratio=True),
    }


def build_fairness_diagnostics(
    frame: pd.DataFrame,
    y_bad: object,
    probabilities: object,
    actions: object,
    *,
    minimum_group_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build all proxy subgroup fairness tables and their audit metadata."""
    validated_minimum = _validate_positive_integer(minimum_group_size, "minimum_group_size")
    required_columns = ["annual_inc", "home_ownership", "addr_state", "emp_length"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"fairness diagnostics missing required columns: {', '.join(missing_columns)}"
        )

    group_values = {
        "income": income_band(frame["annual_inc"]).astype("string"),
        "home_ownership": _normalize_text_groups(
            frame["home_ownership"],
            column_name="home_ownership",
        ),
        "region": census_region(frame["addr_state"]),
        "employment": _normalize_text_groups(
            frame["emp_length"],
            column_name="emp_length",
        ),
    }
    tables = {
        name: subgroup_metrics(
            y_bad,
            probabilities,
            actions,
            groups,
            minimum_size=validated_minimum,
        )
        for name, groups in group_values.items()
    }
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "minimum_group_size": validated_minimum,
        "metric_semantics": {
            "target": "bad=1 (default)",
            "favorable_ground_truth_outcome": "good/repaid (1 - bad)",
            "favorable_decision": "action == approve",
            "not_selected_actions": ["manual_review", "decline"],
            "probability": "frozen calibrated default probability",
            "selection_rate": "overall approval rate",
            "true_positive_rate": "approval rate among actually good/repaid loans",
            "false_positive_rate": "approval rate among actually bad/defaulted loans",
            "roc_auc_and_brier_score_target": "bad=1 (default)",
        },
        "limitations": (
            "These are proxy subgroup reliability diagnostics, not a statutory fair-lending "
            "audit; protected attributes are absent."
        ),
        "attributes": {name: _attribute_summary(name, table) for name, table in tables.items()},
    }
    return tables, summary
