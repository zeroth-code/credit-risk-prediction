import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss  # noqa: E402

from credit_risk.calibration import expected_calibration_error  # noqa: E402
from credit_risk.config import load_config  # noqa: E402
from credit_risk.costs import assign_actions, policy_cost  # noqa: E402
from credit_risk.explainability import generate_shap_explanations  # noqa: E402
from credit_risk.fairness import build_fairness_diagnostics  # noqa: E402
from credit_risk.features import build_feature_frame, load_feature_dictionary  # noqa: E402
from credit_risk.metrics import binary_metrics, bootstrap_metric  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "configs/base.yaml"
FEATURE_DICTIONARY_PATH = PROJECT_ROOT / "configs/features.yaml"
CLASSIFICATION_THRESHOLD = 0.5
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_INTERVAL_METHOD = "percentile"
BOOTSTRAP_POINT_ESTIMATE_INCLUDED = True
BOOTSTRAP_RESAMPLING = "stratified_with_replacement"
ECE_BINS = 10
ECE_BINNING = "equal_width"
ECE_FINAL_BIN_INCLUSIVE = True
TEST_SCORING_PROBABILITY_SOURCE = "frozen_calibrated_model"
POLICY_FIELDS = (
    "approve_below",
    "decline_at",
    "lgd",
    "margin",
    "review_cost",
    "currency",
    "selected_calibration_method",
    "probability_source",
    "selection_partition",
    "threshold_selection_protocol",
    "calibration_evaluation_protocol",
)
ALLOWED_POLICY_PROVENANCE = {
    "selected_calibration_method": {"uncalibrated", "sigmoid", "isotonic"},
    "probability_source": {"base_model_calibration_partition", "stratified_oof"},
    "selection_partition": {"calibration"},
    "threshold_selection_protocol": {"grid_search_on_calibration_evaluation_probabilities"},
    "calibration_evaluation_protocol": {"stratified_oof", "base_model_holdout_only"},
}
TEMPORAL_COLUMNS = [
    "month",
    "status",
    "count",
    "prevalence",
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "approval_rate",
    "review_rate",
    "decline_rate",
    "policy_cost",
    "policy_cost_per_1000_applications",
    "total_exposure",
    "currency",
]
FAIRNESS_GROUPING_COLUMNS = ["annual_inc", "home_ownership", "addr_state", "emp_length"]
FAIRNESS_OUTPUT_FILES = {
    "income": "fairness_income.csv",
    "home_ownership": "fairness_home_ownership.csv",
    "region": "fairness_region.csv",
    "employment": "fairness_employment.csv",
}


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {description} not found: {path}")


def _load_joblib(path: Path, description: str) -> object:
    _require_file(path, description)
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ValueError(f"could not load {description} from {path}: {exc}") from exc


def _finite_float(
    payload: dict[str, object],
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = payload[field]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"policy field {field} must not be boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy field {field} must be a finite numeric value") from exc
    if not np.isfinite(parsed) or parsed < minimum or (maximum is not None and parsed > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"policy field {field} must be finite and in {interval}")
    return parsed


def _nonempty_string(payload: dict[str, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy field {field} must be a non-empty string")
    return value


def _policy_provenance(payload: dict[str, object]) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for field, allowed_values in ALLOWED_POLICY_PROVENANCE.items():
        value = _nonempty_string(payload, field)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ValueError(f"policy field {field} must be one of: {allowed}")
        provenance[field] = value
    method = provenance["selected_calibration_method"]
    probability_source = provenance["probability_source"]
    calibration_protocol = provenance["calibration_evaluation_protocol"]
    if method in {"sigmoid", "isotonic"} and (
        probability_source != "stratified_oof" or calibration_protocol != "stratified_oof"
    ):
        raise ValueError(
            "policy selected_calibration_method sigmoid/isotonic requires "
            "probability_source=stratified_oof and "
            "calibration_evaluation_protocol=stratified_oof"
        )
    if method == "uncalibrated" and probability_source != "base_model_calibration_partition":
        raise ValueError(
            "policy selected_calibration_method uncalibrated requires "
            "probability_source=base_model_calibration_partition; "
            "calibration_evaluation_protocol may be stratified_oof or base_model_holdout_only"
        )
    return provenance


def _load_policy(path: Path) -> dict[str, object]:
    _require_file(path, "frozen policy artifact")
    try:
        with path.open(encoding="utf-8") as policy_file:
            payload = json.load(policy_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load frozen policy from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen policy must be a JSON object")
    missing = [field for field in POLICY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"frozen policy missing required fields: {', '.join(missing)}")

    approve_below = _finite_float(payload, "approve_below", minimum=0.0, maximum=1.0)
    decline_at = _finite_float(payload, "decline_at", minimum=0.0, maximum=1.0)
    if approve_below >= decline_at:
        raise ValueError("policy thresholds must satisfy approve_below < decline_at")
    lgd = _finite_float(payload, "lgd", minimum=0.0, maximum=1.0)
    margin = _finite_float(payload, "margin", minimum=0.0, maximum=1.0)
    review_cost = _finite_float(payload, "review_cost", minimum=0.0)
    currency = _nonempty_string(payload, "currency")
    if currency != "USD":
        raise ValueError("policy field currency must be USD")
    provenance = _policy_provenance(payload)
    threshold_selection_probability_source = provenance.pop("probability_source")

    return {
        "approve_below": approve_below,
        "decline_at": decline_at,
        "lgd": lgd,
        "margin": margin,
        "review_cost": review_cost,
        "currency": currency,
        "threshold_selection_probability_source": threshold_selection_probability_source,
        **provenance,
    }


def _validated_target(frame: pd.DataFrame) -> np.ndarray:
    values = frame["bad"].to_numpy(copy=True)
    contains_boolean = np.issubdtype(values.dtype, np.bool_) or (
        values.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in values)
    )
    if contains_boolean:
        raise ValueError("test bad target must not contain boolean values")
    try:
        is_binary = bool(np.isin(values, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("test bad target must contain only 0 and 1") from exc
    if not is_binary:
        raise ValueError("test bad target must contain only 0 and 1")
    target = values.astype(int, copy=False)
    if not np.any(target == 0) or not np.any(target == 1):
        raise ValueError("test bad target must contain both classes 0 and 1")
    return target


def _validated_loan_amounts(frame: pd.DataFrame) -> np.ndarray:
    values = frame["loan_amnt"].to_numpy(copy=True)
    contains_boolean = np.issubdtype(values.dtype, np.bool_) or (
        values.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in values)
    )
    if contains_boolean:
        raise ValueError("test loan_amnt must not contain boolean values")
    try:
        amounts = np.asarray(pd.to_numeric(values, errors="raise"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("test loan_amnt must contain numeric values") from exc
    if not np.isfinite(amounts).all():
        raise ValueError("test loan_amnt must contain only finite values")
    if not (amounts >= 0.0).all():
        raise ValueError("test loan_amnt must contain only nonnegative values")
    return amounts


def _validated_months(frame: pd.DataFrame) -> np.ndarray:
    issue_d = frame["issue_d"]
    if issue_d.empty:
        raise ValueError("test issue_d must be non-empty")
    if pd.api.types.is_datetime64_any_dtype(issue_d.dtype):
        issue_dates = issue_d
    else:
        raw_values = issue_d.to_numpy(dtype=object, copy=True)
        if not all(isinstance(value, str) for value in raw_values):
            raise ValueError("test issue_d must use a datetime dtype or ISO-8601 date strings")
        date_strings = issue_d.astype("string").str.strip()
        if date_strings.isna().any() or date_strings.eq("").any():
            raise ValueError("test issue_d date strings must be non-empty")
        try:
            issue_dates = pd.to_datetime(date_strings, format="ISO8601", errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("test issue_d strings must contain valid ISO-8601 dates") from exc
    if issue_dates.isna().any():
        raise ValueError("test issue_d must not contain missing dates")
    if issue_dates.dt.tz is not None:
        issue_dates = issue_dates.dt.tz_convert(None)
    return issue_dates.dt.to_period("M").astype(str).to_numpy()


def _validated_model_classes(model: object) -> tuple[np.ndarray, int]:
    if not hasattr(model, "classes_"):
        raise ValueError("calibrated model artifact must provide classes_")
    try:
        classes = np.asarray(model.classes_)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "calibrated model classes_ must be a one-dimensional binary array"
        ) from exc
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("calibrated model classes_ must be one-dimensional with length 2")
    contains_boolean = np.issubdtype(classes.dtype, np.bool_) or (
        classes.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in classes)
    )
    if contains_boolean:
        raise ValueError("calibrated model classes_ must not contain boolean values")
    try:
        has_missing = bool(pd.isna(classes).any())
        is_binary = bool(np.isin(classes, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model classes_ must contain exactly labels 0 and 1") from exc
    if has_missing or not is_binary or len(np.unique(classes)) != 2:
        raise ValueError("calibrated model classes_ must contain exactly unique labels 0 and 1")
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("calibrated model classes_ must contain exactly one positive label 1")
    return classes, int(positive_indices[0])


def _validated_probabilities(model: object, matrix: object, expected_rows: int) -> np.ndarray:
    classes, positive_class_index = _validated_model_classes(model)
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError("calibrated model artifact must provide predict_proba")
    try:
        probability_matrix = np.asarray(predict_proba(matrix)).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model probability matrix must contain numeric values") from exc
    expected_shape = (expected_rows, len(classes))
    if probability_matrix.ndim != 2 or probability_matrix.shape != expected_shape:
        raise ValueError(
            "calibrated model predict_proba output must have shape "
            f"{expected_shape}, got {probability_matrix.shape}"
        )
    if not np.isfinite(probability_matrix).all():
        raise ValueError("calibrated model probability matrix must contain only finite values")
    if not ((probability_matrix >= 0.0) & (probability_matrix <= 1.0)).all():
        raise ValueError("calibrated model probability matrix values must be between 0 and 1")
    if not np.allclose(probability_matrix.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("calibrated model probability matrix rows must sum to 1")
    return probability_matrix[:, positive_class_index].copy()


def _policy_summary(
    target: np.ndarray,
    loan_amounts: np.ndarray,
    actions: np.ndarray,
    policy: dict[str, object],
) -> dict[str, float]:
    cost = policy_cost(
        target,
        loan_amounts,
        actions,
        lgd=float(policy["lgd"]),
        margin=float(policy["margin"]),
        review_cost=float(policy["review_cost"]),
    )
    return {
        "approval_rate": float(np.mean(actions == "approve")),
        "review_rate": float(np.mean(actions == "manual_review")),
        "decline_rate": float(np.mean(actions == "decline")),
        "policy_cost": cost,
        "policy_cost_per_1000_applications": cost * 1000.0 / len(target),
        "total_exposure": float(np.sum(loan_amounts)),
    }


def _temporal_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    loan_amounts: np.ndarray,
    actions: np.ndarray,
    months: np.ndarray,
    policy: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in sorted(set(months.tolist())):
        members = months == month
        monthly_target = target[members]
        monthly_probabilities = probabilities[members]
        monthly_actions = actions[members]
        monthly_amounts = loan_amounts[members]
        has_both_classes = bool(np.any(monthly_target == 0) and np.any(monthly_target == 1))
        discrimination: dict[str, float | None]
        if has_both_classes:
            monthly_predictive = binary_metrics(
                monthly_target,
                monthly_probabilities,
                threshold=CLASSIFICATION_THRESHOLD,
            )
            discrimination = {
                "roc_auc": monthly_predictive["roc_auc"],
                "average_precision": monthly_predictive["average_precision"],
            }
            status = "ok"
        else:
            discrimination = {"roc_auc": None, "average_precision": None}
            status = "single_class_discrimination_undefined"
        monthly_policy = _policy_summary(
            monthly_target,
            monthly_amounts,
            monthly_actions,
            policy,
        )
        rows.append(
            {
                "month": month,
                "status": status,
                "count": int(len(monthly_target)),
                "prevalence": float(np.mean(monthly_target)),
                **discrimination,
                "brier_score": float(brier_score_loss(monthly_target, monthly_probabilities)),
                "log_loss": float(log_loss(monthly_target, monthly_probabilities, labels=[0, 1])),
                "expected_calibration_error": expected_calibration_error(
                    monthly_target, monthly_probabilities, bins=ECE_BINS
                ),
                "approval_rate": monthly_policy["approval_rate"],
                "review_rate": monthly_policy["review_rate"],
                "decline_rate": monthly_policy["decline_rate"],
                "policy_cost": monthly_policy["policy_cost"],
                "policy_cost_per_1000_applications": monthly_policy[
                    "policy_cost_per_1000_applications"
                ],
                "total_exposure": monthly_policy["total_exposure"],
                "currency": policy["currency"],
            }
        )
    return pd.DataFrame.from_records(rows, columns=TEMPORAL_COLUMNS)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")


def main(
    *,
    config_path: str | Path = BASE_CONFIG_PATH,
    feature_dictionary_path: str | Path = FEATURE_DICTIONARY_PATH,
) -> None:
    resolved_config_path = _project_path(config_path)
    resolved_feature_path = _project_path(feature_dictionary_path)
    config = load_config(resolved_config_path)
    feature_dictionary = load_feature_dictionary(resolved_feature_path)
    challenger = feature_dictionary["challenger"]
    selected_columns = list(challenger["numeric"]) + list(challenger["categorical"])  # type: ignore[index]

    processed_dir = _project_path(config.processed_dir)
    artifact_dir = _project_path(config.artifact_dir)
    figure_dir = _project_path(config.figure_dir)
    test_path = processed_dir / "test.parquet"
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    model_path = artifact_dir / "calibrated_model.joblib"
    explanation_model_path = artifact_dir / "uncalibrated_model.joblib"
    policy_path = artifact_dir / "policy.json"
    for path, description in (
        (test_path, "test partition"),
        (preprocessor_path, "frozen preprocessor artifact"),
        (model_path, "frozen calibrated model artifact"),
        (explanation_model_path, "frozen uncalibrated explanation model artifact"),
        (policy_path, "frozen policy artifact"),
    ):
        _require_file(path, description)

    test_frame = pd.read_parquet(test_path)
    if test_frame.empty:
        raise ValueError("test partition must be non-empty")
    required_columns = list(
        dict.fromkeys(
            [*selected_columns, "bad", "issue_d", "loan_amnt", *FAIRNESS_GROUPING_COLUMNS]
        )
    )
    missing_columns = [column for column in required_columns if column not in test_frame.columns]
    if missing_columns:
        raise ValueError(f"test partition missing required columns: {', '.join(missing_columns)}")
    reserved_columns = [
        column
        for column in ("default_probability", "predicted_bad", "action")
        if column in test_frame.columns
    ]
    if reserved_columns:
        raise ValueError(
            f"test partition contains reserved output columns: {', '.join(reserved_columns)}"
        )

    target = _validated_target(test_frame)
    loan_amounts = _validated_loan_amounts(test_frame)
    months = _validated_months(test_frame)
    policy = _load_policy(policy_path)
    feature_frame = build_feature_frame(
        test_frame,
        selected_columns,
        path=resolved_feature_path,
    )
    preprocessor = _load_joblib(preprocessor_path, "frozen preprocessor artifact")
    transform = getattr(preprocessor, "transform", None)
    if not callable(transform):
        raise ValueError("preprocessor artifact must provide transform")
    transformed = transform(feature_frame)
    transformed_shape = getattr(transformed, "shape", None)
    if transformed_shape is None or transformed_shape[0] != len(test_frame):
        raise ValueError("preprocessor transform output rows must align with the test partition")
    get_feature_names_out = getattr(preprocessor, "get_feature_names_out", None)
    if not callable(get_feature_names_out):
        raise ValueError("preprocessor artifact must provide get_feature_names_out")
    transformed_feature_names = get_feature_names_out()
    model = _load_joblib(model_path, "frozen calibrated model artifact")
    probabilities = _validated_probabilities(model, transformed, len(test_frame))
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    actions = assign_actions(
        probabilities,
        approve_below=float(policy["approve_below"]),
        decline_at=float(policy["decline_at"]),
    )
    if not len(target) == len(probabilities) == len(predictions) == len(actions):
        raise RuntimeError("target, probability, prediction, and action rows are not aligned")

    fairness_tables, fairness_summary = build_fairness_diagnostics(
        test_frame,
        target,
        probabilities,
        actions,
        minimum_group_size=config.minimum_group_size,
    )

    predictive_metrics = binary_metrics(
        target,
        probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
    )
    confidence_intervals = {
        metric_name: bootstrap_metric(
            target,
            probabilities,
            metric_name=metric_name,
            samples=BOOTSTRAP_SAMPLES,
            random_seed=config.random_seed,
        )
        for metric_name in ("roc_auc", "average_precision", "brier_score")
    }
    policy_summary = _policy_summary(target, loan_amounts, actions, policy)
    temporal = _temporal_metrics(
        target,
        probabilities,
        loan_amounts,
        actions,
        months,
        policy,
    )

    final_metrics: dict[str, object] = {
        "test_samples": int(len(target)),
        "prevalence": float(np.mean(target)),
        "predictive_metrics": predictive_metrics,
        "expected_calibration_error": expected_calibration_error(
            target, probabilities, bins=ECE_BINS
        ),
        "confidence_intervals": confidence_intervals,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "bootstrap_methodology": {
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "interval_method": BOOTSTRAP_INTERVAL_METHOD,
            "point_estimate_included": BOOTSTRAP_POINT_ESTIMATE_INCLUDED,
            "resampling": BOOTSTRAP_RESAMPLING,
            "random_seed": config.random_seed,
        },
        "ece_methodology": {
            "bins": ECE_BINS,
            "binning": ECE_BINNING,
            "final_bin_inclusive": ECE_FINAL_BIN_INCLUSIVE,
        },
        "model_provenance": {
            "feature_set": "challenger",
            "evaluation_partition": "test",
            "preprocessor_artifact": preprocessor_path.name,
            "model_artifact": model_path.name,
            "test_scoring_probability_source": TEST_SCORING_PROBABILITY_SOURCE,
        },
        "policy_provenance": {
            "policy_artifact": policy_path.name,
            "approve_below": policy["approve_below"],
            "decline_at": policy["decline_at"],
            "selected_calibration_method": policy["selected_calibration_method"],
            "threshold_selection_probability_source": policy[
                "threshold_selection_probability_source"
            ],
            "calibration_evaluation_protocol": policy["calibration_evaluation_protocol"],
            "selection_partition": policy["selection_partition"],
            "threshold_selection_protocol": policy["threshold_selection_protocol"],
        },
    }
    confusion = confusion_matrix(target, predictions, labels=[0, 1])
    confusion_rows = [
        {
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "count": int(confusion[actual_label, predicted_label]),
        }
        for actual_label in (0, 1)
        for predicted_label in (0, 1)
    ]
    policy_results: dict[str, object] = {
        **policy,
        "test_scoring_probability_source": TEST_SCORING_PROBABILITY_SOURCE,
        "test_samples": int(len(target)),
        "total_exposure": policy_summary["total_exposure"],
        "test_cost": policy_summary["policy_cost"],
        "test_cost_per_1000_applications": policy_summary["policy_cost_per_1000_applications"],
        "test_approval_rate": policy_summary["approval_rate"],
        "test_review_rate": policy_summary["review_rate"],
        "test_decline_rate": policy_summary["decline_rate"],
    }
    scored = test_frame.copy(deep=True)
    scored["default_probability"] = probabilities
    scored["predicted_bad"] = predictions
    scored["action"] = actions
    if len(scored) != len(test_frame):
        raise RuntimeError("scored test rows are not aligned with the input test partition")

    explanation_model = _load_joblib(
        explanation_model_path,
        "frozen uncalibrated explanation model artifact",
    )
    row_identifier_column = "id" if "id" in scored.columns else None
    explanation_columns = ["action", "default_probability"]
    if row_identifier_column is not None:
        explanation_columns.append(row_identifier_column)
    explanation_scored = scored.loc[:, explanation_columns].rename(
        columns={"default_probability": "probability"}
    )
    generate_shap_explanations(
        explanation_model,
        transformed,
        transformed_feature_names,
        explanation_scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        row_identifier_column=row_identifier_column,
        model_artifact_name=explanation_model_path.name,
    )

    _write_json(artifact_dir / "final_test_metrics.json", final_metrics)
    pd.DataFrame.from_records(
        confusion_rows,
        columns=["actual_label", "predicted_label", "count"],
    ).to_csv(artifact_dir / "confusion_matrix.csv", index=False, encoding="utf-8")
    _write_json(artifact_dir / "policy_test_results.json", policy_results)
    temporal.to_csv(artifact_dir / "temporal_metrics.csv", index=False, encoding="utf-8")
    for attribute, output_file in FAIRNESS_OUTPUT_FILES.items():
        fairness_tables[attribute].to_csv(
            artifact_dir / output_file,
            index=False,
            encoding="utf-8",
        )
    _write_json(artifact_dir / "fairness_summary.json", fairness_summary)
    scored.to_parquet(artifact_dir / "scored_test.parquet", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen credit-risk artifacts on the out-of-time test partition"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_CONFIG_PATH,
        help="project config path (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURE_DICTIONARY_PATH,
        help="feature dictionary path (default: configs/features.yaml)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    main(config_path=arguments.config, feature_dictionary_path=arguments.features)
