"""Trusted loading and scoring for the frozen credit-risk demonstration release."""

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from credit_risk.artifacts import (
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    validate_release_bundle,
)
from credit_risk.costs import assign_actions
from credit_risk.features import feature_columns
from credit_risk.schemas import CreditApplication, CreditPrediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "artifacts/release"
FEATURE_DICTIONARY_PATH = PROJECT_ROOT / "configs/features.yaml"
ALLOWED_POLICY_PROVENANCE = {
    "selected_calibration_method": {"uncalibrated", "sigmoid", "isotonic"},
    "probability_source": {"base_model_calibration_partition", "stratified_oof"},
    "selection_partition": {"calibration"},
    "threshold_selection_protocol": {"grid_search_on_calibration_evaluation_probabilities"},
    "calibration_evaluation_protocol": {"stratified_oof", "base_model_holdout_only"},
}
FAIRNESS_FILES = {
    "income": "fairness_income.csv",
    "home_ownership": "fairness_home_ownership.csv",
    "region": "fairness_region.csv",
    "employment": "fairness_employment.csv",
}
TEST_SCORING_PROBABILITY_SOURCE = "frozen_calibrated_model"
SYNTHETIC_APPLICATION_VALUES: dict[str, object] = {
    "loan_amnt": 25_000.0,
    "annual_inc": 85_000.0,
    "dti": 28.0,
    "delinq_2yrs": 1.0,
    "fico_range_low": 680.0,
    "fico_range_high": 720.0,
    "inq_last_6mths": 2.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_bal": 6_200.0,
    "revol_util": 32.0,
    "total_acc": 18.0,
    "purpose": "debt_consolidation",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "emp_length": "5 years",
    "addr_state": "TX",
}


class StartupError(RuntimeError):
    """Raised when the frozen release cannot be trusted or loaded."""


class PredictionError(RuntimeError):
    """Raised when a validated application cannot be scored safely."""


@dataclass(frozen=True)
class Policy:
    approve_below: float
    decline_at: float
    lgd: float
    margin: float
    review_cost: float
    currency: str
    selected_calibration_method: str
    probability_source: str
    selection_partition: str
    threshold_selection_protocol: str
    calibration_evaluation_protocol: str


@dataclass(frozen=True)
class ReleaseArtifacts:
    release_dir: Path
    manifest: ReleaseManifest
    preprocessor: object
    model: object
    policy: Policy
    validation_metrics: dict[str, Any]
    calibration_metrics: dict[str, Any]
    calibration_curve: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    final_test_metrics: dict[str, Any]
    confusion_matrix: pd.DataFrame
    policy_test_results: dict[str, Any]
    temporal_metrics: pd.DataFrame
    fairness_tables: dict[str, pd.DataFrame]
    fairness_summary: dict[str, Any]
    shap_importance: pd.DataFrame
    shap_explanations: dict[str, Any]


@dataclass(frozen=True)
class ExplanationEvidence:
    rows: list[tuple[str, float]]
    source: str
    context: str
    value_column: str
    number_format: str


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _finite_policy_number(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"policy field {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy field {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"policy field {field} is outside its allowed range")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"policy field {field} is outside its allowed range")
    return parsed


def _policy_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy field {field} must be a non-empty string")
    return value.strip()


def _parse_policy(payload: dict[str, Any]) -> Policy:
    approve_below = _finite_policy_number(payload, "approve_below", minimum=0.0, maximum=1.0)
    decline_at = _finite_policy_number(payload, "decline_at", minimum=0.0, maximum=1.0)
    if approve_below >= decline_at:
        raise ValueError("policy thresholds must satisfy approve_below < decline_at")
    currency = _policy_string(payload, "currency")
    if currency != "USD":
        raise ValueError("policy currency must be USD")
    provenance = {field: _policy_string(payload, field) for field in ALLOWED_POLICY_PROVENANCE}
    for field, allowed_values in ALLOWED_POLICY_PROVENANCE.items():
        if provenance[field] not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ValueError(f"policy field {field} must be one of: {allowed}")
    method = provenance["selected_calibration_method"]
    probability_source = provenance["probability_source"]
    calibration_protocol = provenance["calibration_evaluation_protocol"]
    if method in {"sigmoid", "isotonic"} and (
        probability_source != "stratified_oof" or calibration_protocol != "stratified_oof"
    ):
        raise ValueError(
            "policy calibrated methods require stratified_oof probability source and protocol"
        )
    if method == "uncalibrated" and probability_source != "base_model_calibration_partition":
        raise ValueError(
            "policy uncalibrated method requires base_model_calibration_partition probabilities"
        )
    return Policy(
        approve_below=approve_below,
        decline_at=decline_at,
        lgd=_finite_policy_number(payload, "lgd", minimum=0.0, maximum=1.0),
        margin=_finite_policy_number(payload, "margin", minimum=0.0, maximum=1.0),
        review_cost=_finite_policy_number(payload, "review_cost", minimum=0.0),
        currency=currency,
        selected_calibration_method=provenance["selected_calibration_method"],
        probability_source=provenance["probability_source"],
        selection_partition=provenance["selection_partition"],
        threshold_selection_protocol=provenance["threshold_selection_protocol"],
        calibration_evaluation_protocol=provenance["calibration_evaluation_protocol"],
    )


def _validated_model_classes(model: object) -> tuple[np.ndarray, int]:
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("calibrated model must provide callable predict_proba")
    if not hasattr(model, "classes_"):
        raise ValueError("calibrated model must provide classes_")
    try:
        classes = np.asarray(model.classes_)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model classes_ must contain labels 0 and 1") from exc
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("calibrated model classes_ must contain two binary labels")
    if np.issubdtype(classes.dtype, np.bool_) or (
        classes.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in classes)
    ):
        raise ValueError("calibrated model classes_ must contain labels 0 and 1")
    try:
        has_missing = bool(pd.isna(classes).any())
        is_binary = bool(np.isin(classes, [0, 1]).all())
        unique_classes = np.unique(classes)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model classes_ must contain labels 0 and 1") from exc
    if has_missing or not is_binary or len(unique_classes) != 2:
        raise ValueError("calibrated model classes_ must contain unique labels 0 and 1")
    bad_indices = np.flatnonzero(classes == 1)
    if len(bad_indices) != 1:
        raise ValueError("calibrated model classes_ has an ambiguous bad class")
    return classes, int(bad_indices[0])


def _required_mapping(payload: dict[str, Any], field: str, artifact: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} field {field} must be an object")
    return value


def _assert_artifact_value(
    payload: dict[str, Any],
    field: str,
    expected: object,
    *,
    artifact: str,
) -> None:
    if field not in payload:
        raise ValueError(f"{artifact} missing required field {field}")
    actual = payload[field]
    if isinstance(expected, float):
        if isinstance(actual, bool):
            matches = False
        else:
            try:
                parsed = float(actual)
            except (TypeError, ValueError):
                matches = False
            else:
                matches = math.isfinite(parsed) and math.isclose(
                    parsed, expected, rel_tol=1e-12, abs_tol=1e-12
                )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"{artifact} field {field} contradicts the frozen release: "
            f"expected {expected!r}, found {actual!r}"
        )


def _validate_release_consistency(
    manifest: ReleaseManifest,
    policy: Policy,
    validation_metrics: dict[str, Any],
    calibration_metrics: dict[str, Any],
    final_test_metrics: dict[str, Any],
    policy_test_results: dict[str, Any],
) -> None:
    _assert_artifact_value(
        validation_metrics,
        "primary_feature_set",
        manifest.feature_set,
        artifact="validation_metrics",
    )

    calibration_expectations = {
        "selected_method": policy.selected_calibration_method,
        "evaluation_protocol": policy.calibration_evaluation_protocol,
        "evaluation_partition": policy.selection_partition,
    }
    for field, expected in calibration_expectations.items():
        _assert_artifact_value(
            calibration_metrics,
            field,
            expected,
            artifact="calibration_metrics",
        )
    calibration_artifact = _required_mapping(calibration_metrics, "artifact", "calibration_metrics")
    _assert_artifact_value(
        calibration_artifact,
        "method",
        policy.selected_calibration_method,
        artifact="calibration_metrics.artifact",
    )
    methods = _required_mapping(calibration_metrics, "methods", "calibration_metrics")
    selected_metrics = _required_mapping(
        methods, policy.selected_calibration_method, "calibration_metrics.methods"
    )
    _assert_artifact_value(
        selected_metrics,
        "probability_source",
        policy.probability_source,
        artifact=f"calibration_metrics.methods.{policy.selected_calibration_method}",
    )

    policy_expectations = {
        "approve_below": policy.approve_below,
        "decline_at": policy.decline_at,
        "lgd": policy.lgd,
        "margin": policy.margin,
        "review_cost": policy.review_cost,
        "currency": policy.currency,
        "selected_calibration_method": policy.selected_calibration_method,
        "threshold_selection_probability_source": policy.probability_source,
        "selection_partition": policy.selection_partition,
        "threshold_selection_protocol": policy.threshold_selection_protocol,
        "calibration_evaluation_protocol": policy.calibration_evaluation_protocol,
        "test_scoring_probability_source": TEST_SCORING_PROBABILITY_SOURCE,
    }
    for field, expected in policy_expectations.items():
        _assert_artifact_value(
            policy_test_results,
            field,
            expected,
            artifact="policy_test_results",
        )

    model_provenance = _required_mapping(
        final_test_metrics, "model_provenance", "final_test_metrics"
    )
    model_expectations = {
        "feature_set": manifest.feature_set,
        "evaluation_partition": "test",
        "preprocessor_artifact": manifest.preprocessor_file,
        "model_artifact": manifest.model_file,
        "test_scoring_probability_source": TEST_SCORING_PROBABILITY_SOURCE,
    }
    for field, expected in model_expectations.items():
        _assert_artifact_value(
            model_provenance,
            field,
            expected,
            artifact="final_test_metrics.model_provenance",
        )

    policy_provenance = _required_mapping(
        final_test_metrics, "policy_provenance", "final_test_metrics"
    )
    final_policy_expectations = {
        "policy_artifact": manifest.policy_file,
        "approve_below": policy.approve_below,
        "decline_at": policy.decline_at,
        "selected_calibration_method": policy.selected_calibration_method,
        "threshold_selection_probability_source": policy.probability_source,
        "calibration_evaluation_protocol": policy.calibration_evaluation_protocol,
        "selection_partition": policy.selection_partition,
        "threshold_selection_protocol": policy.threshold_selection_protocol,
    }
    for field, expected in final_policy_expectations.items():
        _assert_artifact_value(
            policy_provenance,
            field,
            expected,
            artifact="final_test_metrics.policy_provenance",
        )


def _shap_number(
    payload: dict[str, Any],
    field: str,
    *,
    artifact: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{artifact} field {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact} field {field} must be numeric") from exc
    if (
        not math.isfinite(parsed)
        or (minimum is not None and parsed < minimum)
        or (maximum is not None and parsed > maximum)
    ):
        raise ValueError(f"{artifact} field {field} is outside its allowed range")
    return parsed


def _validate_shap_explanations(
    payload: dict[str, Any],
    policy_test_results: dict[str, Any],
) -> None:
    _assert_artifact_value(
        payload,
        "schema_version",
        "1.0",
        artifact="shap_explanations",
    )
    explanation_model = _required_mapping(
        payload,
        "explanation_model",
        "shap_explanations",
    )
    model_expectations = {
        "artifact": "uncalibrated_model.joblib",
        "source": "frozen_uncalibrated_lightgbm",
        "objective": "binary",
        "sigmoid": 1.0,
        "output_space": "raw_model_output",
        "units": "log_odds",
        "calibrated_probability_source": "frozen_calibrated_model",
        "calibration_note": (
            "SHAP values explain the frozen base LightGBM score, not the post-calibration "
            "probability."
        ),
    }
    for field, expected in model_expectations.items():
        _assert_artifact_value(
            explanation_model,
            field,
            expected,
            artifact="shap_explanations.explanation_model",
        )

    local_explanations = _required_mapping(
        payload,
        "local_explanations",
        "shap_explanations",
    )
    required_actions = {"approve", "manual_review", "decline"}
    if set(local_explanations) != required_actions:
        raise ValueError(
            "shap_explanations local_explanations must contain exactly "
            "approve, manual_review, and decline"
        )
    files = _required_mapping(payload, "files", "shap_explanations")
    waterfalls = _required_mapping(files, "waterfalls", "shap_explanations.files")
    action_rate_fields = {
        "approve": "test_approval_rate",
        "manual_review": "test_review_rate",
        "decline": "test_decline_rate",
    }
    action_rates = {
        action: _shap_number(
            policy_test_results,
            field,
            artifact="policy_test_results",
            minimum=0.0,
            maximum=1.0,
        )
        for action, field in action_rate_fields.items()
    }
    required_local_fields = {
        "policy_action",
        "scored_index",
        "row_identifier",
        "calibrated_probability",
        "base_value",
        "base_model_raw_output",
        "top_contributions",
        "waterfall",
    }
    for action in sorted(required_actions):
        example = local_explanations[action]
        artifact = f"shap_explanations.local_explanations.{action}"
        if example is None:
            rate_field = action_rate_fields[action]
            if action_rates[action] > 0.0:
                raise ValueError(
                    f"{artifact} is unavailable but policy_test_results field "
                    f"{rate_field} is positive"
                )
            continue
        if not isinstance(example, dict):
            raise ValueError(f"{artifact} must be an object")
        missing_fields = sorted(required_local_fields - set(example))
        if missing_fields:
            raise ValueError(f"{artifact} missing required field {missing_fields[0]}")
        _assert_artifact_value(example, "policy_action", action, artifact=artifact)
        _assert_artifact_value(
            example,
            "waterfall",
            f"shap_waterfall_{action}.png",
            artifact=artifact,
        )
        _shap_number(
            example,
            "calibrated_probability",
            artifact=artifact,
            minimum=0.0,
            maximum=1.0,
        )
        _shap_number(example, "base_value", artifact=artifact)
        _shap_number(example, "base_model_raw_output", artifact=artifact)

        contributions = example["top_contributions"]
        if not isinstance(contributions, list) or not contributions:
            raise ValueError(f"{artifact} top_contributions must be a non-empty list")
        feature_names: list[str] = []
        for index, contribution in enumerate(contributions):
            contribution_artifact = f"{artifact}.top_contributions[{index}]"
            if not isinstance(contribution, dict):
                raise ValueError(f"{contribution_artifact} must be an object")
            for field in ("feature", "feature_value", "shap_value"):
                if field not in contribution:
                    raise ValueError(f"{contribution_artifact} missing required field {field}")
            feature = contribution["feature"]
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError(f"{contribution_artifact} field feature must be non-empty")
            feature_names.append(feature.strip())
            _shap_number(contribution, "feature_value", artifact=contribution_artifact)
            _shap_number(contribution, "shap_value", artifact=contribution_artifact)
        if len(feature_names) != len(set(feature_names)):
            raise ValueError(f"{artifact} top_contributions feature names must be unique")

    expected_waterfalls = {
        action: f"shap_waterfall_{action}.png"
        for action, example in local_explanations.items()
        if example is not None
    }
    if waterfalls != expected_waterfalls:
        raise ValueError(
            "shap_explanations files.waterfalls must exactly match available local explanations"
        )


def _load_verified_joblib(
    release_path: Path,
    manifest: ReleaseManifest,
    artifact_name: str,
) -> object:
    manifest_entry = next(
        (item for item in manifest.files if item.path == artifact_name),
        None,
    )
    if manifest_entry is None:
        raise ValueError(f"release manifest does not inventory {artifact_name}")

    artifact_path = release_path / artifact_name
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif artifact_path.is_symlink():
        raise ValueError(f"release artifact must not be a symlink: {artifact_name}")

    descriptor = os.open(artifact_path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = -1
            metadata = os.fstat(input_file.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"release artifact must be a regular file: {artifact_name}")
            if metadata.st_size != manifest_entry.size_bytes:
                raise ValueError(
                    f"release artifact size mismatch for {artifact_name}: "
                    f"expected {manifest_entry.size_bytes}, got {metadata.st_size}"
                )

            digest = hashlib.sha256()
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != manifest_entry.sha256:
                raise ValueError(f"release artifact SHA-256 mismatch for {artifact_name}")

            input_file.seek(0)
            return joblib.load(input_file)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_release_artifacts(release_dir: str | Path = DEFAULT_RELEASE_DIR) -> ReleaseArtifacts:
    release_path = Path(release_dir)
    try:
        manifest = validate_release_bundle(release_path)
        if manifest.feature_set != "challenger":
            raise ValueError("release feature_set must be challenger")

        policy = _parse_policy(_load_json_object(release_path / manifest.policy_file))
        validation_metrics = _load_json_object(release_path / "validation_metrics.json")
        calibration_metrics = _load_json_object(release_path / "calibration_metrics.json")
        final_test_metrics = _load_json_object(release_path / "final_test_metrics.json")
        policy_test_results = _load_json_object(release_path / "policy_test_results.json")
        shap_explanations = _load_json_object(release_path / "shap_explanations.json")
        _validate_shap_explanations(shap_explanations, policy_test_results)
        preprocessor = _load_verified_joblib(
            release_path,
            manifest,
            manifest.preprocessor_file,
        )
        model = _load_verified_joblib(release_path, manifest, manifest.model_file)
        if not callable(getattr(preprocessor, "transform", None)):
            raise ValueError("frozen preprocessor must provide callable transform")
        _validated_model_classes(model)
        _validate_release_consistency(
            manifest,
            policy,
            validation_metrics,
            calibration_metrics,
            final_test_metrics,
            policy_test_results,
        )
        _score_frozen_application(
            SYNTHETIC_APPLICATION_VALUES,
            manifest=manifest,
            preprocessor=preprocessor,
            model=model,
            policy=policy,
        )
        fairness_tables = {
            name: pd.read_csv(release_path / filename) for name, filename in FAIRNESS_FILES.items()
        }
        return ReleaseArtifacts(
            release_dir=release_path,
            manifest=manifest,
            preprocessor=preprocessor,
            model=model,
            policy=policy,
            validation_metrics=validation_metrics,
            calibration_metrics=calibration_metrics,
            calibration_curve=pd.read_csv(release_path / "calibration_curve.csv"),
            cost_sensitivity=pd.read_csv(release_path / "cost_sensitivity.csv"),
            final_test_metrics=final_test_metrics,
            confusion_matrix=pd.read_csv(release_path / "confusion_matrix.csv"),
            policy_test_results=policy_test_results,
            temporal_metrics=pd.read_csv(release_path / "temporal_metrics.csv"),
            fairness_tables=fairness_tables,
            fairness_summary=_load_json_object(release_path / "fairness_summary.json"),
            shap_importance=pd.read_csv(release_path / "shap_importance.csv"),
            shap_explanations=shap_explanations,
        )
    except Exception as exc:
        if isinstance(exc, StartupError):
            raise
        raise StartupError(f"release bundle is unavailable or inconsistent: {exc}") from exc


def _bad_class_probability(model: object, matrix: object) -> float:
    _, bad_index = _validated_model_classes(model)
    predict_proba = model.predict_proba  # type: ignore[attr-defined]
    try:
        probabilities = np.asarray(predict_proba(matrix), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model probabilities must be numeric") from exc
    if probabilities.shape != (1, 2):
        raise ValueError("calibrated model predict_proba must return shape (1, 2)")
    if not np.isfinite(probabilities).all():
        raise ValueError("calibrated model probabilities must be finite")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("calibrated model probabilities must be between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("calibrated model probabilities must sum to 1")
    return float(probabilities[0, bad_index])


def _score_frozen_application(
    values: Mapping[str, object],
    *,
    manifest: ReleaseManifest,
    preprocessor: object,
    model: object,
    policy: Policy,
) -> CreditPrediction:
    application = CreditApplication.model_validate(dict(values))
    numeric_columns, categorical_columns = feature_columns(
        manifest.feature_set,
        path=FEATURE_DICTIONARY_PATH,
    )
    selected_columns = [*numeric_columns, *categorical_columns]
    schema_columns = list(CreditApplication.model_fields)
    if selected_columns != schema_columns:
        raise ValueError("challenger feature order must match the credit application schema")

    fitted_feature_names = getattr(preprocessor, "feature_names_in_", None)
    if fitted_feature_names is not None:
        names = np.asarray(fitted_feature_names, dtype=object)
        if names.ndim != 1 or names.tolist() != selected_columns:
            raise ValueError(
                "preprocessor feature_names_in_ must match the challenger feature order"
            )

    frame = pd.DataFrame([application.model_dump()], columns=selected_columns)
    transform = getattr(preprocessor, "transform", None)
    if not callable(transform):
        raise ValueError("frozen preprocessor must provide transform")
    transformed = transform(frame)
    transformed_shape = getattr(transformed, "shape", None)
    if not isinstance(transformed_shape, tuple) or len(transformed_shape) != 2:
        raise ValueError("preprocessor transform must return a two-dimensional matrix")
    row_count, transformed_width = transformed_shape
    if row_count != 1:
        raise ValueError("preprocessor transform must return one scored row")
    if not isinstance(transformed_width, (int, np.integer)) or transformed_width <= 0:
        raise ValueError("preprocessor transform must return a nonzero feature width")

    model_width = getattr(model, "n_features_in_", None)
    if model_width is not None and (
        isinstance(model_width, (bool, np.bool_))
        or not isinstance(model_width, (int, np.integer))
        or int(model_width) != int(transformed_width)
    ):
        raise ValueError("model n_features_in_ must match the transformed feature width")

    probability = _bad_class_probability(model, transformed)
    actions = assign_actions(
        [probability],
        approve_below=policy.approve_below,
        decline_at=policy.decline_at,
    )
    if len(actions) != 1:
        raise ValueError("policy assignment must return exactly one action")
    return CreditPrediction(
        default_probability=float(probability),
        action=str(actions[0]),
        explanation=[],
    )


def explanation_evidence(
    artifacts: ReleaseArtifacts,
    action: str,
) -> ExplanationEvidence:
    local_explanations = artifacts.shap_explanations.get("local_explanations")
    if isinstance(local_explanations, dict):
        example = local_explanations.get(action)
        if isinstance(example, dict):
            contributions = example.get("top_contributions")
            if isinstance(contributions, list):
                rows: list[tuple[str, float]] = []
                for contribution in contributions:
                    if not isinstance(contribution, dict):
                        continue
                    feature = contribution.get("feature")
                    shap_value = contribution.get("shap_value")
                    if (
                        isinstance(feature, str)
                        and feature.strip()
                        and not isinstance(shap_value, bool)
                    ):
                        try:
                            parsed_value = float(shap_value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(parsed_value):
                            rows.append((feature.strip(), parsed_value))
                if rows:
                    return ExplanationEvidence(
                        rows=rows,
                        source="local_action_example",
                        context=(
                            "Frozen release example for the assigned action; these directional "
                            "SHAP values were not generated for the entered application. Positive "
                            "values increase and negative values decrease the frozen base-model "
                            "log-odds relative to its baseline; this is not calibrated probability."
                        ),
                        value_column="Directional association (SHAP value)",
                        number_format="%+.4f",
                    )

    rows = []
    required_columns = {"feature", "mean_abs_shap"}
    if required_columns.issubset(artifacts.shap_importance.columns):
        for row in artifacts.shap_importance.loc[:, ["feature", "mean_abs_shap"]].itertuples(
            index=False
        ):
            if isinstance(row.feature, str) and row.feature.strip():
                value = float(row.mean_abs_shap)
                if math.isfinite(value) and value >= 0.0:
                    rows.append((row.feature.strip(), value))
    if not rows:
        raise ValueError("release explanation payload contains no usable associations")
    return ExplanationEvidence(
        rows=rows[:5],
        source="global_mean_absolute_importance",
        context=(
            "Global unsigned mean absolute SHAP association/importance; this is not a local "
            "explanation, is not directional, and is not action-specific. The sign is not retained."
        ),
        value_column="Mean absolute association (unsigned)",
        number_format="%.4f",
    )


def predict_application(
    values: Mapping[str, object],
    artifacts: ReleaseArtifacts,
) -> CreditPrediction:
    try:
        prediction = _score_frozen_application(
            values,
            manifest=artifacts.manifest,
            preprocessor=artifacts.preprocessor,
            model=artifacts.model,
            policy=artifacts.policy,
        )
        explanation = explanation_evidence(artifacts, prediction.action)
        return CreditPrediction(
            default_probability=prediction.default_probability,
            action=prediction.action,
            explanation=[(feature, float(value)) for feature, value in explanation.rows],
        )
    except Exception as exc:
        if isinstance(exc, ValidationError):
            raise
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError(f"credit prediction failed: {exc}") from exc


def release_cache_identity(release_dir: str | Path) -> tuple[str, str, str]:
    release_path = Path(release_dir)
    manifest_path = release_path / RELEASE_MANIFEST_FILENAME
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif manifest_path.is_symlink():
        raise StartupError("release manifest must not be a symlink")

    descriptor = -1
    try:
        descriptor = os.open(manifest_path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = -1
            metadata = os.fstat(input_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
                raise ValueError("release manifest must be a non-empty regular file")
            manifest_bytes = input_file.read()
        manifest = ReleaseManifest.model_validate_json(manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        return str(release_path), manifest.version, manifest_digest
    except Exception as exc:
        if isinstance(exc, StartupError):
            raise
        raise StartupError(f"release cache identity is unavailable: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def release_directory() -> Path:
    override = os.environ.get("CREDIT_RISK_RELEASE_DIR")
    if not override:
        return DEFAULT_RELEASE_DIR

    override_path = Path(override)
    if not override_path.is_absolute() or any(part == ".." for part in override_path.parts):
        raise StartupError("release directory override must be an absolute canonical path")
    try:
        canonical_path = override_path.resolve(strict=True)
    except OSError as exc:
        raise StartupError(f"release bundle override is not canonical: {exc}") from exc
    if override_path.absolute() != canonical_path:
        raise StartupError("release directory override must be a canonical path without symlinks")
    return canonical_path
