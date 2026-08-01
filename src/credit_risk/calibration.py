from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

MAX_CALIBRATION_FOLDS = 5
MIN_ISOTONIC_SAMPLES = 1000
MIN_ISOTONIC_CLASS_COUNT = 50
CALIBRATION_CURVE_COLUMNS = [
    "method",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "sample_count",
    "mean_probability",
    "observed_default_rate",
]


@dataclass(frozen=True)
class CalibrationSelection:
    method: str
    scores: dict[str, float]


@dataclass(frozen=True)
class CalibrationEvaluation:
    selection: CalibrationSelection
    probabilities: dict[str, np.ndarray]
    metrics: dict[str, dict[str, object]]
    curve: pd.DataFrame
    folds: int
    evaluation_protocol: str


def _validated_target(y_true: object) -> np.ndarray:
    try:
        target = np.asarray(y_true)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be a one-dimensional sequence") from exc
    if target.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if target.size == 0:
        raise ValueError("y_true must be non-empty")
    try:
        contains_only_binary_values = bool(np.isin(target, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true values must be 0 or 1") from exc
    if not contains_only_binary_values:
        raise ValueError("y_true values must be 0 or 1")
    return target.astype(int, copy=False)


def _validated_probabilities(name: str, probabilities: object) -> np.ndarray:
    try:
        probability_values = np.asarray(probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if probability_values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if probability_values.size == 0:
        raise ValueError(f"{name} must be non-empty")
    try:
        probability_array = probability_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(probability_array).all():
        raise ValueError(f"{name} must contain only finite values")
    if not ((probability_array >= 0.0) & (probability_array <= 1.0)).all():
        raise ValueError(f"{name} must be between 0 and 1")
    return probability_array


def _validated_inputs(
    y_true: object,
    probabilities: object,
    *,
    probability_name: str = "probabilities",
) -> tuple[np.ndarray, np.ndarray]:
    target = _validated_target(y_true)
    probability_array = _validated_probabilities(probability_name, probabilities)
    if len(target) != len(probability_array):
        raise ValueError(f"y_true and {probability_name} must have the same length")
    return target, probability_array


def expected_calibration_error(
    y_true: object,
    probabilities: object,
    bins: int = 10,
) -> float:
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive int and must not be a bool")
    target, probability_array = _validated_inputs(y_true, probabilities)
    bin_indices, _ = _calibration_bins(probability_array, bins)
    error = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        if members.any():
            error += float(members.mean()) * abs(
                float(probability_array[members].mean()) - float(target[members].mean())
            )
    return error


def _calibration_bins(probabilities: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.searchsorted(edges, probabilities, side="right") - 1
    return np.clip(bin_indices, 0, bins - 1), edges


def select_calibration(
    y_true: object,
    candidates: dict[str, object],
) -> CalibrationSelection:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    target = _validated_target(y_true)
    scores: dict[str, float] = {}
    for method, probabilities in candidates.items():
        _, probability_array = _validated_inputs(
            target,
            probabilities,
            probability_name=f"{method} probabilities",
        )
        scores[method] = float(brier_score_loss(target, probability_array))
    return CalibrationSelection(method=min(scores, key=scores.__getitem__), scores=scores)


def calibration_curve_frame(
    y_true: object,
    candidates: dict[str, object],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, probabilities in candidates.items():
        target, probability_array = _validated_inputs(
            y_true,
            probabilities,
            probability_name=f"{method} probabilities",
        )
        if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
            raise ValueError("bins must be a positive int and must not be a bool")
        bin_indices, edges = _calibration_bins(probability_array, bins)
        for bin_index in range(bins):
            members = bin_indices == bin_index
            sample_count = int(members.sum())
            rows.append(
                {
                    "method": method,
                    "bin_index": bin_index,
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                    "sample_count": sample_count,
                    "mean_probability": (
                        float(probability_array[members].mean()) if sample_count else np.nan
                    ),
                    "observed_default_rate": (
                        float(target[members].mean()) if sample_count else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(rows, columns=CALIBRATION_CURVE_COLUMNS)


def _feature_row_count(features: object) -> int:
    shape = getattr(features, "shape", None)
    if shape is None:
        raise ValueError("calibration features must expose a row count")
    try:
        return int(shape[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("calibration features must expose a row count") from exc


def _take_rows(features: object, indices: np.ndarray) -> object:
    if isinstance(features, pd.DataFrame):
        return features.iloc[indices]
    return features[indices]  # type: ignore[index]


def _fit_frozen_calibrator(
    base_model: object,
    features: object,
    target: np.ndarray,
    *,
    method: str,
) -> object:
    indices = np.arange(len(target))
    return CalibratedClassifierCV(
        FrozenEstimator(base_model),
        method=method,
        cv=[(indices, indices)],
    ).fit(features, target)


def _skip_reason(method: str, *, samples: int, minority_class_count: int) -> str | None:
    if method == "uncalibrated":
        return None
    if minority_class_count < 2:
        return "requires at least 2 samples in each class for stratified OOF calibration"
    if method == "isotonic" and samples < MIN_ISOTONIC_SAMPLES:
        return f"requires at least {MIN_ISOTONIC_SAMPLES} calibration samples"
    if method == "isotonic" and minority_class_count < MIN_ISOTONIC_CLASS_COUNT:
        return f"requires at least {MIN_ISOTONIC_CLASS_COUNT} samples in each class"
    return None


def _evaluated_method_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    probability_source: str,
    bins: int,
) -> dict[str, object]:
    return {
        "status": "evaluated",
        "probability_source": probability_source,
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            target,
            probabilities,
            bins=bins,
        ),
    }


def evaluate_calibration(
    base_model: object,
    features: object,
    y_true: object,
    *,
    methods: list[str],
    random_seed: int,
    bins: int = 10,
) -> CalibrationEvaluation:
    target = _validated_target(y_true)
    if _feature_row_count(features) != len(target):
        raise ValueError("calibration feature and target rows must match")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an int and must not be a bool")

    class_counts = np.bincount(target, minlength=2)
    minority_class_count = int(class_counts.min())
    folds = min(MAX_CALIBRATION_FOLDS, minority_class_count)

    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, object]] = {}
    performed_oof = False
    for method in methods:
        reason = _skip_reason(
            method,
            samples=len(target),
            minority_class_count=minority_class_count,
        )
        if reason is not None:
            metrics[method] = {"status": "skipped", "skip_reason": reason}
            continue

        if method == "uncalibrated":
            method_probabilities = base_model.predict_proba(features)[:, 1]  # type: ignore[attr-defined]
            probability_source = "base_model_calibration_partition"
        else:
            performed_oof = True
            method_probabilities = np.full(len(target), np.nan)
            assignment_counts = np.zeros(len(target), dtype=int)
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_seed,
            )
            for fit_indices, holdout_indices in splitter.split(np.zeros(len(target)), target):
                calibrator = _fit_frozen_calibrator(
                    base_model,
                    _take_rows(features, fit_indices),
                    target[fit_indices],
                    method=method,
                )
                fold_probabilities = calibrator.predict_proba(  # type: ignore[attr-defined]
                    _take_rows(features, holdout_indices)
                )[:, 1]
                method_probabilities[holdout_indices] = fold_probabilities
                assignment_counts[holdout_indices] += 1
            if not np.all(assignment_counts == 1):
                raise RuntimeError("each calibration row must receive exactly one OOF prediction")
            probability_source = "stratified_oof"

        _, validated_probabilities = _validated_inputs(
            target,
            method_probabilities,
            probability_name=f"{method} probabilities",
        )
        probabilities[method] = validated_probabilities
        metrics[method] = _evaluated_method_metrics(
            target,
            validated_probabilities,
            probability_source=probability_source,
            bins=bins,
        )

    selection = select_calibration(target, probabilities)
    curve = calibration_curve_frame(target, probabilities, bins=bins)
    evaluation_protocol = "stratified_oof" if performed_oof else "base_model_holdout_only"
    return CalibrationEvaluation(
        selection=selection,
        probabilities=probabilities,
        metrics=metrics,
        curve=curve,
        folds=folds if performed_oof else 0,
        evaluation_protocol=evaluation_protocol,
    )


def fit_calibrated_model(
    base_model: object,
    features: object,
    y_true: object,
    *,
    method: str,
) -> object:
    if method == "uncalibrated":
        return base_model
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"unknown calibration method: {method}")
    target = _validated_target(y_true)
    if _feature_row_count(features) != len(target):
        raise ValueError("calibration feature and target rows must match")
    return _fit_frozen_calibrator(base_model, features, target, method=method)
