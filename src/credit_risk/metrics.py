from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _one_dimensional_array(name: str, values: Sequence[object]) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def binary_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float]:
    y_array = _one_dimensional_array("y_true", y_true)
    probability_values = _one_dimensional_array("probabilities", probabilities)

    if len(y_array) != len(probability_values):
        raise ValueError("y_true and probabilities must have the same length")
    if not np.isin(y_array, [0, 1]).all():
        raise ValueError("y_true values must be 0 or 1")
    if not np.any(y_array == 0) or not np.any(y_array == 1):
        raise ValueError("y_true must contain both classes 0 and 1 for ROC AUC and KS")

    try:
        probability_array = probability_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain finite numeric values") from exc
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")
    if not ((probability_array >= 0.0) & (probability_array <= 1.0)).all():
        raise ValueError("probabilities must be between 0 and 1")

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    target = y_array.astype(int)
    predictions = (probability_array >= threshold_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability_array)
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "roc_auc": float(roc_auc_score(target, probability_array)),
        "average_precision": float(average_precision_score(target, probability_array)),
        "brier_score": float(brier_score_loss(target, probability_array)),
        "log_loss": float(log_loss(target, probability_array, labels=[0, 1])),
        "ks": float(np.max(true_positive_rate - false_positive_rate)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "specificity": float(specificity),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
