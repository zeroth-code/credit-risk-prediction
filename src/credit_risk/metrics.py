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
    try:
        contains_only_binary_values = bool(np.isin(y_array, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true values must be 0 or 1") from exc
    if not contains_only_binary_values:
        raise ValueError("y_true values must be 0 or 1")
    try:
        has_zero = bool(np.any(y_array == 0))
        has_one = bool(np.any(y_array == 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true values must be 0 or 1") from exc
    if not has_zero or not has_one:
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


def bootstrap_metric(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    metric_name: str,
    samples: int,
    random_seed: int,
) -> dict[str, float]:
    metrics = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "brier_score": brier_score_loss,
    }
    if not isinstance(metric_name, str) or metric_name not in metrics:
        raise ValueError("metric_name must be one of roc_auc, average_precision, or brier_score")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive int and must not be bool")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative int and must not be bool")

    target_values = _one_dimensional_array("y_true", y_true)
    probability_values = _one_dimensional_array("probabilities", probabilities)
    if len(target_values) != len(probability_values):
        raise ValueError("y_true and probabilities must have the same length")

    target_contains_boolean = np.issubdtype(target_values.dtype, np.bool_) or (
        target_values.dtype == object
        and any(isinstance(value, (bool, np.bool_)) for value in target_values)
    )
    if target_contains_boolean:
        raise ValueError("y_true must not contain boolean values")
    try:
        contains_only_binary_values = bool(np.isin(target_values, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true values must be 0 or 1") from exc
    if not contains_only_binary_values:
        raise ValueError("y_true values must be 0 or 1")
    target = target_values.astype(int, copy=False)
    if not np.any(target == 0) or not np.any(target == 1):
        raise ValueError("y_true must contain both classes 0 and 1")

    probabilities_contain_boolean = np.issubdtype(probability_values.dtype, np.bool_) or (
        probability_values.dtype == object
        and any(isinstance(value, (bool, np.bool_)) for value in probability_values)
    )
    if probabilities_contain_boolean:
        raise ValueError("probabilities must not contain boolean values")
    try:
        probability_array = probability_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain only finite numeric values") from exc
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite numeric values")
    if not ((probability_array >= 0.0) & (probability_array <= 1.0)).all():
        raise ValueError("probabilities must be between 0 and 1")

    metric = metrics[metric_name]
    negative_indices = np.flatnonzero(target == 0)
    positive_indices = np.flatnonzero(target == 1)
    rng = np.random.default_rng(random_seed)
    bootstrap_values = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        indices = np.concatenate(
            (
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
            )
        )
        bootstrap_values[sample_index] = metric(target[indices], probability_array[indices])

    estimate = float(metric(target, probability_array))
    percentile_lower = float(np.quantile(bootstrap_values, 0.025))
    percentile_upper = float(np.quantile(bootstrap_values, 0.975))
    return {
        "estimate": estimate,
        "lower": min(percentile_lower, estimate),
        "upper": max(percentile_upper, estimate),
    }
