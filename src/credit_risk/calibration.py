from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class CalibrationSelection:
    method: str
    scores: dict[str, float]


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
    bin_indices = np.minimum((probability_array * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        if members.any():
            error += float(members.mean()) * abs(
                float(probability_array[members].mean()) - float(target[members].mean())
            )
    return error


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
