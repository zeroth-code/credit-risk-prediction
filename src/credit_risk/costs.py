import numpy as np
import pandas as pd

ALLOWED_ACTIONS = ("approve", "manual_review", "decline")


def _is_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _contains_boolean(values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return any(_is_boolean(value) for value in values)
    return False


def _one_dimensional_values(name: str, values: object) -> np.ndarray:
    if _is_boolean(values):
        raise ValueError(f"{name} must not be boolean")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _validated_probabilities(probabilities: object) -> np.ndarray:
    values = _one_dimensional_values("probabilities", probabilities)
    if _contains_boolean(values):
        raise ValueError("probabilities must not contain boolean values")
    try:
        probability_array = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain numeric values") from exc
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")
    if not ((probability_array >= 0.0) & (probability_array <= 1.0)).all():
        raise ValueError("probabilities must be between 0 and 1")
    return probability_array


def _validated_target(y_true: object) -> np.ndarray:
    target = _one_dimensional_values("y_true", y_true)
    if _contains_boolean(target):
        raise ValueError("y_true must not contain boolean values")
    try:
        binary = bool(np.isin(target, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true values must be 0 or 1") from exc
    if not binary:
        raise ValueError("y_true values must be 0 or 1")
    return target.astype(int, copy=False)


def _validated_loan_amount(loan_amount: object) -> np.ndarray:
    values = _one_dimensional_values("loan_amount", loan_amount)
    if _contains_boolean(values):
        raise ValueError("loan_amount must not contain boolean values")
    try:
        amounts = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("loan_amount must contain numeric values") from exc
    if not np.isfinite(amounts).all():
        raise ValueError("loan_amount must contain only finite values")
    if not (amounts >= 0.0).all():
        raise ValueError("loan_amount must contain only nonnegative values")
    return amounts


def _validated_actions(actions: object) -> np.ndarray:
    action_array = _one_dimensional_values("actions", actions)
    if bool(pd.isna(action_array).any()):
        raise ValueError("actions must not contain missing values")
    try:
        allowed = bool(np.isin(action_array, ALLOWED_ACTIONS).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("actions contain values that cannot be compared") from exc
    if not allowed:
        raise ValueError("actions must contain only approve, manual_review, or decline")
    return action_array.astype(str, copy=False)


def _validated_probability_cost(name: str, value: object) -> float:
    if _is_boolean(value):
        raise ValueError(f"{name} must not be boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value between 0 and 1") from exc
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _validated_review_cost(review_cost: object) -> float:
    if _is_boolean(review_cost):
        raise ValueError("review_cost must not be boolean")
    try:
        parsed = float(review_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError("review_cost must be a finite nonnegative numeric value") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError("review_cost must be finite and nonnegative")
    return parsed


def _validated_thresholds(approve_below: object, decline_at: object) -> tuple[float, float]:
    if _is_boolean(approve_below) or _is_boolean(decline_at):
        raise ValueError("thresholds must not be boolean")
    try:
        approve_threshold = float(approve_below)
        decline_threshold = float(decline_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be finite numeric values") from exc
    if not (
        np.isfinite(approve_threshold)
        and np.isfinite(decline_threshold)
        and 0.0 <= approve_threshold < decline_threshold <= 1.0
    ):
        raise ValueError("thresholds must satisfy 0 <= approve_below < decline_at <= 1")
    return approve_threshold, decline_threshold


def assign_actions(
    probabilities: object,
    *,
    approve_below: float,
    decline_at: float,
) -> np.ndarray:
    probability_array = _validated_probabilities(probabilities)
    approve_threshold, decline_threshold = _validated_thresholds(approve_below, decline_at)
    return _assign_actions_validated(
        probability_array,
        approve_below=approve_threshold,
        decline_at=decline_threshold,
    )


def _assign_actions_validated(
    probabilities: np.ndarray,
    *,
    approve_below: float,
    decline_at: float,
) -> np.ndarray:
    return np.where(
        probabilities < approve_below,
        "approve",
        np.where(probabilities >= decline_at, "decline", "manual_review"),
    )


def _policy_cost_validated(
    target: np.ndarray,
    amounts: np.ndarray,
    action_array: np.ndarray,
    *,
    lgd: float,
    margin: float,
    review_cost: float,
) -> float:
    approved_bad_loss = amounts[(action_array == "approve") & (target == 1)].sum() * lgd
    declined_good_cost = amounts[(action_array == "decline") & (target == 0)].sum() * margin
    manual_review_cost = (action_array == "manual_review").sum() * review_cost
    return float(approved_bad_loss + declined_good_cost + manual_review_cost)


def policy_cost(
    y_true: object,
    loan_amount: object,
    actions: object,
    *,
    lgd: float,
    margin: float,
    review_cost: float,
) -> float:
    target = _validated_target(y_true)
    amounts = _validated_loan_amount(loan_amount)
    action_array = _validated_actions(actions)
    if not len(target) == len(amounts) == len(action_array):
        raise ValueError("y_true, loan_amount, and actions must have the same length")
    validated_lgd = _validated_probability_cost("lgd", lgd)
    validated_margin = _validated_probability_cost("margin", margin)
    validated_review_cost = _validated_review_cost(review_cost)
    return _policy_cost_validated(
        target,
        amounts,
        action_array,
        lgd=validated_lgd,
        margin=validated_margin,
        review_cost=validated_review_cost,
    )


def search_policy(
    y_true: object,
    loan_amount: object,
    probabilities: object,
    *,
    lgd: float,
    margin: float,
    review_cost: float,
) -> pd.DataFrame:
    target = _validated_target(y_true)
    amounts = _validated_loan_amount(loan_amount)
    probability_array = _validated_probabilities(probabilities)
    if not len(target) == len(amounts) == len(probability_array):
        raise ValueError("y_true, loan_amount, and probabilities must have the same length")
    validated_lgd = _validated_probability_cost("lgd", lgd)
    validated_margin = _validated_probability_cost("margin", margin)
    validated_review_cost = _validated_review_cost(review_cost)

    grid = np.arange(5, 100, 5, dtype=float) / 100.0
    rows: list[dict[str, float]] = []
    for approve_index, approve_below in enumerate(grid[:-1]):
        for decline_at in grid[approve_index + 1 :]:
            actions = _assign_actions_validated(
                probability_array,
                approve_below=float(approve_below),
                decline_at=float(decline_at),
            )
            rows.append(
                {
                    "approve_below": float(approve_below),
                    "decline_at": float(decline_at),
                    "cost": _policy_cost_validated(
                        target,
                        amounts,
                        actions,
                        lgd=validated_lgd,
                        margin=validated_margin,
                        review_cost=validated_review_cost,
                    ),
                    "approval_rate": float(np.mean(actions == "approve")),
                    "review_rate": float(np.mean(actions == "manual_review")),
                    "decline_rate": float(np.mean(actions == "decline")),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values("cost", kind="stable").reset_index(drop=True)
