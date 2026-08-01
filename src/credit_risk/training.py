import math

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

_RESERVED_LIGHTGBM_OVERRIDE_KEYS = ("objective", "n_jobs", "random_state")


def _validate_random_seed(random_seed: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an int and must not be a bool")


def _validate_scale_pos_weight(scale_pos_weight: float) -> None:
    if (
        isinstance(scale_pos_weight, bool)
        or not isinstance(scale_pos_weight, (int, float))
        or not math.isfinite(scale_pos_weight)
        or scale_pos_weight <= 0
    ):
        raise ValueError("scale_pos_weight must be finite and greater than 0")


def _validate_binary_target(y: np.ndarray) -> np.ndarray:
    try:
        target = np.asarray(y)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be a one-dimensional binary array") from exc

    if target.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if target.size == 0:
        raise ValueError("y must be non-empty")
    try:
        contains_only_binary_values = bool(np.isin(target, [0, 1]).all())
    except (TypeError, ValueError) as exc:
        raise ValueError("y values must be 0 and 1") from exc
    if not contains_only_binary_values:
        raise ValueError("y values must be 0 and 1")

    has_zero = bool(np.any(target == 0))
    has_one = bool(np.any(target == 1))
    if not has_zero or not has_one:
        raise ValueError("y must contain both classes 0 and 1")
    return target


def positive_class_weight(y: np.ndarray) -> float:
    target = _validate_binary_target(y)
    negatives = int(np.count_nonzero(target == 0))
    positives = int(np.count_nonzero(target == 1))
    return float(negatives / positives)


def random_undersample_indices(y: np.ndarray, random_seed: int) -> np.ndarray:
    _validate_random_seed(random_seed)
    target = _validate_binary_target(y)
    negative_indices = np.flatnonzero(target == 0)
    positive_indices = np.flatnonzero(target == 1)
    if positive_indices.size > negative_indices.size:
        raise ValueError("positive class must be the minority or tied for random undersampling")

    rng = np.random.default_rng(random_seed)
    sampled_negative_indices = rng.choice(
        negative_indices, size=positive_indices.size, replace=False
    )
    return np.sort(np.concatenate([sampled_negative_indices, positive_indices]))


def _feature_row_count(name: str, features: object) -> int:
    shape = getattr(features, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must expose a row count") from exc
    try:
        return len(features)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must expose a row count") from exc


def _validate_tuning_inputs(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    *,
    n_trials: int,
    random_seed: int,
    scale_pos_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        raise ValueError("n_trials must be a positive int and must not be a bool")
    _validate_random_seed(random_seed)
    _validate_scale_pos_weight(scale_pos_weight)

    try:
        train_target = _validate_binary_target(y_train)
    except ValueError as exc:
        raise ValueError(f"train target invalid: {exc}") from exc
    try:
        validation_target = _validate_binary_target(y_validation)
    except ValueError as exc:
        raise ValueError(f"validation target invalid: {exc}") from exc

    if _feature_row_count("x_train", x_train) != train_target.size:
        raise ValueError("train feature and target rows must match")
    if _feature_row_count("x_validation", x_validation) != validation_target.size:
        raise ValueError("validation feature and target rows must match")
    return train_target, validation_target


def run_lightgbm_study(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    *,
    n_trials: int,
    random_seed: int,
    scale_pos_weight: float = 1.0,
) -> optuna.Study:
    train_target, validation_target = _validate_tuning_inputs(
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_trials=n_trials,
        random_seed=random_seed,
        scale_pos_weight=scale_pos_weight,
    )

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        model = make_lightgbm_model(
            random_seed=random_seed,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        probabilities = model.fit(x_train, train_target).predict_proba(x_validation)[:, 1]
        return float(average_precision_score(validation_target, probabilities))

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
        )
        study.optimize(objective, n_trials=n_trials)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
    return study


def tune_lightgbm(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    *,
    n_trials: int = 30,
    random_seed: int,
    scale_pos_weight: float = 1.0,
) -> dict[str, float | int]:
    study = run_lightgbm_study(
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_trials=n_trials,
        random_seed=random_seed,
        scale_pos_weight=scale_pos_weight,
    )
    return dict(study.best_params)


def make_dummy_model() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def make_logistic_model(*, class_weight: str | None, random_seed: int) -> LogisticRegression:
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")
    _validate_random_seed(random_seed)

    return LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=2000,
        random_state=random_seed,
        solver="saga",
    )


def make_lightgbm_model(
    *, random_seed: int, scale_pos_weight: float = 1.0, **overrides: float | int
) -> LGBMClassifier:
    _validate_random_seed(random_seed)
    _validate_scale_pos_weight(scale_pos_weight)

    conflicting_keys = [key for key in _RESERVED_LIGHTGBM_OVERRIDE_KEYS if key in overrides]
    if conflicting_keys:
        joined_keys = ", ".join(conflicting_keys)
        raise ValueError(f"reserved/conflicting LightGBM override keys: {joined_keys}")

    params: dict[str, object] = {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(overrides)
    return LGBMClassifier(**params)
