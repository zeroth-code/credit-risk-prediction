import math

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def _validate_random_seed(random_seed: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an int and must not be a bool")


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
    if (
        isinstance(scale_pos_weight, bool)
        or not isinstance(scale_pos_weight, (int, float))
        or not math.isfinite(scale_pos_weight)
        or scale_pos_weight <= 0
    ):
        raise ValueError("scale_pos_weight must be finite and greater than 0")

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
