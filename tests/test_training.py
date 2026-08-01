from collections.abc import Callable

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.training import (
    make_dummy_model,
    make_lightgbm_model,
    make_logistic_model,
)


def test_make_dummy_model_uses_class_prior_strategy() -> None:
    model = make_dummy_model()

    assert isinstance(model, DummyClassifier)
    assert model.get_params()["strategy"] == "prior"


def test_make_logistic_model_uses_planned_parameters() -> None:
    model = make_logistic_model(class_weight="balanced", random_seed=17)

    assert isinstance(model, LogisticRegression)
    params = model.get_params()
    assert params["C"] == pytest.approx(1.0)
    assert params["class_weight"] == "balanced"
    assert params["max_iter"] == 2000
    assert params["random_state"] == 17
    assert params["solver"] == "saga"


def test_make_lightgbm_model_uses_planned_parameters() -> None:
    model = make_lightgbm_model(random_seed=29, scale_pos_weight=2.5)

    assert isinstance(model, LGBMClassifier)
    params = model.get_params()
    assert params["objective"] == "binary"
    assert params["n_estimators"] == 500
    assert params["learning_rate"] == pytest.approx(0.03)
    assert params["num_leaves"] == 31
    assert params["max_depth"] == -1
    assert params["min_child_samples"] == 50
    assert params["subsample"] == pytest.approx(0.8)
    assert params["subsample_freq"] == 1
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["reg_alpha"] == pytest.approx(0.0)
    assert params["reg_lambda"] == pytest.approx(1.0)
    assert params["scale_pos_weight"] == pytest.approx(2.5)
    assert params["random_state"] == 29
    assert params["n_jobs"] == -1
    assert params["verbosity"] == -1


def test_make_lightgbm_model_applies_overrides_without_losing_explicit_inputs() -> None:
    model = make_lightgbm_model(
        random_seed=31,
        scale_pos_weight=3.0,
        n_estimators=7,
        learning_rate=0.2,
        num_leaves=15,
    )

    params = model.get_params()
    assert params["n_estimators"] == 7
    assert params["learning_rate"] == pytest.approx(0.2)
    assert params["num_leaves"] == 15
    assert params["random_state"] == 31
    assert params["scale_pos_weight"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("reserved_key", "value"),
    [
        ("objective", "multiclass"),
        ("n_jobs", 1),
        ("random_state", 99),
    ],
)
def test_make_lightgbm_model_rejects_each_reserved_override(
    reserved_key: str, value: object
) -> None:
    with pytest.raises(ValueError, match=rf"reserved/conflicting.*{reserved_key}"):
        make_lightgbm_model(random_seed=11, **{reserved_key: value})  # type: ignore[arg-type]


def test_make_lightgbm_model_lists_all_reserved_overrides_in_stable_order() -> None:
    with pytest.raises(ValueError) as exc_info:
        make_lightgbm_model(
            random_seed=11,
            random_state=99,
            objective="multiclass",  # type: ignore[arg-type]
            n_jobs=1,
        )

    assert str(exc_info.value) == (
        "reserved/conflicting LightGBM override keys: objective, n_jobs, random_state"
    )


def test_factories_return_independent_model_instances() -> None:
    first_dummy = make_dummy_model()
    second_dummy = make_dummy_model()
    first_logistic = make_logistic_model(class_weight=None, random_seed=1)
    second_logistic = make_logistic_model(class_weight=None, random_seed=1)
    first_lightgbm = make_lightgbm_model(random_seed=1)
    second_lightgbm = make_lightgbm_model(random_seed=1)

    first_dummy.set_params(strategy="most_frequent")
    first_logistic.set_params(C=2.0)
    first_lightgbm.set_params(n_estimators=3)

    assert first_dummy is not second_dummy
    assert second_dummy.get_params()["strategy"] == "prior"
    assert first_logistic is not second_logistic
    assert second_logistic.get_params()["C"] == pytest.approx(1.0)
    assert first_lightgbm is not second_lightgbm
    assert second_lightgbm.get_params()["n_estimators"] == 500


@pytest.mark.parametrize("class_weight", ["auto", "none", 1])
def test_make_logistic_model_rejects_unsupported_class_weight(
    class_weight: object,
) -> None:
    with pytest.raises(ValueError, match="class_weight.*None.*balanced"):
        make_logistic_model(class_weight=class_weight, random_seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda seed: make_logistic_model(class_weight=None, random_seed=seed),
        lambda seed: make_lightgbm_model(random_seed=seed),
    ],
)
@pytest.mark.parametrize("invalid_seed", [True, False, 1.0, "1"])
def test_model_factories_reject_non_integer_or_boolean_random_seed(
    factory: Callable[[object], object], invalid_seed: object
) -> None:
    with pytest.raises(ValueError, match="random_seed.*int"):
        factory(invalid_seed)


@pytest.mark.parametrize("scale_pos_weight", [0.0, -1.0, np.nan, np.inf, -np.inf, True])
def test_make_lightgbm_model_rejects_invalid_scale_pos_weight(
    scale_pos_weight: object,
) -> None:
    with pytest.raises(ValueError, match="scale_pos_weight.*finite.*greater than 0"):
        make_lightgbm_model(random_seed=1, scale_pos_weight=scale_pos_weight)  # type: ignore[arg-type]


def test_all_model_factories_fit_and_return_finite_binary_probabilities() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    target = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    models = [
        make_dummy_model(),
        make_logistic_model(class_weight=None, random_seed=11),
        make_lightgbm_model(random_seed=11, n_estimators=2, min_child_samples=1),
    ]

    for model in models:
        probabilities = model.fit(features, target).predict_proba(features)

        assert probabilities.shape == (len(features), 2)
        assert np.isfinite(probabilities).all()
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(features)))
