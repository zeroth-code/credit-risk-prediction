import importlib.util
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from credit_risk.calibration import evaluate_calibration
from credit_risk.costs import search_policy
from credit_risk.features import load_feature_dictionary
from credit_risk.training import (
    positive_class_weight,
    random_undersample_indices,
    run_lightgbm_study,
    tune_lightgbm,
)

TUNED_PARAMETER_NAMES = {
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}


def test_positive_class_weight_returns_negative_to_positive_ratio() -> None:
    target = np.array([0, 0, 0, 1])

    assert positive_class_weight(target) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.array([]), "non-empty"),
        (np.array([[0, 1]]), "one-dimensional"),
        (np.array([0, 0]), "both classes"),
        (np.array([1, 1]), "both classes"),
        (np.array([0, 2, 1]), "0 and 1"),
        (np.array(["0", "1"]), "0 and 1"),
    ],
)
def test_positive_class_weight_rejects_invalid_targets(target: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        positive_class_weight(target)


def test_random_undersample_indices_keeps_all_positives_and_equal_negatives() -> None:
    target = np.array([0, 1, 0, 0, 1, 0])

    indices = random_undersample_indices(target, random_seed=17)

    assert indices.ndim == 1
    assert np.issubdtype(indices.dtype, np.integer)
    assert indices.tolist() == sorted(indices.tolist())
    assert np.flatnonzero(target[indices] == 1).size == 2
    assert np.flatnonzero(target[indices] == 0).size == 2
    assert set(np.flatnonzero(target == 1)).issubset(indices)


def test_random_undersample_indices_is_deterministic_and_does_not_modify_input() -> None:
    target = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    original = target.copy()

    first = random_undersample_indices(target, random_seed=23)
    second = random_undersample_indices(target, random_seed=23)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(target, original)


@pytest.mark.parametrize("random_seed", [True, False, 1.0, "1"])
def test_random_undersample_indices_rejects_non_integer_or_boolean_seed(
    random_seed: object,
) -> None:
    with pytest.raises(ValueError, match="random_seed.*int"):
        random_undersample_indices(
            np.array([0, 0, 1]),
            random_seed=random_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.array([]), "non-empty"),
        (np.array([[0, 1]]), "one-dimensional"),
        (np.array([0, 0]), "both classes"),
        (np.array([1, 1]), "both classes"),
        (np.array([0, 2, 1]), "0 and 1"),
        (np.array([0, 1, 1]), "positive.*minority"),
    ],
)
def test_random_undersample_indices_rejects_invalid_targets(
    target: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        random_undersample_indices(target, random_seed=11)


def _tuning_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, target = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=19,
    )
    return features[:80], target[:80], features[80:], target[80:]


def _assert_tuned_parameter_ranges(params: dict[str, float | int]) -> None:
    assert set(params) == TUNED_PARAMETER_NAMES
    assert 200 <= params["n_estimators"] <= 900
    assert 0.01 <= params["learning_rate"] <= 0.08
    assert 15 <= params["num_leaves"] <= 63
    assert 20 <= params["min_child_samples"] <= 150
    assert 0.65 <= params["subsample"] <= 1.0
    assert 0.65 <= params["colsample_bytree"] <= 1.0
    assert 1e-6 <= params["reg_alpha"] <= 5.0
    assert 1e-6 <= params["reg_lambda"] <= 10.0


def test_run_lightgbm_study_returns_completed_reproducible_trials() -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()

    first = run_lightgbm_study(
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_trials=2,
        random_seed=31,
        scale_pos_weight=2.0,
    )
    second = run_lightgbm_study(
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_trials=2,
        random_seed=31,
        scale_pos_weight=2.0,
    )

    assert isinstance(first, optuna.Study)
    assert first.direction == optuna.study.StudyDirection.MAXIMIZE
    assert len(first.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in first.trials)
    _assert_tuned_parameter_ranges(first.best_params)
    assert second.best_params == first.best_params
    assert second.best_value == pytest.approx(first.best_value, abs=0.0)


def test_tune_lightgbm_returns_only_best_parameter_dictionary() -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()

    params = tune_lightgbm(
        x_train,
        y_train,
        x_validation,
        y_validation,
        n_trials=2,
        random_seed=37,
        scale_pos_weight=1.0,
    )

    assert isinstance(params, dict)
    _assert_tuned_parameter_ranges(params)


def test_tune_lightgbm_defaults_to_thirty_trials() -> None:
    assert inspect.signature(tune_lightgbm).parameters["n_trials"].default == 30


def test_run_lightgbm_study_temporarily_suppresses_info_logging() -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()
    original_verbosity = optuna.logging.get_verbosity()
    messages: list[str] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = optuna.logging.get_logger("optuna")
    handler = RecordingHandler()
    logger.addHandler(handler)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        run_lightgbm_study(
            x_train,
            y_train,
            x_validation,
            y_validation,
            n_trials=1,
            random_seed=39,
            scale_pos_weight=1.0,
        )

        assert not any("A new study created" in message for message in messages)
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
    finally:
        optuna.logging.set_verbosity(original_verbosity)
        logger.removeHandler(handler)


@pytest.mark.parametrize("n_trials", [True, False, 0, -1, 1.0, "1"])
def test_run_lightgbm_study_rejects_invalid_trial_count(n_trials: object) -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()

    with pytest.raises(ValueError, match="n_trials.*positive int"):
        run_lightgbm_study(
            x_train,
            y_train,
            x_validation,
            y_validation,
            n_trials=n_trials,  # type: ignore[arg-type]
            random_seed=41,
            scale_pos_weight=1.0,
        )


@pytest.mark.parametrize("random_seed", [True, False, 1.0, "1"])
def test_run_lightgbm_study_rejects_invalid_random_seed(random_seed: object) -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()

    with pytest.raises(ValueError, match="random_seed.*int"):
        run_lightgbm_study(
            x_train,
            y_train,
            x_validation,
            y_validation,
            n_trials=1,
            random_seed=random_seed,  # type: ignore[arg-type]
            scale_pos_weight=1.0,
        )


@pytest.mark.parametrize("scale_pos_weight", [True, 0.0, -1.0, np.nan, np.inf, "1"])
def test_run_lightgbm_study_rejects_invalid_positive_class_weight(
    scale_pos_weight: object,
) -> None:
    x_train, y_train, x_validation, y_validation = _tuning_data()

    with pytest.raises(ValueError, match="scale_pos_weight.*finite.*greater than 0"):
        run_lightgbm_study(
            x_train,
            y_train,
            x_validation,
            y_validation,
            n_trials=1,
            random_seed=41,
            scale_pos_weight=scale_pos_weight,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("x_train_rows", "y_train", "x_validation_rows", "y_validation", "message"),
    [
        (5, np.array([0, 0, 1, 1]), 4, np.array([0, 0, 1, 1]), "train.*rows"),
        (4, np.array([0, 0, 1, 1]), 5, np.array([0, 0, 1, 1]), "validation.*rows"),
        (4, np.array([0, 0, 0, 0]), 4, np.array([0, 0, 1, 1]), "train.*both classes"),
        (4, np.array([0, 0, 1, 1]), 4, np.array([1, 1, 1, 1]), "validation.*both classes"),
    ],
)
def test_run_lightgbm_study_rejects_invalid_partition_inputs(
    x_train_rows: int,
    y_train: np.ndarray,
    x_validation_rows: int,
    y_validation: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_lightgbm_study(
            np.zeros((x_train_rows, 2)),
            y_train,
            np.zeros((x_validation_rows, 2)),
            y_validation,
            n_trials=1,
            random_seed=41,
            scale_pos_weight=1.0,
        )


def _load_train_script(module_name: str) -> object:
    script_path = Path("scripts/train.py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_partition(*, rows: int, positives: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dictionary = load_feature_dictionary()
    numeric_columns = list(
        dict.fromkeys(
            dictionary["challenger"]["numeric"] + dictionary["full_underwriting"]["numeric"]
        )
    )
    categorical_columns = list(
        dict.fromkeys(
            dictionary["challenger"]["categorical"] + dictionary["full_underwriting"]["categorical"]
        )
    )
    target = np.zeros(rows, dtype=int)
    target[rng.choice(rows, size=positives, replace=False)] = 1
    payload: dict[str, object] = {"bad": target}
    for column_number, column in enumerate(numeric_columns, start=1):
        payload[column] = rng.normal(
            loc=column_number * 10.0 + target * 2.0,
            scale=3.0,
            size=rows,
        )
    for column_number, column in enumerate(categorical_columns):
        categories = np.array([f"{column}_a", f"{column}_b", f"{column}_c"])
        payload[column] = categories[(np.arange(rows) + column_number + target) % 3]
    return pd.DataFrame(payload)


def _fixed_study() -> optuna.Study:
    params: dict[str, float | int] = {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-4,
        "reg_lambda": 1.0,
    }
    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        "n_estimators": optuna.distributions.IntDistribution(200, 900),
        "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.08, log=True),
        "num_leaves": optuna.distributions.IntDistribution(15, 63),
        "min_child_samples": optuna.distributions.IntDistribution(20, 150),
        "subsample": optuna.distributions.FloatDistribution(0.65, 1.0),
        "colsample_bytree": optuna.distributions.FloatDistribution(0.65, 1.0),
        "reg_alpha": optuna.distributions.FloatDistribution(1e-6, 5.0, log=True),
        "reg_lambda": optuna.distributions.FloatDistribution(1e-6, 10.0, log=True),
    }
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=0.75,
        )
    )
    return study


def test_select_challenger_lightgbm_strategy_uses_stable_tie_break() -> None:
    train_script = _load_train_script("train_tie_break")
    records = [
        {
            "model": "lightgbm",
            "feature_set": "challenger",
            "imbalance_strategy": strategy,
            "average_precision": 0.7,
        }
        for strategy in ("natural", "weighted", "undersampled")
    ]
    records.append(
        {
            "model": "lightgbm",
            "feature_set": "full_underwriting",
            "imbalance_strategy": "natural",
            "average_precision": 1.0,
        }
    )

    assert train_script.select_challenger_lightgbm_strategy(records) == "natural"


def test_partition_target_rejects_values_that_would_be_truncated_to_binary() -> None:
    train_script = _load_train_script("train_target_validation")

    with pytest.raises(ValueError, match="train.*0 and 1"):
        train_script.partition_target(
            pd.DataFrame({"bad": [0, 0.5, 1]}),
            partition_name="train",
        )


def test_calibration_loan_amounts_converts_numeric_values_without_modifying_frame() -> None:
    train_script = _load_train_script("train_calibration_loan_amounts")
    calibration = pd.DataFrame({"loan_amnt": pd.Series(["1000", " 2500.5 "], dtype="string")})
    original = calibration.copy(deep=True)

    amounts = train_script.calibration_loan_amounts(calibration)

    np.testing.assert_array_equal(amounts, np.array([1000.0, 2500.5]))
    pd.testing.assert_frame_equal(calibration, original)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-number", "numeric"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-1.0, "nonnegative"),
    ],
)
def test_calibration_loan_amounts_rejects_invalid_values(value: object, message: str) -> None:
    train_script = _load_train_script("train_invalid_calibration_loan_amounts")

    with pytest.raises(ValueError, match=f"calibration loan_amnt.*{message}"):
        train_script.calibration_loan_amounts(pd.DataFrame({"loan_amnt": [value]}))


@pytest.mark.parametrize(
    "values",
    [
        [True, False],
        [np.bool_(True)],
        ["1000", True],
        [1000.0, np.bool_(False)],
    ],
)
def test_calibration_loan_amounts_rejects_boolean_values(values: list[object]) -> None:
    train_script = _load_train_script("train_boolean_calibration_loan_amounts")

    with pytest.raises(ValueError, match="calibration loan_amnt.*boolean"):
        train_script.calibration_loan_amounts(pd.DataFrame({"loan_amnt": values}))


def test_train_main_anchors_project_paths_when_called_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_script = _load_train_script("train_project_paths")
    project_root = train_script.PROJECT_ROOT
    observed: dict[str, Path] = {}
    saved_calibration_metrics: dict[str, object] = {}
    saved_policy: dict[str, object] = {}
    saved_sensitivity = pd.DataFrame()
    costs = SimpleNamespace(
        base=SimpleNamespace(lgd=0.6, margin=0.05, review_cost=30.0),
        lgd_values=[0.4, 0.6, 0.8],
        margin_values=[0.03, 0.05, 0.08],
        review_cost_values=[15.0, 30.0, 60.0],
    )
    config = SimpleNamespace(
        random_seed=47,
        processed_dir=Path("relative/processed"),
        artifact_dir=Path("relative/artifacts"),
        calibration_methods=["uncalibrated", "sigmoid", "isotonic"],
        costs=costs,
    )
    feature_dictionary = {
        "challenger": {"numeric": ["loan_amnt"], "categorical": ["purpose"]},
        "full_underwriting": {"numeric": ["loan_amnt"], "categorical": ["grade"]},
    }
    train = pd.DataFrame({"bad": [0, 0, 1, 1], "loan_amnt": [500, 600, 700, 800]})
    validation = pd.DataFrame({"bad": [0, 1], "loan_amnt": [900, 1000]})
    calibration = pd.DataFrame({"bad": [0, 1], "loan_amnt": [1000, 2000]})

    class FastPreprocessor:
        def transform(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(frame), 1))

    matrices = {
        "challenger": {
            "tree_train": np.zeros((4, 1)),
            "tree_validation": np.zeros((2, 1)),
            "tree_preprocessor": FastPreprocessor(),
        }
    }

    def fake_load_config(path: str | Path) -> SimpleNamespace:
        observed["config"] = Path(path)
        return config

    def fake_load_feature_dictionary(path: str | Path) -> dict[str, object]:
        observed["features_load"] = Path(path)
        return feature_dictionary

    def fake_load_partitions(
        processed_dir: Path, required_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        observed["processed"] = processed_dir
        assert required_columns == ["loan_amnt", "purpose", "grade"]
        return train, validation, calibration

    def fake_build_feature_matrices(
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        *,
        feature_dictionary_path: str | Path,
    ) -> dict[str, dict[str, object]]:
        observed["features_build"] = Path(feature_dictionary_path)
        assert train_frame is train
        assert validation_frame is validation
        return matrices

    def fake_build_feature_frame(
        frame: pd.DataFrame,
        columns: list[str],
        *,
        path: str | Path,
    ) -> pd.DataFrame:
        observed["calibration_build"] = Path(path)
        assert frame is calibration
        assert columns == ["loan_amnt", "purpose"]
        return pd.DataFrame({"loan_amnt": [1000.0, 2000.0], "purpose": ["a", "b"]})

    def fake_run_experiments(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "model": "lightgbm",
                "feature_set": "challenger",
                "imbalance_strategy": strategy,
                "average_precision": score,
            }
            for strategy, score in (
                ("natural", 0.8),
                ("weighted", 0.7),
                ("undersampled", 0.6),
            )
        ]

    class FastModel:
        def fit(self, x: object, y: np.ndarray) -> "FastModel":
            return self

        def predict_proba(self, x: object) -> np.ndarray:
            probabilities = np.linspace(0.25, 0.75, x.shape[0])  # type: ignore[attr-defined]
            return np.column_stack([1.0 - probabilities, probabilities])

    def fake_save_training_artifacts(
        artifact_dir: Path,
        *,
        model: object,
        preprocessor: object,
        metrics_payload: dict[str, object],
        study: object,
    ) -> None:
        observed["artifacts"] = artifact_dir

    def fake_save_calibration_artifacts(
        artifact_dir: Path,
        *,
        calibrated_model: object,
        metrics_payload: dict[str, object],
        curve: pd.DataFrame,
    ) -> None:
        observed["calibration_artifacts"] = artifact_dir
        saved_calibration_metrics.update(metrics_payload)

    def fake_save_policy_artifacts(
        artifact_dir: Path,
        *,
        policy_payload: dict[str, object],
        sensitivity: pd.DataFrame,
    ) -> None:
        nonlocal saved_sensitivity
        observed["policy_artifacts"] = artifact_dir
        saved_policy.update(policy_payload)
        saved_sensitivity = sensitivity.copy()

    monkeypatch.setattr(train_script, "load_config", fake_load_config)
    monkeypatch.setattr(train_script, "load_feature_dictionary", fake_load_feature_dictionary)
    monkeypatch.setattr(train_script, "load_partitions", fake_load_partitions)
    monkeypatch.setattr(train_script, "build_feature_matrices", fake_build_feature_matrices)
    monkeypatch.setattr(train_script, "build_feature_frame", fake_build_feature_frame)
    monkeypatch.setattr(train_script, "run_experiments", fake_run_experiments)
    monkeypatch.setattr(
        train_script,
        "run_lightgbm_study",
        lambda *args, **kwargs: SimpleNamespace(best_params={}),
    )
    monkeypatch.setattr(train_script, "make_lightgbm_model", lambda **kwargs: FastModel())
    selected_probabilities = np.array([0.05, 0.95])
    evaluation = SimpleNamespace(
        selection=SimpleNamespace(method="sigmoid"),
        probabilities={
            "uncalibrated": np.array([0.4, 0.6]),
            "sigmoid": selected_probabilities,
        },
        metrics={
            "uncalibrated": {
                "status": "evaluated",
                "probability_source": "base_model_calibration_partition",
                "brier_score": 0.0625,
                "log_loss": 0.2876820724517809,
                "expected_calibration_error": 0.25,
            },
            "sigmoid": {
                "status": "evaluated",
                "probability_source": "stratified_oof",
                "brier_score": 0.0025,
                "log_loss": 0.05129329438755058,
                "expected_calibration_error": 0.05,
            },
            "isotonic": {"status": "skipped", "skip_reason": "small sample"},
        },
        curve=pd.DataFrame(
            {
                "method": ["uncalibrated"],
                "bin_index": [0],
                "bin_lower": [0.0],
                "bin_upper": [1.0],
                "sample_count": [2],
                "mean_probability": [0.5],
                "observed_default_rate": [0.5],
            }
        ),
        folds=2,
        evaluation_protocol="stratified_oof",
    )
    monkeypatch.setattr(train_script, "evaluate_calibration", lambda *args, **kwargs: evaluation)

    class FullRefitArtifact:
        def __init__(self) -> None:
            self.predict_calls = 0

        def predict_proba(self, features: object) -> np.ndarray:
            self.predict_calls += 1
            probabilities = np.array([0.95, 0.05])
            return np.column_stack([1.0 - probabilities, probabilities])

    full_refit_artifact = FullRefitArtifact()
    monkeypatch.setattr(
        train_script,
        "fit_calibrated_model",
        lambda model, *args, **kwargs: full_refit_artifact,
    )
    search_calls: list[dict[str, object]] = []

    def fake_search_policy(
        y_true: object,
        loan_amount: object,
        probabilities: object,
        *,
        lgd: float,
        margin: float,
        review_cost: float,
    ) -> pd.DataFrame:
        search_calls.append(
            {
                "y_true": np.asarray(y_true).copy(),
                "loan_amount": np.asarray(loan_amount).copy(),
                "probabilities": np.asarray(probabilities).copy(),
                "lgd": lgd,
                "margin": margin,
                "review_cost": review_cost,
            }
        )
        is_base_search = len(search_calls) == 1
        return pd.DataFrame(
            [
                {
                    "approve_below": 0.2 if is_base_search else 0.25,
                    "decline_at": 0.7 if is_base_search else 0.75,
                    "cost": 123.0 if is_base_search else 456.0,
                    "approval_rate": 0.5,
                    "review_rate": 0.0,
                    "decline_rate": 0.5,
                }
            ]
        )

    monkeypatch.setattr(train_script, "search_policy", fake_search_policy, raising=False)
    monkeypatch.setattr(train_script, "save_training_artifacts", fake_save_training_artifacts)
    monkeypatch.setattr(
        train_script,
        "save_calibration_artifacts",
        fake_save_calibration_artifacts,
    )
    monkeypatch.setattr(
        train_script,
        "save_policy_artifacts",
        fake_save_policy_artifacts,
        raising=False,
    )
    monkeypatch.chdir(tmp_path)

    train_script.main(n_trials=1)

    assert observed == {
        "config": project_root / "configs/base.yaml",
        "features_load": project_root / "configs/features.yaml",
        "features_build": project_root / "configs/features.yaml",
        "calibration_build": project_root / "configs/features.yaml",
        "processed": project_root / "relative/processed",
        "artifacts": project_root / "relative/artifacts",
        "calibration_artifacts": project_root / "relative/artifacts",
        "policy_artifacts": project_root / "relative/artifacts",
    }
    assert saved_calibration_metrics["evaluation_protocol"] == "stratified_oof"
    assert saved_calibration_metrics["folds"] == 2
    assert full_refit_artifact.predict_calls == 0
    assert len(search_calls) == 28
    for call in search_calls:
        np.testing.assert_array_equal(call["y_true"], np.array([0, 1]))
        np.testing.assert_array_equal(call["loan_amount"], np.array([1000.0, 2000.0]))
        np.testing.assert_array_equal(call["probabilities"], selected_probabilities)
    assert {(call["lgd"], call["margin"], call["review_cost"]) for call in search_calls[1:]} == {
        (lgd, margin, review_cost)
        for lgd in costs.lgd_values
        for margin in costs.margin_values
        for review_cost in costs.review_cost_values
    }
    assert saved_policy == {
        "approve_below": 0.2,
        "decline_at": 0.7,
        "lgd": 0.6,
        "margin": 0.05,
        "review_cost": 30.0,
        "calibration_cost": 123.0,
        "calibration_approval_rate": 0.5,
        "calibration_review_rate": 0.0,
        "calibration_decline_rate": 0.5,
        "calibration_samples": 2,
        "total_loan_amount": 3000.0,
        "currency": "USD",
        "calibration_cost_per_1000_applications": 61500.0,
        "selected_calibration_method": "sigmoid",
        "probability_source": "stratified_oof",
        "selection_partition": "calibration",
        "threshold_selection_protocol": "grid_search_on_calibration_evaluation_probabilities",
        "calibration_evaluation_protocol": "stratified_oof",
    }
    assert saved_sensitivity.columns.tolist() == [
        "lgd",
        "margin",
        "review_cost",
        "is_base_scenario",
        "optimal_approve_below",
        "optimal_decline_at",
        "optimal_cost",
        "optimal_cost_per_1000_applications",
        "optimal_approval_rate",
        "optimal_review_rate",
        "optimal_decline_rate",
        "base_approve_below",
        "base_decline_at",
        "frozen_base_cost",
        "frozen_base_cost_per_1000_applications",
        "frozen_base_approval_rate",
        "frozen_base_review_rate",
        "frozen_base_decline_rate",
    ]
    assert len(saved_sensitivity) == 27
    assert saved_sensitivity["is_base_scenario"].sum() == 1
    assert (saved_sensitivity["optimal_approve_below"] == 0.25).all()
    assert (saved_sensitivity["base_approve_below"] == 0.2).all()
    assert np.allclose(
        saved_sensitivity["optimal_cost_per_1000_applications"],
        saved_sensitivity["optimal_cost"] / 2 * 1000,
    )
    assert np.allclose(
        saved_sensitivity["frozen_base_cost_per_1000_applications"],
        saved_sensitivity["frozen_base_cost"] / 2 * 1000,
    )


def test_train_main_writes_reproducible_calibrated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_script = _load_train_script("train_integration")
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    processed_dir.mkdir()
    train = _synthetic_partition(rows=120, positives=30, seed=101)
    validation = _synthetic_partition(rows=60, positives=20, seed=103)
    calibration = _synthetic_partition(rows=80, positives=24, seed=107)
    validation.loc[0, "purpose"] = "validation_only_purpose"
    validation.loc[0, "annual_inc"] = 10_000_000.0
    validation.loc[1, "annual_inc"] = np.nan
    calibration.loc[0, "purpose"] = "calibration_only_purpose"
    calibration.loc[0, "annual_inc"] = 20_000_000.0
    calibration.loc[1, "annual_inc"] = np.nan
    train_path = processed_dir / "train.parquet"
    validation_path = processed_dir / "validation.parquet"
    calibration_path = processed_dir / "calibration.parquet"
    test_path = processed_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    calibration.to_parquet(calibration_path, index=False)
    _synthetic_partition(rows=40, positives=10, seed=109).to_parquet(test_path, index=False)
    input_bytes = {
        train_path: train_path.read_bytes(),
        validation_path: validation_path.read_bytes(),
        calibration_path: calibration_path.read_bytes(),
        test_path: test_path.read_bytes(),
    }
    costs = SimpleNamespace(
        base=SimpleNamespace(lgd=0.6, margin=0.05, review_cost=30.0),
        lgd_values=[0.4, 0.6, 0.8],
        margin_values=[0.03, 0.05, 0.08],
        review_cost_values=[15.0, 30.0, 60.0],
    )
    config = SimpleNamespace(
        random_seed=43,
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
        calibration_methods=["uncalibrated", "sigmoid", "isotonic"],
        costs=costs,
    )
    monkeypatch.setattr(train_script, "load_config", lambda path: config)

    original_read_parquet = train_script.pd.read_parquet
    read_partitions: list[str] = []

    def guarded_read_parquet(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        partition_name = Path(path).stem
        read_partitions.append(partition_name)
        if partition_name == "test":
            raise AssertionError("test partition must not be read during training")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(train_script.pd, "read_parquet", guarded_read_parquet)

    original_make_lightgbm_model = train_script.make_lightgbm_model

    def fast_make_lightgbm_model(
        *, random_seed: int, scale_pos_weight: float = 1.0, **overrides: float | int
    ) -> object:
        overrides["n_estimators"] = 8
        overrides["min_child_samples"] = 5
        return original_make_lightgbm_model(
            random_seed=random_seed,
            scale_pos_weight=scale_pos_weight,
            **overrides,
        )

    monkeypatch.setattr(train_script, "make_lightgbm_model", fast_make_lightgbm_model)
    study_calls: list[dict[str, float | int]] = []

    def fake_run_lightgbm_study(
        x_train: object,
        y_train: np.ndarray,
        x_validation: object,
        y_validation: np.ndarray,
        *,
        n_trials: int,
        random_seed: int,
        scale_pos_weight: float = 1.0,
    ) -> optuna.Study:
        study_calls.append(
            {
                "train_rows": x_train.shape[0],  # type: ignore[attr-defined]
                "train_target_rows": len(y_train),
                "validation_rows": x_validation.shape[0],  # type: ignore[attr-defined]
                "validation_target_rows": len(y_validation),
                "n_trials": n_trials,
                "random_seed": random_seed,
                "scale_pos_weight": scale_pos_weight,
            }
        )
        return _fixed_study()

    monkeypatch.setattr(train_script, "run_lightgbm_study", fake_run_lightgbm_study)

    train_script.main(n_trials=1)

    expected_artifacts = {
        "uncalibrated_model.joblib",
        "preprocessor.joblib",
        "validation_metrics.json",
        "tuning_trials.csv",
        "calibrated_model.joblib",
        "calibration_metrics.json",
        "calibration_curve.csv",
        "policy.json",
        "cost_sensitivity.csv",
    }
    assert {path.name for path in artifact_dir.iterdir()} == expected_artifacts
    model = joblib.load(artifact_dir / "uncalibrated_model.joblib")
    calibrated_model = joblib.load(artifact_dir / "calibrated_model.joblib")
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    metrics_payload = json.loads(
        (artifact_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    calibration_metrics = json.loads(
        (artifact_dir / "calibration_metrics.json").read_text(encoding="utf-8")
    )
    calibration_curve = pd.read_csv(artifact_dir / "calibration_curve.csv")
    policy_text = (artifact_dir / "policy.json").read_text(encoding="utf-8")
    sensitivity_text = (artifact_dir / "cost_sensitivity.csv").read_text(encoding="utf-8")
    policy = json.loads(policy_text)
    cost_sensitivity = pd.read_csv(artifact_dir / "cost_sensitivity.csv")
    trials = pd.read_csv(artifact_dir / "tuning_trials.csv")

    assert read_partitions == ["train", "validation", "calibration"]
    assert calibration_metrics["calibration_samples"] == len(calibration)
    assert calibration_metrics["calibration_prevalence"] == pytest.approx(calibration["bad"].mean())
    assert calibration_metrics["evaluation_protocol"] == "stratified_oof"
    assert calibration_metrics["evaluation_partition"] == "calibration"
    assert calibration_metrics["folds"] == 5
    assert calibration_metrics["random_seed"] == 43
    assert list(calibration_metrics["methods"]) == config.calibration_methods
    for method in ("uncalibrated", "sigmoid"):
        method_metrics = calibration_metrics["methods"][method]
        assert set(method_metrics) == {
            "status",
            "probability_source",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        }
        assert all(
            np.isfinite(method_metrics[key])
            for key in ("brier_score", "log_loss", "expected_calibration_error")
        )
    assert calibration_metrics["methods"]["isotonic"]["status"] == "skipped"
    assert (
        "at least 1000 calibration samples"
        in calibration_metrics["methods"]["isotonic"]["skip_reason"]
    )
    selected_method = calibration_metrics["selected_method"]
    assert selected_method == min(
        ["uncalibrated", "sigmoid"],
        key=lambda method: calibration_metrics["methods"][method]["brier_score"],
    )
    assert calibration_metrics["artifact"]["method"] == selected_method
    if selected_method == "uncalibrated":
        assert calibration_metrics["artifact"] == {
            "method": "uncalibrated",
            "fit_protocol": "base_model_train_fit",
            "fit_partition": "train",
        }
    else:
        assert calibration_metrics["artifact"] == {
            "method": selected_method,
            "fit_protocol": "full_calibration_refit",
            "fit_partition": "calibration",
        }
    assert calibration_curve.columns.tolist() == [
        "method",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "sample_count",
        "mean_probability",
        "observed_default_rate",
    ]
    evaluated_methods = ["uncalibrated", "sigmoid"]
    assert len(calibration_curve) == 10 * len(evaluated_methods)
    assert calibration_curve["method"].drop_duplicates().tolist() == evaluated_methods
    assert calibration_curve.groupby("method", sort=False)["sample_count"].sum().to_dict() == {
        method: len(calibration) for method in evaluated_methods
    }

    assert metrics_payload["primary_feature_set"] == "challenger"
    assert metrics_payload["random_seed"] == 43
    assert metrics_payload["n_trials"] == 1
    assert metrics_payload["positive_class_weight"] == pytest.approx(3.0)
    assert set(metrics_payload["tuned_best_params"]) == TUNED_PARAMETER_NAMES
    assert set(metrics_payload["tuned_metrics"]) >= {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ks",
        "precision",
        "recall",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    expected_experiments = {
        ("dummy", "challenger", "natural"),
        ("logistic_regression", "challenger", "natural"),
        ("logistic_regression", "challenger", "weighted"),
        ("logistic_regression", "full_underwriting", "natural"),
        ("logistic_regression", "full_underwriting", "weighted"),
        ("lightgbm", "challenger", "natural"),
        ("lightgbm", "challenger", "weighted"),
        ("lightgbm", "challenger", "undersampled"),
        ("lightgbm", "full_underwriting", "natural"),
    }
    experiments = metrics_payload["experiments"]
    assert {
        (record["model"], record["feature_set"], record["imbalance_strategy"])
        for record in experiments
    } == expected_experiments
    for record in experiments:
        assert record["train_samples"] > 0
        assert record["validation_samples"] == len(validation)
        assert 0.0 < record["train_prevalence"] < 1.0
        assert record["validation_prevalence"] == pytest.approx(validation["bad"].mean())
        assert set(record) >= set(metrics_payload["tuned_metrics"])

    challenger_lightgbm_records = [
        record
        for record in experiments
        if record["model"] == "lightgbm" and record["feature_set"] == "challenger"
    ]
    expected_selected = max(
        challenger_lightgbm_records,
        key=lambda record: record["average_precision"],
    )["imbalance_strategy"]
    assert metrics_payload["selected_strategy"] == expected_selected
    assert study_calls == [
        {
            "train_rows": 60 if expected_selected == "undersampled" else 120,
            "train_target_rows": 60 if expected_selected == "undersampled" else 120,
            "validation_rows": 60,
            "validation_target_rows": 60,
            "n_trials": 1,
            "random_seed": 43,
            "scale_pos_weight": 3.0 if expected_selected == "weighted" else 1.0,
        }
    ]

    numeric_columns = load_feature_dictionary()["challenger"]["numeric"]
    annual_inc_index = numeric_columns.index("annual_inc")
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_[annual_inc_index] == pytest.approx(
        train["annual_inc"].median()
    )
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    purpose_categories = categorical_encoder.categories_[0]
    assert "validation_only_purpose" not in purpose_categories
    assert "calibration_only_purpose" not in purpose_categories

    challenger_columns = (
        load_feature_dictionary()["challenger"]["numeric"]
        + load_feature_dictionary()["challenger"]["categorical"]
    )
    transformed_validation = preprocessor.transform(validation.loc[:, challenger_columns])
    probabilities = model.predict_proba(transformed_validation)[:, 1]
    assert probabilities.shape == (len(validation),)
    assert np.isfinite(probabilities).all()
    transformed_calibration = preprocessor.transform(calibration.loc[:, challenger_columns])
    calibrated_probabilities = calibrated_model.predict_proba(transformed_calibration)[:, 1]
    assert calibrated_probabilities.shape == (len(calibration),)
    assert np.isfinite(calibrated_probabilities).all()
    if selected_method == "uncalibrated":
        assert brier_score_loss(calibration["bad"], calibrated_probabilities) == pytest.approx(
            calibration_metrics["methods"][selected_method]["brier_score"]
        )
    else:
        assert isinstance(calibrated_model, CalibratedClassifierCV)
        assert calibrated_model.method == selected_method
        assert isinstance(calibrated_model.estimator, FrozenEstimator)

    expected_calibration_evaluation = evaluate_calibration(
        model,
        transformed_calibration,
        calibration["bad"].to_numpy(),
        methods=config.calibration_methods,
        random_seed=config.random_seed,
    )
    expected_policy = search_policy(
        calibration["bad"].to_numpy(),
        calibration["loan_amnt"].to_numpy(),
        expected_calibration_evaluation.probabilities[selected_method],
        lgd=costs.base.lgd,
        margin=costs.base.margin,
        review_cost=costs.base.review_cost,
    ).iloc[0]
    assert policy == {
        "approve_below": expected_policy["approve_below"],
        "decline_at": expected_policy["decline_at"],
        "lgd": costs.base.lgd,
        "margin": costs.base.margin,
        "review_cost": costs.base.review_cost,
        "calibration_cost": expected_policy["cost"],
        "calibration_approval_rate": expected_policy["approval_rate"],
        "calibration_review_rate": expected_policy["review_rate"],
        "calibration_decline_rate": expected_policy["decline_rate"],
        "calibration_samples": len(calibration),
        "total_loan_amount": pytest.approx(calibration["loan_amnt"].sum()),
        "currency": "USD",
        "calibration_cost_per_1000_applications": pytest.approx(
            expected_policy["cost"] / len(calibration) * 1000
        ),
        "selected_calibration_method": selected_method,
        "probability_source": expected_calibration_evaluation.metrics[selected_method][
            "probability_source"
        ],
        "selection_partition": "calibration",
        "threshold_selection_protocol": "grid_search_on_calibration_evaluation_probabilities",
        "calibration_evaluation_protocol": expected_calibration_evaluation.evaluation_protocol,
    }
    assert cost_sensitivity.columns.tolist() == [
        "lgd",
        "margin",
        "review_cost",
        "is_base_scenario",
        "optimal_approve_below",
        "optimal_decline_at",
        "optimal_cost",
        "optimal_cost_per_1000_applications",
        "optimal_approval_rate",
        "optimal_review_rate",
        "optimal_decline_rate",
        "base_approve_below",
        "base_decline_at",
        "frozen_base_cost",
        "frozen_base_cost_per_1000_applications",
        "frozen_base_approval_rate",
        "frozen_base_review_rate",
        "frozen_base_decline_rate",
    ]
    assert len(cost_sensitivity) == 27
    assert cost_sensitivity["is_base_scenario"].sum() == 1
    assert set(
        cost_sensitivity[["lgd", "margin", "review_cost"]].itertuples(index=False, name=None)
    ) == {
        (lgd, margin, review_cost)
        for lgd in costs.lgd_values
        for margin in costs.margin_values
        for review_cost in costs.review_cost_values
    }
    assert np.allclose(
        cost_sensitivity[
            ["optimal_approval_rate", "optimal_review_rate", "optimal_decline_rate"]
        ].sum(axis=1),
        1.0,
    )
    assert np.allclose(
        cost_sensitivity[
            [
                "frozen_base_approval_rate",
                "frozen_base_review_rate",
                "frozen_base_decline_rate",
            ]
        ].sum(axis=1),
        1.0,
    )
    base_scenario = cost_sensitivity[
        (cost_sensitivity["lgd"] == costs.base.lgd)
        & (cost_sensitivity["margin"] == costs.base.margin)
        & (cost_sensitivity["review_cost"] == costs.base.review_cost)
    ].iloc[0]
    assert base_scenario["optimal_approve_below"] == pytest.approx(policy["approve_below"])
    assert base_scenario["optimal_decline_at"] == pytest.approx(policy["decline_at"])
    assert base_scenario["optimal_cost"] == pytest.approx(policy["calibration_cost"])
    assert base_scenario["frozen_base_cost"] == pytest.approx(policy["calibration_cost"])
    assert np.allclose(
        cost_sensitivity["optimal_cost_per_1000_applications"],
        cost_sensitivity["optimal_cost"] / len(calibration) * 1000,
    )
    assert np.allclose(
        cost_sensitivity["frozen_base_cost_per_1000_applications"],
        cost_sensitivity["frozen_base_cost"] / len(calibration) * 1000,
    )
    assert "0.39999999999999997" not in policy_text
    assert "0.39999999999999997" not in sensitivity_text
    assert set(trials.columns) >= {
        "number",
        "value",
        "state",
        *(f"params_{name}" for name in TUNED_PARAMETER_NAMES),
    }
    assert len(trials) == 1
    for path, original_bytes in input_bytes.items():
        assert path.read_bytes() == original_bytes


def test_train_script_help_runs_from_another_working_directory(tmp_path: Path) -> None:
    script_path = Path("scripts/train.py").resolve()

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--n-trials" in result.stdout


def test_train_main_defaults_to_thirty_trials() -> None:
    train_script = _load_train_script("train_default_trials")

    assert inspect.signature(train_script.main).parameters["n_trials"].default == 30
