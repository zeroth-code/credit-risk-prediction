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
from sklearn.datasets import make_classification

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


def test_train_main_writes_reproducible_uncalibrated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_script = _load_train_script("train_integration")
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    processed_dir.mkdir()
    train = _synthetic_partition(rows=120, positives=30, seed=101)
    validation = _synthetic_partition(rows=60, positives=20, seed=103)
    validation.loc[0, "purpose"] = "validation_only_purpose"
    validation.loc[0, "annual_inc"] = 10_000_000.0
    validation.loc[1, "annual_inc"] = np.nan
    train_path = processed_dir / "train.parquet"
    validation_path = processed_dir / "validation.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    input_bytes = {
        train_path: train_path.read_bytes(),
        validation_path: validation_path.read_bytes(),
    }
    config = SimpleNamespace(
        random_seed=43,
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
    )
    monkeypatch.setattr(train_script, "load_config", lambda path: config)

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
    }
    assert {path.name for path in artifact_dir.iterdir()} == expected_artifacts
    model = joblib.load(artifact_dir / "uncalibrated_model.joblib")
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    metrics_payload = json.loads(
        (artifact_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    trials = pd.read_csv(artifact_dir / "tuning_trials.csv")

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

    challenger_columns = (
        load_feature_dictionary()["challenger"]["numeric"]
        + load_feature_dictionary()["challenger"]["categorical"]
    )
    transformed_validation = preprocessor.transform(validation.loc[:, challenger_columns])
    probabilities = model.predict_proba(transformed_validation)[:, 1]
    assert probabilities.shape == (len(validation),)
    assert np.isfinite(probabilities).all()
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
