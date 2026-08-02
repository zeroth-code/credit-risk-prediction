import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.calibration import expected_calibration_error as real_expected_calibration_error
from credit_risk.costs import assign_actions
from credit_risk.features import make_tree_preprocessor
from credit_risk.metrics import bootstrap_metric as real_bootstrap_metric

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT_PATH = PROJECT_ROOT / "scripts/evaluate.py"
OUTPUT_NAMES = {
    "final_test_metrics.json",
    "confusion_matrix.csv",
    "policy_test_results.json",
    "temporal_metrics.csv",
    "scored_test.parquet",
    "shap_importance.csv",
    "shap_explanations.json",
    "fairness_income.csv",
    "fairness_home_ownership.csv",
    "fairness_region.csv",
    "fairness_employment.csv",
    "fairness_summary.json",
}
SHAP_FIGURE_NAMES = {
    "shap_beeswarm.png",
    "shap_dependence_01.png",
    "shap_dependence_02.png",
    "shap_dependence_03.png",
    "shap_dependence_04.png",
    "shap_dependence_05.png",
    "shap_waterfall_approve.png",
    "shap_waterfall_manual_review.png",
    "shap_waterfall_decline.png",
}
PREDICTIVE_METRIC_KEYS = {
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
FINAL_METRICS_KEYS = {
    "test_samples",
    "prevalence",
    "predictive_metrics",
    "expected_calibration_error",
    "confidence_intervals",
    "classification_threshold",
    "bootstrap_methodology",
    "ece_methodology",
    "model_provenance",
    "policy_provenance",
}
POLICY_RESULT_KEYS = {
    "approve_below",
    "decline_at",
    "lgd",
    "margin",
    "review_cost",
    "selected_calibration_method",
    "threshold_selection_probability_source",
    "test_scoring_probability_source",
    "selection_partition",
    "threshold_selection_protocol",
    "calibration_evaluation_protocol",
    "test_samples",
    "total_exposure",
    "currency",
    "test_cost",
    "test_cost_per_1000_applications",
    "test_approval_rate",
    "test_review_rate",
    "test_decline_rate",
}
MODEL_PROVENANCE_KEYS = {
    "feature_set",
    "evaluation_partition",
    "preprocessor_artifact",
    "model_artifact",
    "test_scoring_probability_source",
}
POLICY_PROVENANCE_KEYS = {
    "policy_artifact",
    "approve_below",
    "decline_at",
    "selected_calibration_method",
    "threshold_selection_probability_source",
    "calibration_evaluation_protocol",
    "selection_partition",
    "threshold_selection_protocol",
}
TEMPORAL_COLUMNS = [
    "month",
    "status",
    "count",
    "prevalence",
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "approval_rate",
    "review_rate",
    "decline_rate",
    "policy_cost",
    "policy_cost_per_1000_applications",
    "total_exposure",
    "currency",
]
FAIRNESS_COLUMNS = [
    "group",
    "count",
    "bad_rate",
    "selection_rate",
    "true_positive_rate",
    "false_positive_rate",
    "roc_auc",
    "brier_score",
    "suppressed",
]


class FitForbiddenPreprocessor:
    def __init__(self, fitted: object, expected_columns: list[str]) -> None:
        self.fitted = fitted
        self.expected_columns = expected_columns

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation must not fit the preprocessor")

    def transform(self, frame: pd.DataFrame) -> object:
        assert frame.columns.tolist() == self.expected_columns
        return self.fitted.transform(frame)  # type: ignore[attr-defined,no-any-return]

    def get_feature_names_out(self) -> np.ndarray:
        return self.fitted.get_feature_names_out()  # type: ignore[attr-defined,no-any-return]


class FitForbiddenModel:
    def __init__(self, fitted: object) -> None:
        self.fitted = fitted
        self.classes_ = fitted.classes_  # type: ignore[attr-defined]

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation must not fit the calibrated model")

    def predict_proba(self, matrix: object) -> np.ndarray:
        return self.fitted.predict_proba(matrix)  # type: ignore[attr-defined,no-any-return]


class PredictProbaStub:
    def __init__(self, classes: object, probabilities: object) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, matrix: object) -> object:
        del matrix
        return self.probabilities


class MissingClassesStub:
    def predict_proba(self, matrix: object) -> np.ndarray:
        del matrix
        return np.array([[0.4, 0.6]])


def _load_evaluate_script(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, EVALUATE_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_test_environment(
    root: Path,
    *,
    forbid_fit: bool,
) -> tuple[Path, Path, Path, Path, pd.DataFrame, dict[str, object]]:
    processed_dir = root / "data/processed"
    artifact_dir = root / "artifacts"
    config_dir = root / "configs"
    processed_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    features_payload = {
        "challenger": {
            "numeric": [
                "income",
                "debt_ratio",
                "credit_history",
                "open_accounts",
                "utilization",
            ],
            "categorical": ["grade"],
        },
        "full_underwriting": {
            "numeric": [
                "income",
                "debt_ratio",
                "credit_history",
                "open_accounts",
                "utilization",
            ],
            "categorical": ["grade"],
        },
        "post_origination": ["recoveries"],
    }
    features_path = config_dir / "features.yaml"
    features_path.write_text(yaml.safe_dump(features_payload, sort_keys=False), encoding="utf-8")

    config_payload = {
        "random_seed": 17,
        "raw_csv": str(root / "data/raw/unused.csv"),
        "processed_dir": str(processed_dir),
        "artifact_dir": str(artifact_dir),
        "figure_dir": str(root / "reports/figures"),
        "train": {"start": "2017-01-01", "end": "2017-03-31"},
        "validation": {"start": "2017-04-01", "end": "2017-06-30"},
        "calibration": {"start": "2017-07-01", "end": "2017-09-30"},
        "test": {"start": "2017-10-01", "end": "2018-03-31"},
        "loan_term": "36 months",
        "good_statuses": ["Fully Paid"],
        "bad_statuses": ["Charged Off"],
        "unresolved_statuses": ["Current"],
        "calibration_methods": ["uncalibrated", "sigmoid", "isotonic"],
        "minimum_group_size": 2,
        "costs": {
            "base": {"lgd": 0.6, "margin": 0.05, "review_cost": 30.0},
            "lgd_values": [0.4, 0.6, 0.8],
            "margin_values": [0.03, 0.05, 0.07],
            "review_cost_values": [15.0, 30.0, 45.0],
        },
    }
    config_path = config_dir / "base.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    training = pd.DataFrame(
        {
            "income": np.linspace(20_000.0, 120_000.0, 40),
            "debt_ratio": np.linspace(0.1, 0.9, 40),
            "credit_history": np.linspace(1.0, 20.0, 40),
            "open_accounts": np.tile(np.arange(1.0, 6.0), 8),
            "utilization": np.linspace(0.95, 0.05, 40),
            "grade": ["A", "B"] * 20,
        }
    )
    training_target = np.array([0, 1] * 20)
    feature_columns = [
        "income",
        "debt_ratio",
        "credit_history",
        "open_accounts",
        "utilization",
        "grade",
    ]
    fitted_preprocessor = make_tree_preprocessor(feature_columns[:-1], ["grade"])
    training_matrix = fitted_preprocessor.fit_transform(training.loc[:, feature_columns])
    fitted_model = LogisticRegression(random_state=17).fit(training_matrix, training_target)
    fitted_uncalibrated_model = LGBMClassifier(
        n_estimators=8,
        num_leaves=5,
        min_child_samples=1,
        random_state=17,
        n_jobs=1,
        verbosity=-1,
    ).fit(training_matrix, training_target)
    preprocessor: object = fitted_preprocessor
    model: object = fitted_model
    if forbid_fit:
        preprocessor = FitForbiddenPreprocessor(fitted_preprocessor, feature_columns)
        model = FitForbiddenModel(fitted_model)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(model, artifact_dir / "calibrated_model.joblib")
    joblib.dump(fitted_uncalibrated_model, artifact_dir / "uncalibrated_model.joblib")

    test_frame = pd.DataFrame(
        {
            "id": [106, 101, 105, 102, 104, 103],
            "income": [25_000.0, 35_000.0, 90_000.0, 45_000.0, 80_000.0, 70_000.0],
            "debt_ratio": [0.15, 0.25, 0.8, 0.35, 0.7, 0.6],
            "credit_history": [2.0, 4.0, 18.0, 7.0, 15.0, 12.0],
            "open_accounts": [1.0, 2.0, 5.0, 3.0, 4.0, 3.0],
            "utilization": [0.9, 0.75, 0.1, 0.6, 0.2, 0.4],
            "grade": ["B", "A", "B", "A", "A", "B"],
            "bad": [0, 0, 1, 0, 1, 0],
            "issue_d": [
                "2018-02-10",
                "2018-01-05",
                "2018-02-01",
                "2018-01-20",
                "2018-02-15",
                "2018-02-28",
            ],
            "loan_amnt": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
            "annual_inc": [25_000.0, 35_000.0, 90_000.0, 45_000.0, 80_000.0, 70_000.0],
            "home_ownership": [" RENT ", "RENT", "MORTGAGE", "MORTGAGE", None, "OWN"],
            "addr_state": ["ny", " NY ", "ca", "CA", "tx", None],
            "emp_length": ["1 year", " 1 year ", "10+ years", "10+ years", "", None],
        }
    )
    test_probabilities = fitted_model.predict_proba(
        fitted_preprocessor.transform(test_frame.loc[:, feature_columns])
    )[:, 1]
    sorted_probabilities = np.sort(test_probabilities)

    policy_payload: dict[str, object] = {
        "approve_below": float(np.mean(sorted_probabilities[1:3])),
        "decline_at": float(np.mean(sorted_probabilities[3:5])),
        "lgd": 0.6,
        "margin": 0.05,
        "review_cost": 30.0,
        "calibration_cost": 100.0,
        "calibration_approval_rate": 0.4,
        "calibration_review_rate": 0.2,
        "calibration_decline_rate": 0.4,
        "calibration_samples": 100,
        "total_loan_amount": 1_000_000.0,
        "currency": "USD",
        "calibration_cost_per_1000_applications": 1000.0,
        "selected_calibration_method": "sigmoid",
        "probability_source": "stratified_oof",
        "selection_partition": "calibration",
        "threshold_selection_protocol": "grid_search_on_calibration_evaluation_probabilities",
        "calibration_evaluation_protocol": "stratified_oof",
    }
    policy_path = artifact_dir / "policy.json"
    policy_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    test_path = processed_dir / "test.parquet"
    test_frame.to_parquet(test_path, index=False)
    return config_path, features_path, artifact_dir, test_path, test_frame, policy_payload


def test_evaluate_uses_only_frozen_test_artifacts_and_writes_stable_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, features_path, artifact_dir, test_path, test_frame, policy = (
        _write_test_environment(tmp_path, forbid_fit=True)
    )
    input_paths = [
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "calibrated_model.joblib",
        artifact_dir / "uncalibrated_model.joblib",
        artifact_dir / "policy.json",
        test_path,
    ]
    original_bytes = {path: path.read_bytes() for path in input_paths}
    evaluate_script = _load_evaluate_script("evaluate_frozen_integration")
    source = inspect.getsource(evaluate_script)
    assert "scripts.train" not in source
    assert "search_policy" not in evaluate_script.__dict__

    original_read_parquet = pd.read_parquet
    read_paths: list[Path] = []

    def guarded_read_parquet(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        resolved = Path(path).resolve()
        read_paths.append(resolved)
        if resolved != test_path.resolve():
            raise AssertionError(f"evaluation read forbidden parquet: {resolved}")
        return original_read_parquet(path, *args, **kwargs)

    bootstrap_calls: list[tuple[str, int, int]] = []
    ece_calls: list[int] = []

    def recording_bootstrap(
        y_true: object,
        probabilities: object,
        *,
        metric_name: str,
        samples: int,
        random_seed: int,
    ) -> dict[str, float]:
        bootstrap_calls.append((metric_name, samples, random_seed))
        return real_bootstrap_metric(
            y_true,  # type: ignore[arg-type]
            probabilities,  # type: ignore[arg-type]
            metric_name=metric_name,
            samples=20,
            random_seed=random_seed,
        )

    def recording_ece(
        y_true: object,
        probabilities: object,
        *,
        bins: int,
    ) -> float:
        ece_calls.append(bins)
        return real_expected_calibration_error(y_true, probabilities, bins=bins)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)
    monkeypatch.setattr(evaluate_script, "bootstrap_metric", recording_bootstrap)
    monkeypatch.setattr(evaluate_script, "expected_calibration_error", recording_ece)

    def forbidden_lightgbm_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation must not fit the explanation model")

    monkeypatch.setattr(LGBMClassifier, "fit", forbidden_lightgbm_fit)

    evaluate_script.main(config_path=config_path, feature_dictionary_path=features_path)

    assert read_paths == [test_path.resolve()]
    assert bootstrap_calls == [
        ("roc_auc", 1000, 17),
        ("average_precision", 1000, 17),
        ("brier_score", 1000, 17),
    ]
    assert ece_calls == [10, 10, 10]
    assert all(path.read_bytes() == original_bytes[path] for path in input_paths)
    assert OUTPUT_NAMES.issubset(path.name for path in artifact_dir.iterdir())
    figure_dir = tmp_path / "reports/figures"
    assert {path.name for path in figure_dir.iterdir()} == SHAP_FIGURE_NAMES

    shap_importance = pd.read_csv(artifact_dir / "shap_importance.csv")
    assert shap_importance.columns.tolist() == ["rank", "feature", "mean_abs_shap"]
    assert len(shap_importance) == len(
        joblib.load(artifact_dir / "preprocessor.joblib").get_feature_names_out()
    )
    shap_payload = json.loads((artifact_dir / "shap_explanations.json").read_text())
    assert shap_payload["explanation_model"]["artifact"] == "uncalibrated_model.joblib"
    assert shap_payload["explanation_model"]["output_space"] == "raw_model_output"
    assert shap_payload["explanation_model"]["units"] == "log_odds"
    assert set(shap_payload["local_explanations"]) == {
        "approve",
        "manual_review",
        "decline",
    }

    final_metrics = json.loads((artifact_dir / "final_test_metrics.json").read_text())
    assert set(final_metrics) == FINAL_METRICS_KEYS
    assert set(final_metrics["predictive_metrics"]) == PREDICTIVE_METRIC_KEYS
    assert set(final_metrics["confidence_intervals"]) == {
        "roc_auc",
        "average_precision",
        "brier_score",
    }
    assert final_metrics["classification_threshold"] == 0.5
    assert final_metrics["test_samples"] == len(test_frame)
    assert final_metrics["bootstrap_methodology"] == {
        "samples": 1000,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "point_estimate_included": True,
        "resampling": "stratified_with_replacement",
        "random_seed": 17,
    }
    assert final_metrics["ece_methodology"] == {
        "bins": 10,
        "binning": "equal_width",
        "final_bin_inclusive": True,
    }
    assert set(final_metrics["model_provenance"]) == MODEL_PROVENANCE_KEYS
    assert set(final_metrics["policy_provenance"]) == POLICY_PROVENANCE_KEYS
    assert (
        final_metrics["model_provenance"]["test_scoring_probability_source"]
        == "frozen_calibrated_model"
    )
    assert (
        final_metrics["policy_provenance"]["threshold_selection_probability_source"]
        == policy["probability_source"]
    )

    confusion = pd.read_csv(artifact_dir / "confusion_matrix.csv")
    assert confusion.columns.tolist() == ["actual_label", "predicted_label", "count"]
    assert confusion[["actual_label", "predicted_label"]].to_records(index=False).tolist() == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    policy_results = json.loads((artifact_dir / "policy_test_results.json").read_text())
    assert set(policy_results) == POLICY_RESULT_KEYS
    assert policy_results["approve_below"] == policy["approve_below"]
    assert policy_results["decline_at"] == policy["decline_at"]
    assert policy_results["currency"] == "USD"
    assert policy_results["threshold_selection_probability_source"] == policy["probability_source"]
    assert policy_results["test_scoring_probability_source"] == "frozen_calibrated_model"
    assert "probability_source" not in policy_results

    temporal = pd.read_csv(artifact_dir / "temporal_metrics.csv")
    assert temporal.columns.tolist() == TEMPORAL_COLUMNS
    assert temporal["month"].tolist() == ["2018-01", "2018-02"]
    assert temporal["status"].tolist() == ["single_class_discrimination_undefined", "ok"]
    assert temporal.loc[0, ["roc_auc", "average_precision"]].isna().all()
    assert temporal.loc[0, ["brier_score", "log_loss", "policy_cost"]].notna().all()

    fairness_files = {
        "income": "fairness_income.csv",
        "home_ownership": "fairness_home_ownership.csv",
        "region": "fairness_region.csv",
        "employment": "fairness_employment.csv",
    }
    for output_name in fairness_files.values():
        fairness = pd.read_csv(artifact_dir / output_name)
        assert fairness.columns.tolist() == FAIRNESS_COLUMNS
        assert fairness["group"].tolist() == sorted(fairness["group"])
        suppressed = fairness["suppressed"]
        assert fairness.loc[suppressed, FAIRNESS_COLUMNS[2:-1]].isna().all().all()

    fairness_summary = json.loads((artifact_dir / "fairness_summary.json").read_text())
    assert fairness_summary["schema_version"] == "1.0"
    assert fairness_summary["minimum_group_size"] == 2
    assert fairness_summary["metric_semantics"]["favorable_ground_truth_outcome"] == (
        "good/repaid (1 - bad)"
    )
    assert fairness_summary["metric_semantics"]["favorable_decision"] == "action == approve"
    assert fairness_summary["metric_semantics"]["not_selected_actions"] == [
        "manual_review",
        "decline",
    ]
    assert "not a statutory fair-lending audit" in fairness_summary["limitations"]
    assert set(fairness_summary["attributes"]) == set(fairness_files)
    for name, output_name in fairness_files.items():
        attribute = fairness_summary["attributes"][name]
        assert attribute["output_file"] == output_name
        assert attribute["total_group_count"] == (
            attribute["evaluated_group_count"] + attribute["suppressed_group_count"]
        )
        for disparity in ("equal_opportunity_difference", "selection_rate_ratio"):
            assert set(attribute[disparity]) == {"status", "value", "reason"}

    scored = original_read_parquet(artifact_dir / "scored_test.parquet")
    assert scored.columns.tolist() == [
        *test_frame.columns,
        "default_probability",
        "predicted_bad",
        "action",
    ]
    pd.testing.assert_frame_equal(scored.loc[:, test_frame.columns], test_frame)
    expected_actions = assign_actions(
        scored["default_probability"].to_numpy(),
        approve_below=float(policy["approve_below"]),
        decline_at=float(policy["decline_at"]),
    )
    assert scored["action"].tolist() == expected_actions.tolist()


def test_evaluate_cli_help_resolves_from_another_working_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(EVALUATE_SCRIPT_PATH), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "frozen" in result.stdout.lower()


@pytest.mark.parametrize(
    "column",
    ["annual_inc", "home_ownership", "addr_state", "emp_length"],
)
def test_evaluate_requires_raw_fairness_grouping_columns(tmp_path: Path, column: str) -> None:
    config_path, features_path, _, test_path, test_frame, _ = _write_test_environment(
        tmp_path,
        forbid_fit=False,
    )
    test_frame.drop(columns=[column]).to_parquet(test_path, index=False)
    evaluate_script = _load_evaluate_script(f"evaluate_missing_fairness_{column}")

    with pytest.raises(ValueError, match=f"missing required columns: {column}"):
        evaluate_script.main(config_path=config_path, feature_dictionary_path=features_path)


def test_evaluate_rerun_is_deterministic(tmp_path: Path) -> None:
    config_path, features_path, artifact_dir, _, _, _ = _write_test_environment(
        tmp_path,
        forbid_fit=False,
    )
    evaluate_script = _load_evaluate_script("evaluate_deterministic")

    evaluate_script.main(config_path=config_path, feature_dictionary_path=features_path)
    first_text_outputs = {
        name: (artifact_dir / name).read_bytes()
        for name in OUTPUT_NAMES
        if not name.endswith(".parquet")
    }
    first_scored = pd.read_parquet(artifact_dir / "scored_test.parquet")

    evaluate_script.main(config_path=config_path, feature_dictionary_path=features_path)

    second_text_outputs = {
        name: (artifact_dir / name).read_bytes()
        for name in OUTPUT_NAMES
        if not name.endswith(".parquet")
    }
    second_scored = pd.read_parquet(artifact_dir / "scored_test.parquet")
    assert second_text_outputs == first_text_outputs
    pd.testing.assert_frame_equal(second_scored, first_scored)


def test_evaluate_cli_executes_from_another_working_directory(tmp_path: Path) -> None:
    environment_root = tmp_path / "environment"
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    config_path, features_path, artifact_dir, _, _, _ = _write_test_environment(
        environment_root,
        forbid_fit=False,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            str(EVALUATE_SCRIPT_PATH),
            "--config",
            str(config_path),
            "--features",
            str(features_path),
        ],
        cwd=other_cwd,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert OUTPUT_NAMES.issubset(path.name for path in artifact_dir.iterdir())


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("selection_partition", "test", "selection_partition.*calibration"),
        (
            "threshold_selection_protocol",
            "grid_search_on_test",
            "threshold_selection_protocol",
        ),
        ("selected_calibration_method", "beta", "selected_calibration_method"),
        (
            "calibration_evaluation_protocol",
            "test_refit",
            "calibration_evaluation_protocol",
        ),
        ("probability_source", "test_predictions", "probability_source"),
    ],
)
def test_evaluate_rejects_policy_with_non_frozen_provenance(
    tmp_path: Path,
    field: str,
    invalid_value: str,
    message: str,
) -> None:
    _, _, artifact_dir, _, _, policy = _write_test_environment(tmp_path, forbid_fit=False)
    policy[field] = invalid_value
    policy_path = artifact_dir / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    evaluate_script = _load_evaluate_script(f"evaluate_invalid_policy_{field}")

    with pytest.raises(ValueError, match=message):
        evaluate_script._load_policy(policy_path)


def test_validated_probabilities_uses_model_class_order_for_positive_class() -> None:
    evaluate_script = _load_evaluate_script("evaluate_reversed_classes")
    model = PredictProbaStub(
        np.array([1, 0]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )

    probabilities = evaluate_script._validated_probabilities(model, object(), 2)

    np.testing.assert_array_equal(probabilities, np.array([0.8, 0.1]))


def test_validated_probabilities_requires_model_classes() -> None:
    evaluate_script = _load_evaluate_script("evaluate_missing_classes")

    with pytest.raises(ValueError, match="classes_"):
        evaluate_script._validated_probabilities(MissingClassesStub(), object(), 1)


@pytest.mark.parametrize(
    "classes",
    [
        np.array([[0, 1]]),
        np.array([0]),
        np.array([0, 1, 2]),
        np.array([0, 0]),
        np.array([False, True]),
        np.array([0, pd.NA], dtype=object),
        np.array(["0", "1"]),
        np.array([0, 2]),
    ],
    ids=[
        "not-one-dimensional",
        "one-class",
        "three-classes",
        "duplicate",
        "boolean",
        "missing",
        "strings",
        "non-binary",
    ],
)
def test_validated_probabilities_rejects_invalid_model_classes(classes: object) -> None:
    evaluate_script = _load_evaluate_script("evaluate_invalid_classes")
    model = PredictProbaStub(classes, np.array([[0.4, 0.6]]))

    with pytest.raises(ValueError, match="classes_"):
        evaluate_script._validated_probabilities(model, object(), 1)


@pytest.mark.parametrize(
    ("probability_matrix", "message"),
    [
        (np.array([[np.nan, 0.6]]), "finite"),
        (np.array([[np.inf, 0.6]]), "finite"),
        (np.array([[-0.1, 0.6]]), "between 0 and 1"),
        (np.array([[1.1, 0.6]]), "between 0 and 1"),
        (np.array([[0.3, 0.6]]), "sum to 1"),
        (np.array([[pd.NA, 0.6]], dtype=object), "numeric"),
    ],
)
def test_validated_probabilities_rejects_invalid_full_probability_matrix(
    probability_matrix: np.ndarray,
    message: str,
) -> None:
    evaluate_script = _load_evaluate_script("evaluate_invalid_probability_matrix")
    model = PredictProbaStub(np.array([0, 1]), probability_matrix)

    with pytest.raises(ValueError, match=message):
        evaluate_script._validated_probabilities(model, object(), 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "selected_calibration_method": "sigmoid",
            "probability_source": "base_model_calibration_partition",
            "calibration_evaluation_protocol": "stratified_oof",
        },
        {
            "selected_calibration_method": "isotonic",
            "probability_source": "stratified_oof",
            "calibration_evaluation_protocol": "base_model_holdout_only",
        },
        {
            "selected_calibration_method": "uncalibrated",
            "probability_source": "stratified_oof",
            "calibration_evaluation_protocol": "stratified_oof",
        },
    ],
    ids=["sigmoid-source", "isotonic-protocol", "uncalibrated-source"],
)
def test_evaluate_rejects_contradictory_policy_provenance(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    _, _, artifact_dir, _, _, policy = _write_test_environment(tmp_path, forbid_fit=False)
    policy.update(overrides)
    policy_path = artifact_dir / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    evaluate_script = _load_evaluate_script("evaluate_contradictory_policy")

    with pytest.raises(
        ValueError,
        match=("selected_calibration_method.*probability_source.*calibration_evaluation_protocol"),
    ):
        evaluate_script._load_policy(policy_path)


@pytest.mark.parametrize(
    "issue_d",
    [
        pd.Series([202001, 202002]),
        pd.Series([202001.0, 202002.0]),
        pd.Series([True, False]),
        pd.Series(["2020-01-01", 202002], dtype=object),
        pd.Series(["2020-01-01", ""], dtype=object),
        pd.Series(["2020-01-01", "   "], dtype=object),
        pd.Series(["2020-01-01", None], dtype=object),
        pd.Series(["2020-01-01", pd.NA], dtype=object),
        pd.Series(["202001", "202002"], dtype=object),
    ],
    ids=[
        "integer",
        "float",
        "boolean",
        "mixed",
        "empty-string",
        "whitespace-string",
        "none",
        "pandas-na",
        "non-iso-string",
    ],
)
def test_validated_months_rejects_non_datetime_or_invalid_strings(issue_d: pd.Series) -> None:
    evaluate_script = _load_evaluate_script("evaluate_invalid_issue_dates")

    with pytest.raises(ValueError, match="issue_d"):
        evaluate_script._validated_months(pd.DataFrame({"issue_d": issue_d}))


@pytest.mark.parametrize(
    ("issue_d", "expected"),
    [
        (
            pd.Series(pd.to_datetime(["2020-02-29", "2020-01-01"])),
            ["2020-02", "2020-01"],
        ),
        (
            pd.Series(["2020-02-29", "2020-01-01"], dtype="string"),
            ["2020-02", "2020-01"],
        ),
    ],
    ids=["datetime-dtype", "iso-strings"],
)
def test_validated_months_accepts_supported_dates(
    issue_d: pd.Series,
    expected: list[str],
) -> None:
    evaluate_script = _load_evaluate_script("evaluate_valid_issue_dates")

    months = evaluate_script._validated_months(pd.DataFrame({"issue_d": issue_d}))

    assert months.tolist() == expected
