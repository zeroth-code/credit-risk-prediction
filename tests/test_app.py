import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from credit_risk import demo
from credit_risk.artifacts import create_release_bundle
from credit_risk.features import feature_columns, make_tree_preprocessor
from credit_risk.schemas import CreditPrediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app/streamlit_app.py"
DEMO_PATH = PROJECT_ROOT / "src/credit_risk/demo.py"


def _load_app_module() -> ModuleType:
    module_name = "credit_risk_streamlit_app_test"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Streamlit app from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


streamlit_app = _load_app_module()


class RecordingPreprocessor:
    def __init__(
        self,
        *,
        transformed: object | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.transform_calls: list[pd.DataFrame] = []
        self.transformed = (
            np.array([[1.0, 2.0]]) if transformed is None else np.asarray(transformed)
        )
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("the application must never fit the frozen preprocessor")

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.transform_calls.append(frame.copy(deep=True))
        return self.transformed.copy()


class FixedProbabilityModel:
    def __init__(
        self,
        *,
        classes: object = (0, 1),
        probabilities: tuple[float, float] = (0.8, 0.2),
        n_features: int | None = None,
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray([probabilities], dtype=float)
        self.predict_calls: list[object] = []
        if n_features is not None:
            self.n_features_in_ = n_features

    def predict_proba(self, matrix: object) -> np.ndarray:
        self.predict_calls.append(matrix)
        return self.probabilities.copy()


class MissingClassesModel:
    def predict_proba(self, _matrix: object) -> np.ndarray:
        return np.array([[0.8, 0.2]])


class MissingPredictProbaModel:
    classes_ = np.array([0, 1])


class MissingTransformPreprocessor:
    pass


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _production_cost_sensitivity_frame() -> pd.DataFrame:
    rows = []
    for lgd in (0.4, 0.6, 0.8):
        for margin in (0.03, 0.05, 0.07):
            for review_cost in (10.0, 30.0, 50.0):
                base_cost = 500.0 + lgd * 1_000.0 + margin * 2_000.0
                rows.append(
                    {
                        "lgd": lgd,
                        "margin": margin,
                        "review_cost": review_cost,
                        "is_base_scenario": lgd == 0.6 and margin == 0.05 and review_cost == 30.0,
                        "optimal_approve_below": 0.25,
                        "optimal_decline_at": 0.65,
                        "optimal_cost": base_cost + review_cost,
                        "optimal_cost_per_1000_applications": base_cost + review_cost,
                        "optimal_approval_rate": 0.5,
                        "optimal_review_rate": 0.3,
                        "optimal_decline_rate": 0.2,
                        "base_approve_below": 0.25,
                        "base_decline_at": 0.65,
                        "frozen_base_cost": base_cost + review_cost * 1.5,
                        "frozen_base_cost_per_1000_applications": base_cost + review_cost * 1.5,
                        "frozen_base_approval_rate": 0.5,
                        "frozen_base_review_rate": 0.3,
                        "frozen_base_decline_rate": 0.2,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _production_shap_explanations_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "explanation_model": {
            "artifact": "uncalibrated_model.joblib",
            "source": "frozen_uncalibrated_lightgbm",
            "objective": "binary",
            "sigmoid": 1.0,
            "output_space": "raw_model_output",
            "units": "log_odds",
            "calibrated_probability_source": "frozen_calibrated_model",
            "calibration_note": (
                "SHAP values explain the frozen base LightGBM score, not the "
                "post-calibration probability."
            ),
        },
        "local_explanations": {
            action: {
                "policy_action": action,
                "scored_index": index,
                "row_identifier": None,
                "calibrated_probability": probability,
                "base_value": 0.1,
                "base_model_raw_output": 0.1 + contribution,
                "top_contributions": [
                    {
                        "feature": "numeric__dti",
                        "feature_value": 28.0,
                        "shap_value": contribution,
                    }
                ],
                "waterfall": f"shap_waterfall_{action}.png",
            }
            for index, (action, probability, contribution) in enumerate(
                (
                    ("approve", 0.12, -0.4),
                    ("manual_review", 0.45, 0.2),
                    ("decline", 0.78, 0.6),
                )
            )
        },
        "files": {
            "importance": "shap_importance.csv",
            "payload": "shap_explanations.json",
            "beeswarm": "shap_beeswarm.png",
            "dependence": [],
            "waterfalls": {
                action: f"shap_waterfall_{action}.png"
                for action in ("approve", "manual_review", "decline")
            },
        },
    }


def _create_release(
    tmp_path: Path,
    *,
    model: object | None = None,
    preprocessor: object | None = None,
    policy_overrides: dict[str, object] | None = None,
    validation_metrics_overrides: dict[str, object] | None = None,
    calibration_metrics_overrides: dict[str, object] | None = None,
    final_metrics_overrides: dict[str, object] | None = None,
    policy_results_overrides: dict[str, object] | None = None,
    shap_explanations_overrides: dict[str, object] | None = None,
) -> Path:
    source_dir = tmp_path / "artifacts"
    source_dir.mkdir()
    joblib.dump(model or FixedProbabilityModel(), source_dir / "calibrated_model.joblib")
    joblib.dump(preprocessor or RecordingPreprocessor(), source_dir / "preprocessor.joblib")
    policy = {
        "approve_below": 0.25,
        "decline_at": 0.65,
        "lgd": 0.6,
        "margin": 0.05,
        "review_cost": 30.0,
        "currency": "USD",
        "selected_calibration_method": "sigmoid",
        "probability_source": "stratified_oof",
        "selection_partition": "calibration",
        "threshold_selection_protocol": "grid_search_on_calibration_evaluation_probabilities",
        "calibration_evaluation_protocol": "stratified_oof",
    }
    policy.update(policy_overrides or {})
    _write_json(source_dir / "policy.json", policy)
    validation_metrics = {"primary_feature_set": "challenger"}
    validation_metrics.update(validation_metrics_overrides or {})
    _write_json(source_dir / "validation_metrics.json", validation_metrics)
    calibration_metrics = {
        "selected_method": policy["selected_calibration_method"],
        "evaluation_protocol": policy["calibration_evaluation_protocol"],
        "evaluation_partition": "calibration",
        "artifact": {"method": policy["selected_calibration_method"]},
        "methods": {
            str(policy["selected_calibration_method"]): {
                "status": "ok",
                "probability_source": policy["probability_source"],
            }
        },
    }
    calibration_metrics.update(calibration_metrics_overrides or {})
    _write_json(source_dir / "calibration_metrics.json", calibration_metrics)
    final_metrics = {
        "test_samples": 120,
        "predictive_metrics": {"roc_auc": 0.74, "brier_score": 0.08},
        "expected_calibration_error": 0.03,
        "confidence_intervals": {
            "roc_auc": {"lower": 0.70, "upper": 0.78},
            "brier_score": {"lower": 0.06, "upper": 0.10},
        },
        "model_provenance": {
            "feature_set": "challenger",
            "evaluation_partition": "test",
            "preprocessor_artifact": "preprocessor.joblib",
            "model_artifact": "calibrated_model.joblib",
            "test_scoring_probability_source": "frozen_calibrated_model",
        },
        "policy_provenance": {
            "policy_artifact": "policy.json",
            "approve_below": policy["approve_below"],
            "decline_at": policy["decline_at"],
            "selected_calibration_method": policy["selected_calibration_method"],
            "threshold_selection_probability_source": policy["probability_source"],
            "calibration_evaluation_protocol": policy["calibration_evaluation_protocol"],
            "selection_partition": policy["selection_partition"],
            "threshold_selection_protocol": policy["threshold_selection_protocol"],
        },
    }
    final_metrics.update(final_metrics_overrides or {})
    _write_json(source_dir / "final_test_metrics.json", final_metrics)
    policy_results = {
        "approve_below": policy["approve_below"],
        "decline_at": policy["decline_at"],
        "lgd": policy["lgd"],
        "margin": policy["margin"],
        "review_cost": policy["review_cost"],
        "currency": policy["currency"],
        "selected_calibration_method": policy["selected_calibration_method"],
        "threshold_selection_probability_source": policy["probability_source"],
        "selection_partition": policy["selection_partition"],
        "threshold_selection_protocol": policy["threshold_selection_protocol"],
        "calibration_evaluation_protocol": policy["calibration_evaluation_protocol"],
        "test_scoring_probability_source": "frozen_calibrated_model",
        "test_cost_per_1000_applications": 1500.0,
        "test_approval_rate": 0.55,
        "test_review_rate": 0.25,
        "test_decline_rate": 0.20,
    }
    policy_results.update(policy_results_overrides or {})
    _write_json(source_dir / "policy_test_results.json", policy_results)
    _write_json(
        source_dir / "fairness_summary.json",
        {
            "schema_version": "1.0",
            "minimum_group_size": 20,
            "limitations": "This is not a statutory fair-lending audit.",
            "attributes": {},
        },
    )
    shap_explanations = _production_shap_explanations_payload()
    shap_explanations.update(shap_explanations_overrides or {})
    _write_json(source_dir / "shap_explanations.json", shap_explanations)
    (source_dir / "calibration_curve.csv").write_text(
        "method,bin_index,mean_probability,observed_default_rate,sample_count\n"
        "sigmoid,0,0.10,0.12,50\n",
        encoding="utf-8",
    )
    _production_cost_sensitivity_frame().to_csv(
        source_dir / "cost_sensitivity.csv", index=False, encoding="utf-8"
    )
    (source_dir / "confusion_matrix.csv").write_text(
        "actual_label,predicted_label,count\n0,0,70\n0,1,10\n1,0,20\n1,1,20\n",
        encoding="utf-8",
    )
    (source_dir / "temporal_metrics.csv").write_text(
        "month,count,roc_auc,brier_score,expected_calibration_error\n"
        "2025-01,60,0.72,0.09,0.04\n2025-02,60,0.76,0.07,0.02\n",
        encoding="utf-8",
    )
    fairness_csv = (
        "group,count,bad_rate,selection_rate,true_positive_rate,false_positive_rate,"
        "roc_auc,brier_score,suppressed\nA,60,0.2,0.6,0.7,0.1,0.75,0.08,False\n"
    )
    for name in (
        "fairness_income.csv",
        "fairness_home_ownership.csv",
        "fairness_region.csv",
        "fairness_employment.csv",
    ):
        (source_dir / name).write_text(fairness_csv, encoding="utf-8")
    (source_dir / "shap_importance.csv").write_text(
        "rank,feature,mean_abs_shap\n1,numeric__dti,0.4\n",
        encoding="utf-8",
    )
    release_dir = source_dir / "release"
    create_release_bundle(
        source_dir,
        release_dir,
        version="test-1",
        feature_set="challenger",
        data_hash="b" * 64,
    )
    return release_dir


def _application_payload() -> dict[str, object]:
    return {
        "loan_amnt": 25_000.0,
        "annual_inc": 85_000.0,
        "dti": 28.0,
        "delinq_2yrs": 1.0,
        "fico_range_low": 680.0,
        "fico_range_high": 720.0,
        "inq_last_6mths": 2.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 6_200.0,
        "revol_util": 32.0,
        "total_acc": 18.0,
        "purpose": " debt_consolidation ",
        "home_ownership": " MORTGAGE ",
        "verification_status": " Verified ",
        "emp_length": " 5 years ",
        "addr_state": " tx ",
    }


def _snapshot_files(paths: list[Path]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths if path.is_file()}


def test_streamlit_app_module_exists() -> None:
    assert APP_PATH.is_file()


def test_threshold_caption_states_the_reported_threshold() -> None:
    caption = streamlit_app._threshold_caption({"classification_threshold": 0.05})

    assert "0.05" in caption
    assert "not at 0.5" in caption


def test_threshold_caption_flags_a_missing_threshold() -> None:
    assert "unrecorded" in streamlit_app._threshold_caption({})
    assert "unrecorded" in streamlit_app._threshold_caption({"classification_threshold": True})


def test_streamlit_script_imports_project_package_without_pytest_pythonpath() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path

project_root = Path({str(PROJECT_ROOT)!r})
src_dir = (project_root / 'src').resolve()
sys.path = [
    entry for entry in sys.path
    if not entry or Path(entry).resolve() != src_dir
]
spec = importlib.util.spec_from_file_location('clean_streamlit_import', {str(APP_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.PAGE_TITLE)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Credit Risk Decision Lab"


def test_css_contract_clears_header_and_keeps_desktop_workflow_compact() -> None:
    css_source = inspect.getsource(streamlit_app._inject_css)

    assert '[data-testid="stHeader"] { display: none; }' in css_source
    assert "padding: 1.25rem 1.35rem 3rem;" in css_source
    assert '[data-testid="stForm"] [data-testid="stVerticalBlock"]' in css_source
    assert "gap: 0.25rem;" in css_source
    assert "height: 2.1rem !important;" in css_source
    assert '[data-testid="stFormSubmitButton"] p { color: white; }' in css_source
    assert "background: var(--cr-teal) !important;" in css_source
    assert '[data-baseweb="tab-highlight"]' in css_source
    assert '[data-testid="stDeployButton"]' in css_source


def test_load_release_artifacts_validates_and_loads_typed_bundle(tmp_path: Path) -> None:
    release_dir = _create_release(tmp_path)

    artifacts = demo.load_release_artifacts(release_dir)

    assert isinstance(artifacts, demo.ReleaseArtifacts)
    assert artifacts.release_dir == release_dir
    assert artifacts.manifest.feature_set == "challenger"
    assert isinstance(artifacts.preprocessor, RecordingPreprocessor)
    assert isinstance(artifacts.model, FixedProbabilityModel)
    assert artifacts.policy.approve_below == pytest.approx(0.25)
    assert artifacts.final_test_metrics["test_samples"] == 120
    assert artifacts.fairness_tables["income"].loc[0, "group"] == "A"


def test_release_directory_rejects_noncanonical_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    release_alias = tmp_path / "release-alias"
    release_alias.symlink_to(release_dir, target_is_directory=True)
    monkeypatch.setenv("CREDIT_RISK_RELEASE_DIR", str(release_alias))

    with pytest.raises(demo.StartupError, match="canonical"):
        demo.release_directory()


def test_cache_identity_reloads_same_path_replacement_and_does_not_reuse_valid_bundle(
    tmp_path: Path,
) -> None:
    release_dir = _create_release(tmp_path)
    streamlit_app.cached_release_artifacts.clear()
    first_identity = demo.release_cache_identity(release_dir)

    first = streamlit_app.cached_release_artifacts(*first_identity)

    source_dir = release_dir.parent
    joblib.dump(MissingClassesModel(), source_dir / "calibrated_model.joblib")
    create_release_bundle(
        source_dir,
        release_dir,
        version="test-2",
        feature_set="challenger",
        data_hash="b" * 64,
    )
    second_identity = demo.release_cache_identity(release_dir)

    assert first.manifest.version == "test-1"
    assert second_identity != first_identity
    with pytest.raises(demo.StartupError, match="classes_"):
        streamlit_app.cached_release_artifacts(*second_identity)
    streamlit_app.cached_release_artifacts.clear()


def test_load_release_artifacts_deserializes_verified_file_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    real_joblib_load = joblib.load
    descriptors: list[int] = []

    def recording_joblib_load(input_file: object) -> object:
        assert not isinstance(input_file, (str, Path))
        assert callable(getattr(input_file, "fileno", None))
        assert input_file.tell() == 0  # type: ignore[attr-defined]
        descriptors.append(input_file.fileno())  # type: ignore[attr-defined]
        return real_joblib_load(input_file)

    monkeypatch.setattr(demo.joblib, "load", recording_joblib_load)

    artifacts = demo.load_release_artifacts(release_dir)

    assert isinstance(artifacts.preprocessor, RecordingPreprocessor)
    assert isinstance(artifacts.model, FixedProbabilityModel)
    assert len(descriptors) == 2


@pytest.mark.parametrize("replacement_kind", ["regular-file", "symlink"])
def test_load_release_artifacts_rejects_post_validation_artifact_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    release_dir = _create_release(tmp_path)
    replacement = tmp_path / "replacement-model.joblib"
    joblib.dump(MissingClassesModel(), replacement)
    real_joblib_load = joblib.load
    real_validate = demo.validate_release_bundle
    deserialization_attempts: list[object] = []

    def validate_then_replace(path: Path) -> object:
        manifest = real_validate(path)
        model_path = path / manifest.model_file
        if replacement_kind == "regular-file":
            replacement.replace(model_path)
        else:
            model_path.unlink()
            model_path.symlink_to(replacement)
        return manifest

    def forbidden_joblib_load(input_file: object) -> object:
        deserialization_attempts.append(input_file)
        if len(deserialization_attempts) == 1:
            return real_joblib_load(input_file)
        raise AssertionError("substituted model reached joblib deserialization")

    monkeypatch.setattr(demo, "validate_release_bundle", validate_then_replace)
    monkeypatch.setattr(demo.joblib, "load", forbidden_joblib_load)

    with pytest.raises(demo.StartupError, match="release bundle"):
        demo.load_release_artifacts(release_dir)
    assert len(deserialization_attempts) == 1


def test_load_release_artifacts_runs_strict_operational_scoring_probe(tmp_path: Path) -> None:
    release_dir = _create_release(tmp_path)

    artifacts = demo.load_release_artifacts(release_dir)

    assert len(artifacts.preprocessor.transform_calls) == 1
    probe_frame = artifacts.preprocessor.transform_calls[0]
    assert probe_frame.columns.tolist() == list(demo.SYNTHETIC_APPLICATION_VALUES)
    assert probe_frame.loc[0].to_dict() == demo.SYNTHETIC_APPLICATION_VALUES
    assert len(artifacts.model.predict_calls) == 1


def test_load_release_artifacts_rejects_feature_dictionary_order_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    numeric_columns, categorical_columns = feature_columns(
        "challenger",
        path=demo.FEATURE_DICTIONARY_PATH,
    )
    monkeypatch.setattr(
        demo,
        "feature_columns",
        lambda *_args, **_kwargs: (list(reversed(numeric_columns)), categorical_columns),
    )

    with pytest.raises(demo.StartupError, match="feature order"):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize(
    ("preprocessor", "model", "message"),
    [
        (
            RecordingPreprocessor(feature_names=list(reversed(_application_payload()))),
            FixedProbabilityModel(),
            "feature_names_in_",
        ),
        (
            RecordingPreprocessor(transformed=np.empty((1, 0))),
            FixedProbabilityModel(),
            "nonzero",
        ),
        (
            RecordingPreprocessor(),
            FixedProbabilityModel(n_features=3),
            "n_features_in_",
        ),
        (
            RecordingPreprocessor(),
            FixedProbabilityModel(probabilities=(np.nan, np.nan)),
            "finite",
        ),
    ],
    ids=["input-feature-order", "zero-width-transform", "model-width", "nonfinite-probability"],
)
def test_load_release_artifacts_rejects_nonoperational_scoring_components(
    tmp_path: Path,
    preprocessor: object,
    model: object,
    message: str,
) -> None:
    release_dir = _create_release(tmp_path, preprocessor=preprocessor, model=model)

    with pytest.raises(demo.StartupError, match=message):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_requires_probe_action_to_validate_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    monkeypatch.setattr(
        demo,
        "assign_actions",
        lambda *_args, **_kwargs: np.array(["unexpected_action"]),
    )

    with pytest.raises(demo.StartupError, match="action"):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_accepts_real_fitted_sklearn_components(tmp_path: Path) -> None:
    numeric_columns, categorical_columns = feature_columns(
        "challenger",
        path=demo.FEATURE_DICTIONARY_PATH,
    )
    training_rows = []
    for index in range(6):
        row = dict(demo.SYNTHETIC_APPLICATION_VALUES)
        row["loan_amnt"] = 10_000.0 + index * 2_500.0
        row["annual_inc"] = 45_000.0 + index * 8_000.0
        row["purpose"] = "credit_card" if index % 2 else "debt_consolidation"
        training_rows.append(row)
    training_frame = pd.DataFrame(
        training_rows,
        columns=[*numeric_columns, *categorical_columns],
    )
    labels = np.array([0, 1, 0, 1, 0, 1])
    preprocessor = make_tree_preprocessor(numeric_columns, categorical_columns)
    transformed = preprocessor.fit_transform(training_frame)
    model = LogisticRegression(random_state=0).fit(transformed, labels)
    release_dir = _create_release(tmp_path, preprocessor=preprocessor, model=model)

    artifacts = demo.load_release_artifacts(release_dir)

    assert artifacts.model.n_features_in_ == transformed.shape[1]


def test_load_release_artifacts_rejects_contradictory_policy_provenance(
    tmp_path: Path,
) -> None:
    release_dir = _create_release(
        tmp_path,
        policy_overrides={"probability_source": "base_model_calibration_partition"},
    )

    with pytest.raises(demo.StartupError, match="policy"):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize(
    ("release_kwargs", "message"),
    [
        ({"model": MissingClassesModel()}, "classes_"),
        ({"model": FixedProbabilityModel(classes=(0, 2))}, "classes_"),
        ({"model": FixedProbabilityModel(classes=(1, 1))}, "classes_"),
        (
            {"model": FixedProbabilityModel(classes=np.array([np.bool_(False), 1], dtype=object))},
            "classes_",
        ),
        ({"model": MissingPredictProbaModel()}, "predict_proba"),
        ({"preprocessor": MissingTransformPreprocessor()}, "transform"),
    ],
    ids=[
        "missing-classes",
        "non-binary-classes",
        "ambiguous-classes",
        "boolean-classes",
        "missing-predict-proba",
        "missing-transform",
    ],
)
def test_load_release_artifacts_rejects_invalid_frozen_components(
    tmp_path: Path,
    release_kwargs: dict[str, object],
    message: str,
) -> None:
    release_dir = _create_release(tmp_path, **release_kwargs)

    with pytest.raises(demo.StartupError, match=message):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize(
    ("release_kwargs", "message"),
    [
        (
            {"calibration_metrics_overrides": {"selected_method": "isotonic"}},
            "selected_method",
        ),
        (
            {"policy_results_overrides": {"decline_at": 0.75}},
            "policy_test_results.*decline_at",
        ),
        (
            {"validation_metrics_overrides": {"primary_feature_set": "full_underwriting"}},
            "primary_feature_set",
        ),
        (
            {
                "final_metrics_overrides": {
                    "model_provenance": {
                        "feature_set": "challenger",
                        "evaluation_partition": "test",
                        "preprocessor_artifact": "preprocessor.joblib",
                        "model_artifact": "different-model.joblib",
                        "test_scoring_probability_source": "frozen_calibrated_model",
                    }
                }
            },
            "model_provenance.*model_artifact",
        ),
    ],
    ids=["calibration-method", "policy-threshold", "validation-feature-set", "model-role"],
)
def test_load_release_artifacts_rejects_cross_artifact_contradictions(
    tmp_path: Path,
    release_kwargs: dict[str, object],
    message: str,
) -> None:
    release_dir = _create_release(tmp_path, **release_kwargs)

    with pytest.raises(demo.StartupError, match=message):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize(
    ("contradiction", "message"),
    [
        ("explanation-model", "explanation_model.*artifact"),
        ("missing-action", "local_explanations"),
        ("policy-action", "policy_action"),
        ("contribution-shape", "feature_value"),
    ],
    ids=["explanation-model", "missing-action", "policy-action", "contribution-shape"],
)
def test_load_release_artifacts_rejects_shap_contract_contradictions(
    tmp_path: Path,
    contradiction: str,
    message: str,
) -> None:
    payload = _production_shap_explanations_payload()
    explanation_model = payload["explanation_model"]
    local_explanations = payload["local_explanations"]
    assert isinstance(explanation_model, dict)
    assert isinstance(local_explanations, dict)
    if contradiction == "explanation-model":
        explanation_model["artifact"] = "calibrated_model.joblib"
    elif contradiction == "missing-action":
        del local_explanations["decline"]
    elif contradiction == "policy-action":
        local_explanations["approve"]["policy_action"] = "decline"
    else:
        del local_explanations["approve"]["top_contributions"][0]["feature_value"]
    release_dir = _create_release(
        tmp_path,
        shap_explanations_overrides=payload,
    )

    with pytest.raises(demo.StartupError, match=message):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_accepts_unavailable_local_action(tmp_path: Path) -> None:
    payload = _production_shap_explanations_payload()
    local_explanations = payload["local_explanations"]
    files = payload["files"]
    assert isinstance(local_explanations, dict)
    assert isinstance(files, dict)
    waterfalls = files["waterfalls"]
    assert isinstance(waterfalls, dict)
    local_explanations["decline"] = None
    del waterfalls["decline"]
    release_dir = _create_release(
        tmp_path,
        policy_results_overrides={
            "test_approval_rate": 0.55,
            "test_review_rate": 0.45,
            "test_decline_rate": 0.0,
        },
        shap_explanations_overrides=payload,
    )

    artifacts = demo.load_release_artifacts(release_dir)

    assert artifacts.shap_explanations["local_explanations"]["decline"] is None


def test_load_release_artifacts_rejects_string_policy_action_rate(tmp_path: Path) -> None:
    release_dir = _create_release(
        tmp_path,
        policy_results_overrides={"test_approval_rate": "0.55"},
    )

    with pytest.raises(demo.StartupError, match="test_approval_rate.*JSON numeric"):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize(
    ("approval_rate", "review_rate", "decline_rate"),
    [
        (0.2, 0.2, 0.0),
        (0.6, 0.6, 0.2),
    ],
    ids=["below-one", "above-one"],
)
def test_load_release_artifacts_rejects_policy_action_rates_not_summing_to_one(
    tmp_path: Path,
    approval_rate: float,
    review_rate: float,
    decline_rate: float,
) -> None:
    payload = _production_shap_explanations_payload()
    if decline_rate == 0.0:
        local_explanations = payload["local_explanations"]
        files = payload["files"]
        assert isinstance(local_explanations, dict)
        assert isinstance(files, dict)
        waterfalls = files["waterfalls"]
        assert isinstance(waterfalls, dict)
        local_explanations["decline"] = None
        del waterfalls["decline"]
    release_dir = _create_release(
        tmp_path,
        policy_results_overrides={
            "test_approval_rate": approval_rate,
            "test_review_rate": review_rate,
            "test_decline_rate": decline_rate,
        },
        shap_explanations_overrides=payload,
    )

    with pytest.raises(demo.StartupError, match="policy action rates.*sum to 1.0"):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_rejects_unavailable_observed_action(tmp_path: Path) -> None:
    payload = _production_shap_explanations_payload()
    local_explanations = payload["local_explanations"]
    files = payload["files"]
    assert isinstance(local_explanations, dict)
    assert isinstance(files, dict)
    waterfalls = files["waterfalls"]
    assert isinstance(waterfalls, dict)
    local_explanations["decline"] = None
    del waterfalls["decline"]
    release_dir = _create_release(
        tmp_path,
        shap_explanations_overrides=payload,
    )

    with pytest.raises(demo.StartupError, match="decline.*test_decline_rate"):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_rejects_example_for_unobserved_action(
    tmp_path: Path,
) -> None:
    release_dir = _create_release(
        tmp_path,
        policy_results_overrides={
            "test_approval_rate": 0.55,
            "test_review_rate": 0.45,
            "test_decline_rate": 0.0,
        },
    )

    with pytest.raises(demo.StartupError, match="decline.*test_decline_rate.*zero"):
        demo.load_release_artifacts(release_dir)


def test_load_release_artifacts_rejects_all_observed_actions_unavailable(
    tmp_path: Path,
) -> None:
    payload = _production_shap_explanations_payload()
    local_explanations = payload["local_explanations"]
    files = payload["files"]
    assert isinstance(local_explanations, dict)
    assert isinstance(files, dict)
    waterfalls = files["waterfalls"]
    assert isinstance(waterfalls, dict)
    for action in local_explanations:
        local_explanations[action] = None
    waterfalls.clear()
    release_dir = _create_release(
        tmp_path,
        shap_explanations_overrides=payload,
    )

    with pytest.raises(demo.StartupError, match="approve.*test_approval_rate"):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize("contradiction", ["missing", "stale"])
def test_load_release_artifacts_rejects_inconsistent_waterfall_mapping(
    tmp_path: Path,
    contradiction: str,
) -> None:
    payload = _production_shap_explanations_payload()
    local_explanations = payload["local_explanations"]
    files = payload["files"]
    assert isinstance(local_explanations, dict)
    assert isinstance(files, dict)
    waterfalls = files["waterfalls"]
    assert isinstance(waterfalls, dict)
    policy_results_overrides: dict[str, object] | None = None
    if contradiction == "missing":
        del waterfalls["decline"]
    else:
        local_explanations["decline"] = None
        policy_results_overrides = {
            "test_approval_rate": 0.55,
            "test_review_rate": 0.45,
            "test_decline_rate": 0.0,
        }
    release_dir = _create_release(
        tmp_path,
        policy_results_overrides=policy_results_overrides,
        shap_explanations_overrides=payload,
    )

    with pytest.raises(demo.StartupError, match="files.waterfalls"):
        demo.load_release_artifacts(release_dir)


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_load_release_artifacts_blocks_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    release_dir = tmp_path / "missing-release"
    if failure == "tampered":
        release_dir = _create_release(tmp_path)
        (release_dir / "policy.json").write_text("{}\n", encoding="utf-8")

    def forbidden_joblib_load(_path: Path) -> object:
        raise AssertionError("unsafe bundle reached joblib deserialization")

    monkeypatch.setattr(demo.joblib, "load", forbidden_joblib_load)

    with pytest.raises(demo.StartupError, match="release bundle"):
        demo.load_release_artifacts(release_dir)


def test_predict_application_uses_strict_schema_frozen_transform_and_bad_class(
    tmp_path: Path,
) -> None:
    release_dir = _create_release(
        tmp_path,
        model=FixedProbabilityModel(classes=(1, 0), probabilities=(0.2, 0.8)),
    )
    artifacts = demo.load_release_artifacts(release_dir)
    before = {path.name: path.read_bytes() for path in release_dir.iterdir()}

    prediction = demo.predict_application(_application_payload(), artifacts)

    assert isinstance(prediction, CreditPrediction)
    assert type(prediction.default_probability) is float
    assert prediction.default_probability == pytest.approx(0.2)
    assert prediction.action == "approve"
    assert prediction.explanation == [("numeric__dti", -0.4)]
    assert len(artifacts.preprocessor.transform_calls) == 2
    transformed_frame = artifacts.preprocessor.transform_calls[-1]
    assert transformed_frame.columns.tolist() == [
        "loan_amnt",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "fico_range_low",
        "fico_range_high",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "purpose",
        "home_ownership",
        "verification_status",
        "emp_length",
        "addr_state",
    ]
    assert transformed_frame.loc[0, "purpose"] == "debt_consolidation"
    assert transformed_frame.loc[0, "addr_state"] == "TX"
    assert {path.name: path.read_bytes() for path in release_dir.iterdir()} == before


def test_predict_application_constructs_strict_credit_application(tmp_path: Path) -> None:
    release_dir = _create_release(tmp_path)
    artifacts = demo.load_release_artifacts(release_dir)
    invalid_payload = _application_payload()
    invalid_payload["loan_amnt"] = "25000"

    with pytest.raises(ValidationError, match="loan_amnt"):
        demo.predict_application(invalid_payload, artifacts)


def test_predict_application_uses_frozen_policy_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(
        tmp_path,
        model=FixedProbabilityModel(probabilities=(0.75, 0.25)),
    )
    artifacts = demo.load_release_artifacts(release_dir)
    observed: dict[str, object] = {}

    def recording_assign_actions(
        probabilities: object,
        *,
        approve_below: float,
        decline_at: float,
    ) -> np.ndarray:
        observed.update(
            probabilities=np.asarray(probabilities).tolist(),
            approve_below=approve_below,
            decline_at=decline_at,
        )
        return np.array(["manual_review"])

    monkeypatch.setattr(demo, "assign_actions", recording_assign_actions)

    prediction = demo.predict_application(_application_payload(), artifacts)

    assert prediction.action == "manual_review"
    assert observed == {
        "probabilities": [0.25],
        "approve_below": 0.25,
        "decline_at": 0.65,
    }


def test_explanation_view_labels_release_example_as_association(tmp_path: Path) -> None:
    release_dir = _create_release(tmp_path)
    artifacts = demo.load_release_artifacts(release_dir)
    prediction = demo.predict_application(_application_payload(), artifacts)

    explanation = streamlit_app.explanation_view(prediction, artifacts)

    assert explanation.label == "Associations, not causal effects"
    assert explanation.source == "local_action_example"
    assert explanation.number_format == "%+.4f"
    assert "assigned action" in explanation.context.lower()
    assert "not generated for the entered application" in explanation.context.lower()
    assert "positive values increase" in explanation.context.lower()
    assert "negative values decrease" in explanation.context.lower()
    assert "base-model log-odds" in explanation.context.lower()
    assert "not calibrated probability" in explanation.context.lower()
    assert explanation.table.to_dict("records") == [
        {"Feature": "numeric__dti", "Directional association (SHAP value)": -0.4}
    ]


def test_explanation_view_labels_global_fallback_as_unsigned_not_local_or_action_specific(
    tmp_path: Path,
) -> None:
    release_dir = _create_release(tmp_path)
    artifacts = demo.load_release_artifacts(release_dir)
    artifacts = replace(
        artifacts,
        shap_explanations={"local_explanations": {}},
    )
    prediction = demo.predict_application(_application_payload(), artifacts)

    explanation = streamlit_app.explanation_view(prediction, artifacts)

    assert explanation.label == "Associations, not causal effects"
    assert explanation.source == "global_mean_absolute_importance"
    assert explanation.number_format == "%.4f"
    assert "global unsigned mean absolute" in explanation.context.lower()
    assert "not a local explanation" in explanation.context.lower()
    assert "not directional" in explanation.context.lower()
    assert "not action-specific" in explanation.context.lower()
    assert "sign is not retained" in explanation.context.lower()
    assert explanation.table.to_dict("records") == [
        {"Feature": "numeric__dti", "Mean absolute association (unsigned)": 0.4}
    ]


def test_calibration_chart_uses_mean_probability_for_x_and_observed_rate_for_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def capture_chart(figure: object, **_kwargs: object) -> None:
        captured.append(figure)

    monkeypatch.setattr(streamlit_app.st, "plotly_chart", capture_chart)
    curve = pd.DataFrame(
        {
            "method": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
            "mean_probability": [0.10, 0.35, np.nan, 0.80],
            "observed_default_rate": [0.12, 0.30, 0.55, np.inf],
        }
    )

    rendered = streamlit_app._render_calibration_chart(curve)

    assert rendered is True
    assert len(captured) == 1
    figure = captured[0]
    assert len(figure.data) == 2
    calibration_trace, reference_trace = figure.data
    assert calibration_trace.name == "Sigmoid"
    np.testing.assert_allclose(calibration_trace.x, [0.10, 0.35])
    np.testing.assert_allclose(calibration_trace.y, [0.12, 0.30])
    assert reference_trace.name == "Perfect calibration"
    np.testing.assert_allclose(reference_trace.x, [0.0, 1.0])
    np.testing.assert_allclose(reference_trace.y, [0.0, 1.0])
    assert reference_trace.line.dash == "dash"


def test_calibration_chart_reports_unavailable_when_no_finite_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_chart(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an unavailable calibration curve must not render a chart")

    monkeypatch.setattr(streamlit_app.st, "plotly_chart", forbidden_chart)
    curve = pd.DataFrame(
        {
            "mean_probability": [np.nan, "invalid"],
            "observed_default_rate": [0.2, np.inf],
        }
    )

    assert streamlit_app._render_calibration_chart(curve) is False


def test_business_cost_chart_groups_each_lgd_margin_scenario_without_cross_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def capture_chart(figure: object, **_kwargs: object) -> None:
        captured.append(figure)

    monkeypatch.setattr(streamlit_app.st, "plotly_chart", capture_chart)
    sensitivity = _production_cost_sensitivity_frame().sample(frac=1.0, random_state=42)

    rendered = streamlit_app._render_business_cost_sensitivity_chart(sensitivity)

    assert rendered is True
    assert len(captured) == 1
    figure = captured[0]
    assert len(figure.data) == 9
    expected_groups = {
        f"LGD {lgd:.0%} / Margin {margin:.0%}"
        for lgd in (0.4, 0.6, 0.8)
        for margin in (0.03, 0.05, 0.07)
    }
    assert {trace.name for trace in figure.data} == expected_groups
    for trace in figure.data:
        assert list(trace.x) == [10.0, 30.0, 50.0]
        assert len(trace.y) == 3


def test_business_cost_chart_reports_unavailable_for_missing_or_nonfinite_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_chart(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid cost sensitivity must not render a chart")

    monkeypatch.setattr(streamlit_app.st, "plotly_chart", forbidden_chart)
    missing = pd.DataFrame({"review_cost": [10.0], "optimal_cost_per_1000_applications": [1.0]})
    nonfinite = pd.DataFrame(
        {
            "lgd": [0.6],
            "margin": [0.05],
            "review_cost": [np.nan],
            "optimal_cost_per_1000_applications": [1_500.0],
        }
    )

    assert streamlit_app._render_business_cost_sensitivity_chart(missing) is False
    assert streamlit_app._render_business_cost_sensitivity_chart(nonfinite) is False


@pytest.mark.parametrize("source_path", [APP_PATH, DEMO_PATH], ids=["streamlit", "demo"])
def test_demo_uses_only_explicit_read_file_apis(source_path: Path) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    read_only_flag_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and "os.O_RDONLY" in ast.unparse(node.value)
    }
    prohibited_calls = {
        "dump",
        "set_query_params",
        "to_csv",
        "to_json",
        "to_parquet",
        "write_bytes",
        "write_text",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        assert node.func.attr not in prohibited_calls
        if node.func.attr != "open":
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
            assert len(node.args) >= 2
            flags = ast.unparse(node.args[1])
            assert "os.O_RDONLY" in flags or flags in read_only_flag_names
            assert "O_WRONLY" not in flags
            assert "O_RDWR" not in flags
            assert "O_CREAT" not in flags
            continue
        positional_mode = (
            node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
        )
        keyword_mode = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
        assert positional_mode in {"r", "rb"} or keyword_mode in {"r", "rb"}


def test_streamlit_nondefault_assessment_does_not_persist_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    sentinel = tmp_path / "fixture-sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    monkeypatch.setenv("CREDIT_RISK_RELEASE_DIR", str(release_dir))
    tracked_result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_paths = [PROJECT_ROOT / line for line in tracked_result.stdout.splitlines()]
    fixture_paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    tracked_before = _snapshot_files(tracked_paths)
    fixture_before = _snapshot_files(fixture_paths)
    status_before = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    nondefault_numbers = [
        12_345.0,
        91_000.0,
        17.5,
        705.0,
        745.0,
        0.0,
        1.0,
        11.0,
        2.0,
        8_765.0,
        48.0,
        24.0,
    ]
    for widget, value in zip(app.number_input, nondefault_numbers, strict=True):
        widget.set_value(value)
    for widget, value in zip(
        app.selectbox,
        ["home_improvement", "OWN", "Not Verified", "2 years"],
        strict=True,
    ):
        widget.set_value(value)
    app.text_input[0].set_value("CA")

    app = app.button[0].click().run(timeout=10)

    assert not app.exception
    assert app.query_params == {}
    rendered_text = "\n".join(element.value for element in app.markdown)
    assert "20.00%" in rendered_text
    assert _snapshot_files(tracked_paths) == tracked_before
    fixture_after_paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert _snapshot_files(fixture_after_paths) == fixture_before
    status_after = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status_after == status_before


def test_streamlit_startup_error_blocks_prediction_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREDIT_RISK_RELEASE_DIR", str(tmp_path / "missing-release"))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert len(app.error) == 1
    assert "release bundle" in app.error[0].value.lower()
    assert not app.button


def test_streamlit_untrusted_component_blocks_prediction_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path, model=MissingClassesModel())
    monkeypatch.setenv("CREDIT_RISK_RELEASE_DIR", str(release_dir))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert len(app.error) == 1
    assert "classes_" in app.error[0].value
    assert not app.button


def test_streamlit_workflow_renders_contract_and_assesses_synthetic_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _create_release(tmp_path)
    monkeypatch.setenv("CREDIT_RISK_RELEASE_DIR", str(release_dir))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Credit Risk Decision Lab"
    assert [warning.value for warning in app.warning] == [
        "Demonstration only - not a lending decision system. Inputs are not stored."
    ]
    assert [tab.label for tab in app.tabs] == [
        "Model Performance",
        "Calibration",
        "Business Cost",
        "Fairness",
        "Limitations",
    ]
    assert app.button[0].label == "Run assessment"
    assert len(app.number_input) == 12
    assert len(app.selectbox) == 4
    assert len(app.text_input) == 1
    assert len(app.get("plotly_chart")) == 3
    assert app.selectbox[0].options == [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "small_business",
        "car",
        "medical",
        "moving",
        "vacation",
        "house",
        "wedding",
        "renewable_energy",
        "educational",
    ]
    assert app.selectbox[3].options == [
        "10+ years",
        "9 years",
        "8 years",
        "7 years",
        "6 years",
        "5 years",
        "4 years",
        "3 years",
        "2 years",
        "1 year",
        "< 1 year",
        "n/a",
    ]
    initial_text = "\n".join(element.value for element in app.markdown)
    assert "Joblib uses Python pickle semantics" in initial_text
    assert "manifest hashes do not authenticate an untrusted bundle" in initial_text

    app = app.button[0].click().run(timeout=10)

    assert not app.exception
    rendered_text = "\n".join(element.value for element in app.markdown)
    assert "20.00%" in rendered_text
    assert "Approve" in rendered_text
    assert "Associations, not causal effects" in rendered_text
