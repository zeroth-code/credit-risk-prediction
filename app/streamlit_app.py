"""Read-only Streamlit demonstration for the frozen credit-risk release."""

import html
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_risk.artifacts import ReleaseManifest, validate_release_bundle  # noqa: E402
from credit_risk.costs import assign_actions  # noqa: E402
from credit_risk.features import feature_columns  # noqa: E402
from credit_risk.schemas import CreditApplication, CreditPrediction  # noqa: E402

DEFAULT_RELEASE_DIR = PROJECT_ROOT / "artifacts/release"
FEATURE_DICTIONARY_PATH = PROJECT_ROOT / "configs/features.yaml"
PAGE_TITLE = "Credit Risk Decision Lab"
PAGE_LAYOUT = "wide"
DEMONSTRATION_WARNING = "Demonstration only - not a lending decision system. Inputs are not stored."
ASSOCIATION_LABEL = "Associations, not causal effects"
EVIDENCE_TABS = (
    "Model Performance",
    "Calibration",
    "Business Cost",
    "Fairness",
    "Limitations",
)
ALLOWED_POLICY_PROVENANCE = {
    "selected_calibration_method": {"uncalibrated", "sigmoid", "isotonic"},
    "probability_source": {"base_model_calibration_partition", "stratified_oof"},
    "selection_partition": {"calibration"},
    "threshold_selection_protocol": {"grid_search_on_calibration_evaluation_probabilities"},
    "calibration_evaluation_protocol": {"stratified_oof", "base_model_holdout_only"},
}
FAIRNESS_FILES = {
    "income": "fairness_income.csv",
    "home_ownership": "fairness_home_ownership.csv",
    "region": "fairness_region.csv",
    "employment": "fairness_employment.csv",
}


class StartupError(RuntimeError):
    """Raised when the frozen release cannot be trusted or loaded."""


class PredictionError(RuntimeError):
    """Raised when a validated application cannot be scored safely."""


@dataclass(frozen=True)
class Policy:
    approve_below: float
    decline_at: float
    lgd: float
    margin: float
    review_cost: float
    currency: str
    selected_calibration_method: str
    probability_source: str
    selection_partition: str
    threshold_selection_protocol: str
    calibration_evaluation_protocol: str


@dataclass(frozen=True)
class ReleaseArtifacts:
    release_dir: Path
    manifest: ReleaseManifest
    preprocessor: object
    model: object
    policy: Policy
    validation_metrics: dict[str, Any]
    calibration_metrics: dict[str, Any]
    calibration_curve: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    final_test_metrics: dict[str, Any]
    confusion_matrix: pd.DataFrame
    policy_test_results: dict[str, Any]
    temporal_metrics: pd.DataFrame
    fairness_tables: dict[str, pd.DataFrame]
    fairness_summary: dict[str, Any]
    shap_importance: pd.DataFrame
    shap_explanations: dict[str, Any]


@dataclass(frozen=True)
class ExplanationView:
    label: str
    context: str
    table: pd.DataFrame


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _finite_policy_number(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"policy field {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy field {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"policy field {field} is outside its allowed range")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"policy field {field} is outside its allowed range")
    return parsed


def _policy_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy field {field} must be a non-empty string")
    return value.strip()


def _parse_policy(payload: dict[str, Any]) -> Policy:
    approve_below = _finite_policy_number(payload, "approve_below", minimum=0.0, maximum=1.0)
    decline_at = _finite_policy_number(payload, "decline_at", minimum=0.0, maximum=1.0)
    if approve_below >= decline_at:
        raise ValueError("policy thresholds must satisfy approve_below < decline_at")
    currency = _policy_string(payload, "currency")
    if currency != "USD":
        raise ValueError("policy currency must be USD")
    provenance = {field: _policy_string(payload, field) for field in ALLOWED_POLICY_PROVENANCE}
    for field, allowed_values in ALLOWED_POLICY_PROVENANCE.items():
        if provenance[field] not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ValueError(f"policy field {field} must be one of: {allowed}")
    method = provenance["selected_calibration_method"]
    probability_source = provenance["probability_source"]
    calibration_protocol = provenance["calibration_evaluation_protocol"]
    if method in {"sigmoid", "isotonic"} and (
        probability_source != "stratified_oof" or calibration_protocol != "stratified_oof"
    ):
        raise ValueError(
            "policy calibrated methods require stratified_oof probability source and protocol"
        )
    if method == "uncalibrated" and probability_source != "base_model_calibration_partition":
        raise ValueError(
            "policy uncalibrated method requires base_model_calibration_partition probabilities"
        )
    return Policy(
        approve_below=approve_below,
        decline_at=decline_at,
        lgd=_finite_policy_number(payload, "lgd", minimum=0.0, maximum=1.0),
        margin=_finite_policy_number(payload, "margin", minimum=0.0, maximum=1.0),
        review_cost=_finite_policy_number(payload, "review_cost", minimum=0.0),
        currency=currency,
        selected_calibration_method=provenance["selected_calibration_method"],
        probability_source=provenance["probability_source"],
        selection_partition=provenance["selection_partition"],
        threshold_selection_protocol=provenance["threshold_selection_protocol"],
        calibration_evaluation_protocol=provenance["calibration_evaluation_protocol"],
    )


def load_release_artifacts(release_dir: str | Path = DEFAULT_RELEASE_DIR) -> ReleaseArtifacts:
    release_path = Path(release_dir)
    try:
        manifest = validate_release_bundle(release_path)
        if manifest.feature_set != "challenger":
            raise ValueError("release feature_set must be challenger")

        policy = _parse_policy(_load_json_object(release_path / manifest.policy_file))
        preprocessor = joblib.load(release_path / manifest.preprocessor_file)
        model = joblib.load(release_path / manifest.model_file)
        fairness_tables = {
            name: pd.read_csv(release_path / filename) for name, filename in FAIRNESS_FILES.items()
        }
        return ReleaseArtifacts(
            release_dir=release_path,
            manifest=manifest,
            preprocessor=preprocessor,
            model=model,
            policy=policy,
            validation_metrics=_load_json_object(release_path / "validation_metrics.json"),
            calibration_metrics=_load_json_object(release_path / "calibration_metrics.json"),
            calibration_curve=pd.read_csv(release_path / "calibration_curve.csv"),
            cost_sensitivity=pd.read_csv(release_path / "cost_sensitivity.csv"),
            final_test_metrics=_load_json_object(release_path / "final_test_metrics.json"),
            confusion_matrix=pd.read_csv(release_path / "confusion_matrix.csv"),
            policy_test_results=_load_json_object(release_path / "policy_test_results.json"),
            temporal_metrics=pd.read_csv(release_path / "temporal_metrics.csv"),
            fairness_tables=fairness_tables,
            fairness_summary=_load_json_object(release_path / "fairness_summary.json"),
            shap_importance=pd.read_csv(release_path / "shap_importance.csv"),
            shap_explanations=_load_json_object(release_path / "shap_explanations.json"),
        )
    except Exception as exc:
        if isinstance(exc, StartupError):
            raise
        raise StartupError(f"release bundle is unavailable or inconsistent: {exc}") from exc


def _bad_class_probability(model: object, matrix: object) -> float:
    if not hasattr(model, "classes_"):
        raise ValueError("calibrated model must provide classes_")
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("calibrated model classes_ must contain two binary labels")
    if np.issubdtype(classes.dtype, np.bool_) or (
        classes.dtype == object and any(isinstance(value, bool) for value in classes)
    ):
        raise ValueError("calibrated model classes_ must contain labels 0 and 1")
    try:
        has_missing = bool(pd.isna(classes).any())
        is_binary = bool(np.isin(classes, [0, 1]).all())
        unique_classes = np.unique(classes)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model classes_ must contain labels 0 and 1") from exc
    if has_missing or not is_binary or len(unique_classes) != 2:
        raise ValueError("calibrated model classes_ must contain unique labels 0 and 1")
    bad_indices = np.flatnonzero(classes == 1)
    if len(bad_indices) != 1:
        raise ValueError("calibrated model classes_ has an ambiguous bad class")

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError("calibrated model must provide predict_proba")
    try:
        probabilities = np.asarray(predict_proba(matrix), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibrated model probabilities must be numeric") from exc
    if probabilities.shape != (1, 2):
        raise ValueError("calibrated model predict_proba must return shape (1, 2)")
    if not np.isfinite(probabilities).all():
        raise ValueError("calibrated model probabilities must be finite")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("calibrated model probabilities must be between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("calibrated model probabilities must sum to 1")
    return float(probabilities[0, int(bad_indices[0])])


def _release_example_explanation(
    artifacts: ReleaseArtifacts,
    action: str,
) -> list[tuple[str, float]]:
    local_explanations = artifacts.shap_explanations.get("local_explanations")
    if isinstance(local_explanations, dict):
        example = local_explanations.get(action)
        if isinstance(example, dict):
            contributions = example.get("top_contributions")
            if isinstance(contributions, list):
                rows: list[tuple[str, float]] = []
                for contribution in contributions:
                    if not isinstance(contribution, dict):
                        continue
                    feature = contribution.get("feature")
                    shap_value = contribution.get("shap_value")
                    if (
                        isinstance(feature, str)
                        and feature.strip()
                        and not isinstance(shap_value, bool)
                    ):
                        try:
                            parsed_value = float(shap_value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(parsed_value):
                            rows.append((feature.strip(), parsed_value))
                if rows:
                    return rows

    rows = []
    required_columns = {"feature", "mean_abs_shap"}
    if required_columns.issubset(artifacts.shap_importance.columns):
        for row in artifacts.shap_importance.loc[:, ["feature", "mean_abs_shap"]].itertuples(
            index=False
        ):
            if isinstance(row.feature, str) and math.isfinite(float(row.mean_abs_shap)):
                rows.append((row.feature.strip(), float(row.mean_abs_shap)))
    if not rows:
        raise ValueError("release explanation payload contains no usable associations")
    return rows[:5]


def predict_application(
    values: Mapping[str, object],
    artifacts: ReleaseArtifacts,
) -> CreditPrediction:
    application = CreditApplication.model_validate(dict(values))
    try:
        numeric_columns, categorical_columns = feature_columns(
            artifacts.manifest.feature_set,
            path=FEATURE_DICTIONARY_PATH,
        )
        selected_columns = [*numeric_columns, *categorical_columns]
        frame = pd.DataFrame([application.model_dump()], columns=selected_columns)
        transform = getattr(artifacts.preprocessor, "transform", None)
        if not callable(transform):
            raise ValueError("frozen preprocessor must provide transform")
        transformed = transform(frame)
        transformed_shape = getattr(transformed, "shape", None)
        if transformed_shape is None or transformed_shape[0] != 1:
            raise ValueError("preprocessor transform must return one scored row")
        probability = _bad_class_probability(artifacts.model, transformed)
        actions = assign_actions(
            [probability],
            approve_below=artifacts.policy.approve_below,
            decline_at=artifacts.policy.decline_at,
        )
        if len(actions) != 1:
            raise ValueError("policy assignment must return exactly one action")
        action = str(actions[0])
        explanation = _release_example_explanation(artifacts, action)
        return CreditPrediction(
            default_probability=float(probability),
            action=action,
            explanation=[(feature, float(value)) for feature, value in explanation],
        )
    except Exception as exc:
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError(f"credit prediction failed: {exc}") from exc


def explanation_view(
    prediction: CreditPrediction,
    _artifacts: ReleaseArtifacts,
) -> ExplanationView:
    return ExplanationView(
        label=ASSOCIATION_LABEL,
        context=(
            "Frozen release example for the assigned action; these values were not generated "
            "for the entered application."
        ),
        table=pd.DataFrame(prediction.explanation, columns=["Feature", "Association"]),
    )


@st.cache_resource(show_spinner=False)
def cached_release_artifacts(release_dir: str) -> ReleaseArtifacts:
    return load_release_artifacts(Path(release_dir))


def _release_directory() -> Path:
    override = os.environ.get("CREDIT_RISK_RELEASE_DIR")
    return Path(override) if override else DEFAULT_RELEASE_DIR


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cr-white: #FFFFFF;
            --cr-band: #F6F8FA;
            --cr-text: #17202A;
            --cr-muted: #5F6B76;
            --cr-border: #D7DEE4;
            --cr-teal: #078A8C;
            --cr-approve: #2E9B55;
            --cr-review: #D99A00;
            --cr-decline: #D64545;
        }
        html, body, [class*="st-"] {
            color: var(--cr-text);
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
            letter-spacing: 0;
        }
        .stApp { background: var(--cr-white); }
        [data-testid="stHeader"] { display: none; }
        [data-testid="stDeployButton"] { display: none; }
        .block-container {
            max-width: 1536px;
            padding: 1.25rem 1.35rem 3rem;
        }
        h1 {
            color: var(--cr-text);
            font-size: 2rem !important;
            line-height: 1.15 !important;
            font-weight: 720 !important;
            margin: 0 !important;
        }
        h2, h3, h4 { color: var(--cr-text); }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        p, label, input, textarea, button, [role="tab"] {
            font-size: 0.93rem !important;
        }
        [data-testid="stAlert"] {
            position: sticky;
            top: 3.1rem;
            z-index: 20;
            border: 1px solid #E7C66A;
            border-radius: 4px;
            box-shadow: none;
        }
        [data-testid="stForm"] {
            border: 0;
            border-radius: 0;
            padding: 0;
        }
        [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 0.25rem; }
        [data-testid="stForm"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0;
            line-height: 1.05;
        }
        [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
            font-size: 0.82rem !important;
        }
        [data-baseweb="input"], [data-baseweb="select"] > div {
            border-radius: 4px !important;
        }
        [data-testid="stForm"] [data-baseweb="input"],
        [data-testid="stForm"] [data-baseweb="select"] > div {
            height: 2.1rem !important;
            min-height: 2.25rem;
        }
        [data-testid="stFormSubmitButton"] button {
            min-height: 2.7rem;
            width: 100%;
            border: 1px solid var(--cr-teal) !important;
            border-radius: 4px;
            background: var(--cr-teal) !important;
            color: white !important;
            font-weight: 680;
            box-shadow: none;
        }
        [data-testid="stFormSubmitButton"] p { color: white; }
        [data-testid="stFormSubmitButton"] button:focus {
            box-shadow: 0 0 0 2px rgba(7, 138, 140, 0.22) !important;
        }
        [data-testid="stFormSubmitButton"] button:disabled { opacity: 0.62; }
        [data-testid="stFormSubmitButton"] button:hover {
            border-color: #056F71 !important;
            background: #056F71 !important;
            color: white !important;
        }
        .release-state {
            color: var(--cr-muted);
            padding-top: 0.35rem;
            text-align: right;
        }
        .release-state strong { color: var(--cr-teal); font-weight: 700; }
        .workflow-heading {
            border-bottom: 1px solid var(--cr-border);
            padding-bottom: 0.55rem;
            margin-bottom: 0.75rem;
        }
        .result-placeholder {
            min-height: 15rem;
            display: grid;
            align-content: center;
            background: var(--cr-band);
            border: 1px solid var(--cr-border);
            border-radius: 4px;
            padding: 1.25rem;
            color: var(--cr-muted);
        }
        .decision-summary {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            border-top: 1px solid var(--cr-border);
            border-bottom: 1px solid var(--cr-border);
            margin-bottom: 1rem;
        }
        .decision-cell { padding: 1rem 0.75rem 1rem 0; }
        .decision-cell + .decision-cell {
            border-left: 1px solid var(--cr-border);
            padding-left: 1.25rem;
        }
        .decision-label { color: var(--cr-muted); font-size: 0.82rem; }
        .decision-value {
            color: var(--cr-teal);
            font-size: 2.3rem;
            font-weight: 740;
            line-height: 1.15;
            margin-top: 0.35rem;
        }
        .decision-value.approve { color: var(--cr-approve); }
        .decision-value.manual_review { color: var(--cr-review); }
        .decision-value.decline { color: var(--cr-decline); }
        .risk-track {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 3px;
            margin: 0.55rem 0 0.35rem;
        }
        .risk-track span { display: block; height: 0.65rem; border-radius: 3px; }
        .risk-track .approve { background: var(--cr-approve); }
        .risk-track .review { background: var(--cr-review); }
        .risk-track .decline { background: var(--cr-decline); }
        .thresholds {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem 1rem;
            color: var(--cr-muted);
            margin-bottom: 1rem;
        }
        .thresholds span { white-space: normal; }
        .thresholds i {
            display: inline-block;
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 50%;
            margin-right: 0.35rem;
        }
        .thresholds .approve i { background: var(--cr-approve); }
        .thresholds .review i { background: var(--cr-review); }
        .thresholds .decline i { background: var(--cr-decline); }
        [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: thin;
            border-bottom: 1px solid var(--cr-border);
        }
        [data-baseweb="tab"] {
            white-space: nowrap;
            min-height: 2.8rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-baseweb="tab-highlight"] { background-color: var(--cr-teal) !important; }
        [aria-selected="true"][role="tab"] { color: var(--cr-teal) !important; }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--cr-border);
            border-radius: 4px;
        }
        hr { border-color: var(--cr-border); }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 0.9rem 2rem; }
            h1 { font-size: 1.7rem !important; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            .release-state { text-align: left; padding: 0 0 0.25rem; }
            .decision-value { font-size: 1.9rem; }
            .decision-cell { padding-right: 0.5rem; }
            .decision-cell + .decision-cell { padding-left: 0.8rem; }
            [data-baseweb="tab"] { padding-left: 0.75rem; padding-right: 0.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_form() -> tuple[bool, dict[str, object]]:
    st.markdown(
        "<div class='workflow-heading'><h2>Application inputs</h2></div>", unsafe_allow_html=True
    )
    with st.form("credit_application", clear_on_submit=False, border=False):
        left, right = st.columns(2, gap="medium")
        with left:
            loan_amnt = st.number_input(
                "Loan amount (USD)", min_value=1.0, max_value=100_000.0, value=25_000.0
            )
            annual_inc = st.number_input("Annual income (USD)", min_value=1.0, value=85_000.0)
            dti = st.number_input(
                "Debt-to-income ratio (%)", min_value=0.0, max_value=100.0, value=28.0
            )
            fico_range_low = st.number_input(
                "FICO range low", min_value=300.0, max_value=850.0, value=680.0
            )
            fico_range_high = st.number_input(
                "FICO range high", min_value=300.0, max_value=850.0, value=720.0
            )
            delinq_2yrs = st.number_input(
                "Delinquencies (2 years)", min_value=0.0, value=1.0, step=1.0
            )
            inq_last_6mths = st.number_input(
                "Inquiries (6 months)", min_value=0.0, value=2.0, step=1.0
            )
            open_acc = st.number_input("Open accounts", min_value=0.0, value=8.0, step=1.0)
        with right:
            pub_rec = st.number_input("Public records", min_value=0.0, value=0.0, step=1.0)
            revol_bal = st.number_input("Revolving balance (USD)", min_value=0.0, value=6_200.0)
            revol_util = st.number_input(
                "Revolving utilization (%)", min_value=0.0, max_value=200.0, value=32.0
            )
            total_acc = st.number_input("Total accounts", min_value=0.0, value=18.0, step=1.0)
            purpose = st.selectbox(
                "Purpose",
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "major_purchase",
                    "medical",
                    "small_business",
                    "other",
                ],
            )
            home_ownership = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
            verification_status = st.selectbox(
                "Verification status", ["Verified", "Source Verified", "Not Verified"]
            )
            emp_length = st.selectbox(
                "Employment length",
                [
                    "5+ years",
                    "10+ years",
                    "2 years",
                    "1 year",
                    "< 1 year",
                    "n/a",
                ],
            )
            addr_state = st.text_input("State (two-letter code)", value="TX", max_chars=2)
        submitted = st.form_submit_button("Run assessment", width="stretch")
    return submitted, {
        "loan_amnt": float(loan_amnt),
        "annual_inc": float(annual_inc),
        "dti": float(dti),
        "delinq_2yrs": float(delinq_2yrs),
        "fico_range_low": float(fico_range_low),
        "fico_range_high": float(fico_range_high),
        "inq_last_6mths": float(inq_last_6mths),
        "open_acc": float(open_acc),
        "pub_rec": float(pub_rec),
        "revol_bal": float(revol_bal),
        "revol_util": float(revol_util),
        "total_acc": float(total_acc),
        "purpose": purpose,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "emp_length": emp_length,
        "addr_state": addr_state,
    }


def _display_action(action: str) -> str:
    return {
        "approve": "Approve",
        "manual_review": "Manual review",
        "decline": "Decline",
    }[action]


def _render_result(
    prediction: CreditPrediction | None,
    artifacts: ReleaseArtifacts,
    error_message: str | None,
) -> None:
    st.markdown(
        "<div class='workflow-heading'><h2>Decision result</h2></div>", unsafe_allow_html=True
    )
    if error_message is not None:
        st.error(error_message)
        return
    if prediction is None:
        st.markdown(
            "<div class='result-placeholder'>Run the synthetic application to view the "
            "calibrated probability, frozen policy action, and release explanation.</div>",
            unsafe_allow_html=True,
        )
        return

    action_label = _display_action(prediction.action)
    probability_text = f"{prediction.default_probability:.2%}"
    st.markdown(
        f"""
        <div class="decision-summary">
          <div class="decision-cell">
            <div class="decision-label">Calibrated default probability (PD)</div>
            <div class="decision-value">{probability_text}</div>
          </div>
          <div class="decision-cell">
            <div class="decision-label">Frozen policy action</div>
            <div class="decision-value {prediction.action}">{html.escape(action_label)}</div>
          </div>
        </div>
        <div class="decision-label">Risk scale (low to high)</div>
        <div class="risk-track" aria-label="risk scale">
          <span class="approve"></span><span class="review"></span><span class="decline"></span>
        </div>
        <div class="thresholds">
          <span class="approve"><i></i>Approve below {artifacts.policy.approve_below:.2%}</span>
          <span class="review"><i></i>Review from {artifacts.policy.approve_below:.2%}
            to {artifacts.policy.decline_at:.2%}</span>
          <span class="decline"><i></i>Decline at or above {artifacts.policy.decline_at:.2%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    explanation = explanation_view(prediction, artifacts)
    st.markdown(f"#### {explanation.label}")
    st.caption(explanation.context)
    st.dataframe(
        explanation.table,
        hide_index=True,
        width="stretch",
        column_config={
            "Association": st.column_config.NumberColumn(format="%+.4f"),
        },
    )


def _metric_rows(metrics: dict[str, Any]) -> pd.DataFrame:
    predictive = metrics.get("predictive_metrics")
    if not isinstance(predictive, dict):
        predictive = {}
    confidence = metrics.get("confidence_intervals")
    if not isinstance(confidence, dict):
        confidence = {}
    labels = {
        "roc_auc": "ROC AUC",
        "average_precision": "Average precision",
        "ks": "KS statistic",
        "precision": "Precision",
        "recall": "Recall",
        "brier_score": "Brier score",
        "log_loss": "Log loss",
    }
    rows = []
    for key, label in labels.items():
        value = predictive.get(key)
        interval = confidence.get(key)
        note = "Unavailable"
        if isinstance(interval, dict):
            lower = interval.get("lower")
            upper = interval.get("upper")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                note = f"95% CI {float(lower):.3f} to {float(upper):.3f}"
        rows.append(
            {
                "Metric": label,
                "Value": f"{float(value):.3f}"
                if isinstance(value, (int, float))
                else "Unavailable",
                "Confidence / Notes": note,
            }
        )
    return pd.DataFrame(rows)


def _render_line_chart(frame: pd.DataFrame, *, y_title: str) -> bool:
    colors = ["#078A8C", "#D99A00", "#5F6B76", "#D64545"]
    figure = go.Figure()
    for index, column in enumerate(frame.columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            figure.add_trace(
                go.Scatter(
                    x=frame.index.astype(str),
                    y=values,
                    mode="lines+markers",
                    name=str(column).replace("_", " ").title(),
                    line={"color": colors[index % len(colors)], "width": 2},
                    marker={"size": 6},
                )
            )
    if not figure.data:
        return False
    figure.update_layout(
        height=300,
        margin={"l": 44, "r": 20, "t": 22, "b": 44},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#17202A", "size": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": None, "gridcolor": "#E8EDF1", "zeroline": False},
        yaxis={"title": y_title, "gridcolor": "#E8EDF1", "zeroline": False},
        hovermode="x unified",
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    return True


def _render_model_performance(artifacts: ReleaseArtifacts) -> None:
    st.subheader("Final out-of-time performance")
    st.dataframe(_metric_rows(artifacts.final_test_metrics), hide_index=True, width="stretch")
    st.caption(
        f"Evaluation sample: {artifacts.final_test_metrics.get('test_samples', 'Unavailable')}"
    )
    st.markdown("#### Confusion matrix")
    st.dataframe(artifacts.confusion_matrix, hide_index=True, width="stretch")
    st.markdown("#### Temporal evidence")
    temporal = artifacts.temporal_metrics.copy()
    chart_columns = [
        column
        for column in ("roc_auc", "brier_score", "expected_calibration_error")
        if column in temporal.columns
    ]
    if "month" in temporal.columns and chart_columns:
        chart = (
            temporal.set_index("month").loc[:, chart_columns].apply(pd.to_numeric, errors="coerce")
        )
        _render_line_chart(chart, y_title="Metric value")
    else:
        st.info("Temporal metrics are unavailable in this release.")
    st.dataframe(temporal, hide_index=True, width="stretch")
    with st.expander("Validation evidence"):
        st.json(artifacts.validation_metrics, expanded=False)


def _render_calibration(artifacts: ReleaseArtifacts) -> None:
    st.subheader("Calibration evidence")
    selected = artifacts.calibration_metrics.get("selected_method", "Unavailable")
    ece = artifacts.final_test_metrics.get("expected_calibration_error")
    first, second = st.columns(2)
    first.metric("Selected method", str(selected))
    second.metric(
        "Expected calibration error",
        f"{float(ece):.3f}" if isinstance(ece, (int, float)) else "Unavailable",
    )
    curve = artifacts.calibration_curve.copy()
    required = {"mean_probability", "observed_default_rate"}
    if required.issubset(curve.columns):
        chart = curve.loc[:, ["mean_probability", "observed_default_rate"]].apply(
            pd.to_numeric, errors="coerce"
        )
        _render_line_chart(chart, y_title="Default rate")
    else:
        st.info("Calibration curve values are unavailable in this release.")
    st.dataframe(curve, hide_index=True, width="stretch")
    with st.expander("Calibration metadata"):
        st.json(artifacts.calibration_metrics, expanded=False)


def _render_business_cost(artifacts: ReleaseArtifacts) -> None:
    st.subheader("Frozen policy and cost sensitivity")
    policy_results = artifacts.policy_test_results
    rows = [
        ("Approve below", f"{artifacts.policy.approve_below:.2%}"),
        ("Decline at", f"{artifacts.policy.decline_at:.2%}"),
        ("LGD", f"{artifacts.policy.lgd:.2%}"),
        ("Margin", f"{artifacts.policy.margin:.2%}"),
        ("Review cost", f"{artifacts.policy.currency} {artifacts.policy.review_cost:,.2f}"),
        (
            "Test cost / 1,000 applications",
            (
                f"{artifacts.policy.currency} "
                f"{float(policy_results['test_cost_per_1000_applications']):,.2f}"
                if isinstance(policy_results.get("test_cost_per_1000_applications"), (int, float))
                else "Unavailable"
            ),
        ),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Policy evidence", "Value"]),
        hide_index=True,
        width="stretch",
    )
    sensitivity = artifacts.cost_sensitivity.copy()
    if (
        "review_cost" in sensitivity.columns
        and "optimal_cost_per_1000_applications" in sensitivity.columns
    ):
        chart = (
            sensitivity.loc[:, ["review_cost", "optimal_cost_per_1000_applications"]]
            .apply(pd.to_numeric, errors="coerce")
            .set_index("review_cost")
        )
        _render_line_chart(chart, y_title="Cost per 1,000 applications")
    else:
        st.info("Cost sensitivity chart values are unavailable in this release.")
    st.dataframe(sensitivity, hide_index=True, width="stretch")


def _render_fairness(artifacts: ReleaseArtifacts) -> None:
    st.subheader("Fairness diagnostics")
    limitations = artifacts.fairness_summary.get("limitations")
    if isinstance(limitations, str) and limitations.strip():
        st.info(limitations.strip())
    st.caption(
        "Minimum displayed group size: "
        f"{artifacts.fairness_summary.get('minimum_group_size', 'Unavailable')}"
    )
    labels = {
        "income": "Income bands",
        "home_ownership": "Home ownership",
        "region": "Region",
        "employment": "Employment length",
    }
    for name, table in artifacts.fairness_tables.items():
        st.markdown(f"#### {labels[name]}")
        st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Fairness summary payload"):
        st.json(artifacts.fairness_summary, expanded=False)


def _render_limitations(artifacts: ReleaseArtifacts) -> None:
    st.subheader("Known limitations")
    st.markdown(
        "- Historical LendingClub data reflects prior underwriting and selection, not the "
        "full applicant population.\n"
        "- Model outputs and SHAP values describe statistical associations, not causal "
        "effects.\n"
        "- The release does not establish statutory fair-lending compliance or production "
        "monitoring.\n"
        "- Operational use would require independent validation, governance, human "
        "oversight, and ongoing drift review."
    )
    note = artifacts.shap_explanations.get("explanation_model")
    if isinstance(note, dict) and isinstance(note.get("calibration_note"), str):
        st.info(str(note["calibration_note"]))
    st.markdown("#### Global release associations")
    if artifacts.shap_importance.empty:
        st.info("Global association importance is unavailable in this release.")
    else:
        st.dataframe(artifacts.shap_importance, hide_index=True, width="stretch")


def _render_evidence(artifacts: ReleaseArtifacts) -> None:
    tabs = st.tabs(EVIDENCE_TABS)
    with tabs[0]:
        _render_model_performance(artifacts)
    with tabs[1]:
        _render_calibration(artifacts)
    with tabs[2]:
        _render_business_cost(artifacts)
    with tabs[3]:
        _render_fairness(artifacts)
    with tabs[4]:
        _render_limitations(artifacts)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT, initial_sidebar_state="collapsed")
    _inject_css()
    try:
        artifacts = cached_release_artifacts(str(_release_directory()))
    except StartupError as exc:
        st.error(f"Cannot start the demonstration: {exc}")
        st.stop()

    header_title, header_status = st.columns([0.72, 0.28], vertical_alignment="center")
    with header_title:
        st.title(PAGE_TITLE)
    with header_status:
        st.markdown(
            "<div class='release-state'><strong>Release bundle verified</strong><br>"
            f"Version {html.escape(artifacts.manifest.version)}</div>",
            unsafe_allow_html=True,
        )
    st.warning(DEMONSTRATION_WARNING)

    input_column, result_column = st.columns([0.48, 0.52], gap="large")
    with input_column:
        submitted, values = _render_form()

    prediction: CreditPrediction | None = None
    error_message: str | None = None
    if submitted:
        try:
            prediction = predict_application(values, artifacts)
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            field = ".".join(str(item) for item in first_error["loc"])
            error_message = f"Invalid {field}: {first_error['msg']}"
        except PredictionError:
            error_message = "Assessment could not be completed from the validated release."

    with result_column:
        _render_result(prediction, artifacts, error_message)

    st.divider()
    _render_evidence(artifacts)


if __name__ == "__main__":
    main()
