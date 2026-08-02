"""Read-only Streamlit demonstration for the frozen credit-risk release."""

import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_risk import demo  # noqa: E402
from credit_risk.schemas import CreditPrediction  # noqa: E402

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
PURPOSE_OPTIONS = (
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
)
EMPLOYMENT_OPTIONS = (
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
)


@dataclass(frozen=True)
class ExplanationView:
    label: str
    source: str
    context: str
    number_format: str
    table: pd.DataFrame


def explanation_view(
    prediction: CreditPrediction,
    artifacts: demo.ReleaseArtifacts,
) -> ExplanationView:
    evidence = demo.explanation_evidence(artifacts, prediction.action)
    return ExplanationView(
        label=ASSOCIATION_LABEL,
        source=evidence.source,
        context=evidence.context,
        number_format=evidence.number_format,
        table=pd.DataFrame(
            prediction.explanation,
            columns=["Feature", evidence.value_column],
        ),
    )


@st.cache_resource(show_spinner=False)
def cached_release_artifacts(
    release_dir: str,
    release_version: str,
    manifest_digest: str,
) -> demo.ReleaseArtifacts:
    expected_identity = (release_dir, release_version, manifest_digest)
    if demo.release_cache_identity(release_dir) != expected_identity:
        raise demo.StartupError("release manifest changed before artifact loading")
    artifacts = demo.load_release_artifacts(Path(release_dir))
    if artifacts.manifest.version != release_version:
        raise demo.StartupError("loaded release version does not match the cache identity")
    if demo.release_cache_identity(release_dir) != expected_identity:
        raise demo.StartupError("release manifest changed while artifacts were loading")
    return artifacts


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
    defaults = demo.SYNTHETIC_APPLICATION_VALUES
    st.markdown(
        "<div class='workflow-heading'><h2>Application inputs</h2></div>", unsafe_allow_html=True
    )
    with st.form("credit_application", clear_on_submit=False, border=False):
        left, right = st.columns(2, gap="medium")
        with left:
            loan_amnt = st.number_input(
                "Loan amount (USD)",
                min_value=1.0,
                max_value=100_000.0,
                value=float(defaults["loan_amnt"]),
            )
            annual_inc = st.number_input(
                "Annual income (USD)", min_value=1.0, value=float(defaults["annual_inc"])
            )
            dti = st.number_input(
                "Debt-to-income ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults["dti"]),
            )
            fico_range_low = st.number_input(
                "FICO range low",
                min_value=300.0,
                max_value=850.0,
                value=float(defaults["fico_range_low"]),
            )
            fico_range_high = st.number_input(
                "FICO range high",
                min_value=300.0,
                max_value=850.0,
                value=float(defaults["fico_range_high"]),
            )
            delinq_2yrs = st.number_input(
                "Delinquencies (2 years)",
                min_value=0.0,
                value=float(defaults["delinq_2yrs"]),
                step=1.0,
            )
            inq_last_6mths = st.number_input(
                "Inquiries (6 months)",
                min_value=0.0,
                value=float(defaults["inq_last_6mths"]),
                step=1.0,
            )
            open_acc = st.number_input(
                "Open accounts", min_value=0.0, value=float(defaults["open_acc"]), step=1.0
            )
        with right:
            pub_rec = st.number_input(
                "Public records", min_value=0.0, value=float(defaults["pub_rec"]), step=1.0
            )
            revol_bal = st.number_input(
                "Revolving balance (USD)", min_value=0.0, value=float(defaults["revol_bal"])
            )
            revol_util = st.number_input(
                "Revolving utilization (%)",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults["revol_util"]),
            )
            total_acc = st.number_input(
                "Total accounts", min_value=0.0, value=float(defaults["total_acc"]), step=1.0
            )
            purpose = st.selectbox(
                "Purpose",
                PURPOSE_OPTIONS,
                index=PURPOSE_OPTIONS.index(str(defaults["purpose"])),
            )
            home_options = ["MORTGAGE", "RENT", "OWN", "OTHER"]
            home_ownership = st.selectbox(
                "Home ownership",
                home_options,
                index=home_options.index(str(defaults["home_ownership"])),
            )
            verification_options = ["Verified", "Source Verified", "Not Verified"]
            verification_status = st.selectbox(
                "Verification status",
                verification_options,
                index=verification_options.index(str(defaults["verification_status"])),
            )
            emp_length = st.selectbox(
                "Employment length",
                EMPLOYMENT_OPTIONS,
                index=EMPLOYMENT_OPTIONS.index(str(defaults["emp_length"])),
            )
            addr_state = st.text_input(
                "State (two-letter code)",
                value=str(defaults["addr_state"]),
                max_chars=2,
            )
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
    artifacts: demo.ReleaseArtifacts,
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
            explanation.table.columns[1]: st.column_config.NumberColumn(
                format=explanation.number_format
            ),
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


def _threshold_caption(metrics: dict[str, Any]) -> str:
    """Say which cut precision, recall, and the confusion matrix describe.

    Without this the table reads as if it used the conventional 0.5, which this model never
    reaches.
    """
    threshold = metrics.get("classification_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return "Precision, recall, and the confusion matrix use an unrecorded threshold."
    return (
        f"Precision, recall, and the confusion matrix are measured at the {float(threshold):.2f} "
        "approve boundary, not at 0.5. Calibrated probabilities never reach 0.5, so that cut "
        "would classify every application as good. ROC AUC, average precision, and Brier score "
        "do not depend on a threshold."
    )


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


def _render_calibration_chart(curve: pd.DataFrame) -> bool:
    required = {"mean_probability", "observed_default_rate"}
    if not required.issubset(curve.columns):
        return False
    paired = pd.DataFrame(
        {
            "mean_probability": pd.to_numeric(curve["mean_probability"], errors="coerce"),
            "observed_default_rate": pd.to_numeric(curve["observed_default_rate"], errors="coerce"),
            "method": curve["method"].astype(str) if "method" in curve.columns else "calibration",
        }
    )
    finite = np.isfinite(paired["mean_probability"]) & np.isfinite(paired["observed_default_rate"])
    paired = paired.loc[finite]
    if paired.empty:
        return False

    figure = go.Figure()
    colors = ["#078A8C", "#D99A00", "#D64545"]
    for index, (method, method_rows) in enumerate(paired.groupby("method", sort=False)):
        method_rows = method_rows.sort_values("mean_probability", kind="stable")
        figure.add_trace(
            go.Scatter(
                x=method_rows["mean_probability"],
                y=method_rows["observed_default_rate"],
                mode="lines+markers",
                name=str(method).replace("_", " ").title(),
                line={"color": colors[index % len(colors)], "width": 2},
                marker={"size": 7},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#5F6B76", "width": 1.5, "dash": "dash"},
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 52, "r": 20, "t": 22, "b": 52},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#17202A", "size": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={
            "title": "Mean predicted probability",
            "range": [0.0, 1.0],
            "gridcolor": "#E8EDF1",
            "zeroline": False,
        },
        yaxis={
            "title": "Observed default rate",
            "range": [0.0, 1.0],
            "gridcolor": "#E8EDF1",
            "zeroline": False,
        },
        hovermode="closest",
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    return True


def _render_business_cost_sensitivity_chart(sensitivity: pd.DataFrame) -> bool:
    required = {
        "lgd",
        "margin",
        "review_cost",
        "optimal_cost_per_1000_applications",
    }
    if not required.issubset(sensitivity.columns):
        return False

    scenarios = sensitivity.loc[:, sorted(required)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(scenarios.loc[:, list(required)]).all(axis=1)
    scenarios = scenarios.loc[finite]
    if scenarios.empty:
        return False

    colors = [
        "#078A8C",
        "#D99A00",
        "#D64545",
        "#5F6B76",
        "#3877B2",
        "#7A5C99",
        "#3E8E5B",
        "#B56B2D",
        "#A64D79",
    ]
    figure = go.Figure()
    grouped = scenarios.groupby(["lgd", "margin"], sort=True)
    for index, ((lgd, margin), rows) in enumerate(grouped):
        rows = rows.sort_values("review_cost", kind="stable")
        figure.add_trace(
            go.Scatter(
                x=rows["review_cost"],
                y=rows["optimal_cost_per_1000_applications"],
                mode="lines+markers",
                name=f"LGD {lgd:.0%} / Margin {margin:.0%}",
                line={"color": colors[index % len(colors)], "width": 2},
                marker={"size": 7},
            )
        )
    figure.update_layout(
        height=360,
        margin={"l": 58, "r": 20, "t": 22, "b": 52},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#17202A", "size": 12},
        legend={"orientation": "h", "y": 1.22, "x": 0},
        xaxis={
            "title": "Manual review cost",
            "gridcolor": "#E8EDF1",
            "zeroline": False,
        },
        yaxis={
            "title": "Cost per 1,000 applications",
            "gridcolor": "#E8EDF1",
            "zeroline": False,
        },
        hovermode="closest",
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    return True


def _render_model_performance(artifacts: demo.ReleaseArtifacts) -> None:
    st.subheader("Final out-of-time performance")
    st.dataframe(_metric_rows(artifacts.final_test_metrics), hide_index=True, width="stretch")
    st.caption(
        f"Evaluation sample: {artifacts.final_test_metrics.get('test_samples', 'Unavailable')}"
    )
    st.caption(_threshold_caption(artifacts.final_test_metrics))
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


def _render_calibration(artifacts: demo.ReleaseArtifacts) -> None:
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
    if not _render_calibration_chart(curve):
        st.info("Calibration curve values are unavailable in this release.")
    st.dataframe(curve, hide_index=True, width="stretch")
    with st.expander("Calibration metadata"):
        st.json(artifacts.calibration_metrics, expanded=False)


def _render_business_cost(artifacts: demo.ReleaseArtifacts) -> None:
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
    if not _render_business_cost_sensitivity_chart(sensitivity):
        st.info("Cost sensitivity chart values are unavailable in this release.")
    st.dataframe(sensitivity, hide_index=True, width="stretch")


def _render_fairness(artifacts: demo.ReleaseArtifacts) -> None:
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


def _render_limitations(artifacts: demo.ReleaseArtifacts) -> None:
    st.subheader("Known limitations")
    st.markdown(
        "- Historical LendingClub data reflects prior underwriting and selection, not the "
        "full applicant population.\n"
        "- Model outputs and SHAP values describe statistical associations, not causal "
        "effects.\n"
        "- The release does not establish statutory fair-lending compliance or production "
        "monitoring.\n"
        "- Joblib uses Python pickle semantics. Load only releases from a trusted source; "
        "manifest hashes do not authenticate an untrusted bundle.\n"
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


def _render_evidence(artifacts: demo.ReleaseArtifacts) -> None:
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
        release_identity = demo.release_cache_identity(demo.release_directory())
        artifacts = cached_release_artifacts(*release_identity)
    except demo.StartupError as exc:
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
            prediction = demo.predict_application(values, artifacts)
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            field = ".".join(str(item) for item in first_error["loc"])
            error_message = f"Invalid {field}: {first_error['msg']}"
        except demo.PredictionError:
            error_message = "Assessment could not be completed from the validated release."

    with result_column:
        _render_result(prediction, artifacts, error_message)

    st.divider()
    _render_evidence(artifacts)


if __name__ == "__main__":
    main()
