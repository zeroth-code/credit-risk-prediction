import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from credit_risk.explainability import (
    _sample_row_positions,
    generate_shap_explanations,
    select_example_indices,
)

FIGURE_FILENAMES = {
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


def _explanation_inputs() -> tuple[LGBMClassifier, np.ndarray, list[str], pd.DataFrame]:
    random = np.random.default_rng(19)
    matrix = random.normal(size=(30, 5))
    target = (matrix[:, 0] + 0.7 * matrix[:, 1] - 0.4 * matrix[:, 2] > 0.0).astype(int)
    model = LGBMClassifier(
        n_estimators=12,
        num_leaves=5,
        min_child_samples=1,
        random_state=19,
        n_jobs=1,
        verbosity=-1,
    ).fit(matrix, target)
    scored = pd.DataFrame(
        {
            "id": np.arange(1000, 1030),
            "action": ["approve"] * 10 + ["manual_review"] * 10 + ["decline"] * 10,
            "probability": np.linspace(0.05, 0.95, 30),
        }
    )
    return model, matrix, [f"feature_{index}" for index in range(5)], scored


def test_select_example_indices_returns_each_policy_action() -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "manual_review", "decline"],
            "probability": [0.1, 0.5, 0.9],
        }
    )

    result = select_example_indices(scored)

    assert set(result) == {"approve", "manual_review", "decline"}


def test_select_example_indices_uses_lowest_index_for_median_ties() -> None:
    scored = pd.DataFrame(
        {
            "action": [
                "approve",
                "approve",
                "manual_review",
                "manual_review",
                "decline",
                "decline",
            ],
            "probability": [0.1, 0.3, 0.4, 0.6, 0.8, 1.0],
        },
        index=[7, 2, 9, 1, 5, 3],
    )

    result = select_example_indices(scored)

    assert result == {"approve": 2, "manual_review": 1, "decline": 3}


@pytest.mark.parametrize("missing_column", ["action", "probability"])
def test_select_example_indices_requires_scored_columns(missing_column: str) -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "manual_review", "decline"],
            "probability": [0.1, 0.5, 0.9],
        }
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match="required columns"):
        select_example_indices(scored)


def test_select_example_indices_rejects_unknown_actions() -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "manual_review", "decline", "refer"],
            "probability": [0.1, 0.5, 0.9, 0.6],
        }
    )

    with pytest.raises(ValueError, match="unsupported actions.*refer"):
        select_example_indices(scored)


@pytest.mark.parametrize(
    "invalid_probability",
    [True, "0.5", np.nan, np.inf, -0.01, 1.01],
    ids=["boolean", "string", "nan", "infinity", "below-zero", "above-one"],
)
def test_select_example_indices_rejects_invalid_probabilities(
    invalid_probability: object,
) -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "manual_review", "decline"],
            "probability": [0.1, invalid_probability, 0.9],
        }
    )

    with pytest.raises(ValueError, match="probability.*finite numeric.*between 0 and 1"):
        select_example_indices(scored)


@pytest.mark.parametrize(
    "index",
    [[0, 0, 2], ["a", "b", "c"], [0.0, 1.0, 2.0], [-1, 1, 2]],
    ids=["duplicate", "string", "float", "negative"],
)
def test_select_example_indices_requires_unique_nonnegative_integer_index(
    index: list[object],
) -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "manual_review", "decline"],
            "probability": [0.1, 0.5, 0.9],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="index.*unique nonnegative integers"):
        select_example_indices(scored)


def test_select_example_indices_requires_every_policy_action() -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "decline"],
            "probability": [0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="no scored example for action: manual_review"):
        select_example_indices(scored)


def test_sample_row_positions_is_capped_deterministic_and_keeps_required_rows() -> None:
    required = np.array([7, 2222, 5999])

    first = _sample_row_positions(6000, required_positions=required)
    second = _sample_row_positions(6000, required_positions=required)

    assert len(first) == 5000
    assert np.array_equal(first, second)
    assert np.all(first[:-1] < first[1:])
    assert set(required).issubset(first)


def test_sample_row_positions_uses_every_row_below_cap() -> None:
    result = _sample_row_positions(12, required_positions=np.array([3, 8]))

    np.testing.assert_array_equal(result, np.arange(12))


def test_generate_shap_explanations_writes_stable_compact_artifacts(
    tmp_path: Path,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "reports/figures"

    payload = generate_shap_explanations(
        model,
        sparse.csr_matrix(matrix),
        feature_names,
        scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        row_identifier_column="id",
    )

    importance_path = artifact_dir / "shap_importance.csv"
    payload_path = artifact_dir / "shap_explanations.json"
    assert importance_path.is_file()
    assert payload_path.is_file()
    assert {path.name for path in figure_dir.iterdir()} == FIGURE_FILENAMES

    importance = pd.read_csv(importance_path)
    assert importance.columns.tolist() == ["rank", "feature", "mean_abs_shap"]
    assert importance["rank"].tolist() == [1, 2, 3, 4, 5]
    assert set(importance["feature"]) == set(feature_names)
    assert importance["mean_abs_shap"].ge(0.0).all()
    assert importance["mean_abs_shap"].is_monotonic_decreasing

    saved_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert saved_payload == payload
    assert payload["schema_version"] == "1.0"
    assert payload["explanation_model"] == {
        "artifact": "uncalibrated_model.joblib",
        "source": "frozen_uncalibrated_lightgbm",
        "output_space": "raw_model_output",
        "units": "log_odds",
        "calibrated_probability_source": "frozen_calibrated_model",
        "calibration_note": (
            "SHAP values explain the frozen base LightGBM score, not the post-calibration "
            "probability."
        ),
    }
    assert payload["sample"] == {
        "random_seed": 42,
        "maximum_rows": 5000,
        "test_rows": 30,
        "explained_rows": 30,
    }
    assert payload["feature_names"] == feature_names
    assert [item["feature"] for item in payload["global_top_features"]] == importance[
        "feature"
    ].tolist()
    assert payload["files"] == {
        "importance": "shap_importance.csv",
        "payload": "shap_explanations.json",
        "beeswarm": "shap_beeswarm.png",
        "dependence": [
            {"feature": feature, "filename": f"shap_dependence_{rank:02d}.png"}
            for rank, feature in enumerate(importance["feature"], start=1)
        ],
        "waterfalls": {
            "approve": "shap_waterfall_approve.png",
            "manual_review": "shap_waterfall_manual_review.png",
            "decline": "shap_waterfall_decline.png",
        },
    }
    assert set(payload["local_explanations"]) == {"approve", "manual_review", "decline"}
    for action, expected_index in {
        "approve": 4,
        "manual_review": 14,
        "decline": 24,
    }.items():
        local = payload["local_explanations"][action]
        assert local["policy_action"] == action
        assert local["scored_index"] == expected_index
        assert local["row_identifier"] == {
            "column": "id",
            "value": int(scored.loc[expected_index, "id"]),
        }
        assert local["calibrated_probability"] == pytest.approx(
            scored.loc[expected_index, "probability"]
        )
        assert 1 <= len(local["top_contributions"]) <= 5
        assert local["waterfall"] == f"shap_waterfall_{action}.png"

    first_artifact_bytes = {
        path.name: path.read_bytes() for path in (importance_path, payload_path)
    }
    first_figure_bytes = {path.name: path.read_bytes() for path in figure_dir.iterdir()}

    generate_shap_explanations(
        model,
        sparse.csr_matrix(matrix),
        feature_names,
        scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        row_identifier_column="id",
    )

    assert {path.name: path.read_bytes() for path in (importance_path, payload_path)} == (
        first_artifact_bytes
    )
    assert {path.name: path.read_bytes() for path in figure_dir.iterdir()} == first_figure_bytes


def test_generate_shap_explanations_accepts_dense_transformed_matrix(tmp_path: Path) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()

    payload = generate_shap_explanations(
        model,
        matrix,
        feature_names,
        scored,
        artifact_dir=tmp_path / "artifacts",
        figure_dir=tmp_path / "figures",
    )

    assert payload["sample"]["explained_rows"] == len(matrix)
    assert {path.name for path in (tmp_path / "figures").iterdir()} == FIGURE_FILENAMES


@pytest.mark.parametrize(
    "feature_names",
    [
        ["one", "two", "three", "four"],
        ["one", "two", "three", "four", "four"],
        ["one", "two", "three", "four", ""],
    ],
    ids=["count-mismatch", "duplicate", "empty"],
)
def test_generate_shap_explanations_validates_feature_names(
    tmp_path: Path,
    feature_names: list[str],
) -> None:
    model, matrix, _, scored = _explanation_inputs()

    with pytest.raises(ValueError, match="transformed feature names"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )


def test_generate_shap_explanations_rejects_wrong_model_type(tmp_path: Path) -> None:
    _, matrix, feature_names, scored = _explanation_inputs()
    wrong_model = LogisticRegression().fit(matrix, np.array([0, 1] * 15))

    with pytest.raises(ValueError, match="fitted LightGBM"):
        generate_shap_explanations(
            wrong_model,
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )


def test_generate_shap_explanations_rejects_unfitted_lightgbm(tmp_path: Path) -> None:
    _, matrix, feature_names, scored = _explanation_inputs()

    with pytest.raises(ValueError, match="fitted LightGBM"):
        generate_shap_explanations(
            LGBMClassifier(),
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )
