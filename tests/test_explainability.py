import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import credit_risk.explainability as explainability
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


def _custom_binary_objective(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = 1.0 / (1.0 + np.exp(-raw_predictions))
    gradient = probabilities - y_true
    hessian = probabilities * (1.0 - probabilities)
    return gradient, hessian


def _explanation_inputs(
    feature_count: int = 5,
) -> tuple[LGBMClassifier, np.ndarray, list[str], pd.DataFrame]:
    random = np.random.default_rng(19)
    matrix = random.normal(size=(30, feature_count))
    signal = matrix[:, 0].copy()
    if feature_count > 1:
        signal += 0.7 * matrix[:, 1]
    if feature_count > 2:
        signal -= 0.4 * matrix[:, 2]
    target = (signal > 0.0).astype(int)
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
    return model, matrix, [f"feature_{index}" for index in range(feature_count)], scored


def _known_output_paths(artifact_dir: Path, figure_dir: Path) -> list[Path]:
    return [
        artifact_dir / "shap_importance.csv",
        *(figure_dir / filename for filename in sorted(FIGURE_FILENAMES)),
        artifact_dir / "shap_explanations.json",
    ]


def _seed_old_outputs(paths: list[Path]) -> dict[Path, bytes]:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"old:{path.name}".encode())
    return {path: path.read_bytes() for path in paths}


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


def test_select_example_indices_does_not_merge_nearby_distances_into_a_tie() -> None:
    scored = pd.DataFrame(
        {
            "action": [
                "approve",
                "approve",
                "approve",
                "manual_review",
                "decline",
            ],
            "probability": [0.2, 0.2000000000000005, 0.9, 0.5, 0.95],
        },
        index=[0, 5, 8, 1, 2],
    )

    result = select_example_indices(scored)

    assert result["approve"] == 5


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


def test_select_example_indices_returns_only_observed_policy_actions() -> None:
    scored = pd.DataFrame(
        {
            "action": ["approve", "decline"],
            "probability": [0.1, 0.9],
        }
    )

    assert select_example_indices(scored) == {"approve": 0, "decline": 1}


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
        "objective": "binary",
        "sigmoid": 1.0,
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
        "approve": 5,
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
        assert local["base_model_raw_output"] == pytest.approx(
            model.predict(matrix[[expected_index]], raw_score=True)[0]
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


def test_generate_shap_explanations_records_unavailable_policy_action(
    tmp_path: Path,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    observed_mask = scored["action"] != "decline"
    observed_scored = scored.loc[observed_mask].copy()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    obsolete_decline_waterfall = figure_dir / "shap_waterfall_decline.png"
    obsolete_decline_waterfall.write_bytes(b"obsolete")

    payload = generate_shap_explanations(
        model,
        matrix[observed_mask.to_numpy()],
        feature_names,
        observed_scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
    )

    assert payload["local_explanations"]["decline"] is None
    assert payload["files"]["waterfalls"] == {
        "approve": "shap_waterfall_approve.png",
        "manual_review": "shap_waterfall_manual_review.png",
    }
    assert not obsolete_decline_waterfall.exists()


def test_generate_shap_explanations_preserves_published_outputs_on_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    final_paths = _known_output_paths(artifact_dir, figure_dir)
    original_bytes = _seed_old_outputs(final_paths)
    original_save_figure = explainability._save_figure
    render_calls = 0

    def fail_second_render(
        plt: object,
        path: Path,
        *,
        width: float,
        height: float,
    ) -> None:
        nonlocal render_calls
        render_calls += 1
        if render_calls == 2:
            raise RuntimeError("injected render failure")
        original_save_figure(plt, path, width=width, height=height)

    monkeypatch.setattr(explainability, "_save_figure", fail_second_render)

    with pytest.raises(RuntimeError, match="injected render failure"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=artifact_dir,
            figure_dir=figure_dir,
        )

    assert {path: path.read_bytes() for path in final_paths} == original_bytes
    assert all(".staging" not in path.name for path in artifact_dir.iterdir())
    assert all(".staging" not in path.name for path in figure_dir.iterdir())


def test_generate_shap_explanations_restores_outputs_on_late_payload_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    observed_mask = scored["action"] != "decline"
    observed_scored = scored.loc[observed_mask].copy()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    final_paths = _known_output_paths(artifact_dir, figure_dir)
    original_bytes = _seed_old_outputs(final_paths)
    current_final = artifact_dir / "shap_importance.csv"
    stale_final = figure_dir / "shap_waterfall_decline.png"
    payload_final = artifact_dir / "shap_explanations.json"
    original_replace = Path.replace
    boundary_observed = False

    def fail_payload_commit(path: Path, target: Path) -> Path:
        nonlocal boundary_observed
        if (
            target == payload_final
            and path.name.startswith(".shap_explanations")
            and ".new.staging" in path.name
        ):
            boundary_observed = True
            assert current_final.read_bytes() != original_bytes[current_final]
            assert not stale_final.exists()
            assert payload_final.read_bytes() == original_bytes[payload_final]
            raise OSError("injected late payload commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_payload_commit)

    with pytest.raises(OSError, match="injected late payload commit failure"):
        generate_shap_explanations(
            model,
            matrix[observed_mask.to_numpy()],
            feature_names,
            observed_scored,
            artifact_dir=artifact_dir,
            figure_dir=figure_dir,
        )

    assert boundary_observed
    assert {path: path.read_bytes() for path in final_paths} == original_bytes
    assert all(".staging" not in path.name for path in artifact_dir.iterdir())
    assert all(".staging" not in path.name for path in figure_dir.iterdir())


def test_generate_shap_explanations_retries_restore_and_recovers_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    final_paths = _known_output_paths(artifact_dir, figure_dir)
    original_bytes = _seed_old_outputs(final_paths)
    original_replace = Path.replace
    restore_attempts = 0

    def injected_replace(path: Path, target: Path) -> Path:
        nonlocal restore_attempts
        if path.name.startswith(".shap_dependence_01") and ".new.staging" in path.name:
            raise OSError("injected publication failure")
        if path.name.startswith(".shap_importance") and ".backup.staging" in path.name:
            restore_attempts += 1
            if restore_attempts == 1:
                raise OSError("transient restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", injected_replace)

    with pytest.raises(OSError, match="injected publication failure"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=artifact_dir,
            figure_dir=figure_dir,
        )

    assert restore_attempts == 2
    assert {path: path.read_bytes() for path in final_paths} == original_bytes
    assert all(".staging" not in path.name for path in artifact_dir.iterdir())
    assert all(".staging" not in path.name for path in figure_dir.iterdir())


def test_generate_shap_explanations_preserves_unresolved_restore_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    final_paths = _known_output_paths(artifact_dir, figure_dir)
    original_bytes = _seed_old_outputs(final_paths)
    importance_path = artifact_dir / "shap_importance.csv"
    original_replace = Path.replace

    def injected_replace(path: Path, target: Path) -> Path:
        if path.name.startswith(".shap_dependence_01") and ".new.staging" in path.name:
            raise OSError("injected publication failure")
        if path.name.startswith(".shap_importance") and ".backup.staging" in path.name:
            raise OSError("persistent restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", injected_replace)

    with pytest.raises(OSError, match="injected publication failure") as exc_info:
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=artifact_dir,
            figure_dir=figure_dir,
        )

    for path in final_paths:
        if path != importance_path:
            assert path.read_bytes() == original_bytes[path]
    remaining_temporary_paths = [
        path
        for directory in (artifact_dir, figure_dir)
        for path in directory.iterdir()
        if ".staging" in path.name
    ]
    assert len(remaining_temporary_paths) == 1
    unresolved_backup = remaining_temporary_paths[0]
    assert unresolved_backup.name.startswith(".shap_importance")
    assert ".backup.staging" in unresolved_backup.name
    assert unresolved_backup.read_bytes() == original_bytes[importance_path]
    notes = "\n".join(getattr(exc_info.value, "__notes__", []))
    assert str(importance_path) in notes
    assert str(unresolved_backup) in notes


def test_generate_shap_explanations_retries_cleanup_after_payload_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    final_paths = _known_output_paths(artifact_dir, figure_dir)
    old_bytes = _seed_old_outputs(final_paths)
    original_unlink = Path.unlink
    cleanup_attempts = 0

    def injected_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal cleanup_attempts
        if path.name.startswith(".shap_importance") and ".backup.staging" in path.name:
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                raise OSError("transient cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", injected_unlink)

    payload = generate_shap_explanations(
        model,
        matrix,
        feature_names,
        scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
    )

    assert cleanup_attempts == 2
    assert json.loads((artifact_dir / "shap_explanations.json").read_text()) == payload
    assert all(path.read_bytes() != old_bytes[path] for path in final_paths)
    assert all(".staging" not in path.name for path in artifact_dir.iterdir())
    assert all(".staging" not in path.name for path in figure_dir.iterdir())
    continuation_path = tmp_path / "continued.txt"
    continuation_path.write_text("continued", encoding="utf-8")
    assert continuation_path.read_text(encoding="utf-8") == "continued"


def test_generate_shap_explanations_removes_obsolete_dependence_plots(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    model, matrix, feature_names, scored = _explanation_inputs()
    generate_shap_explanations(
        model,
        matrix,
        feature_names,
        scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
    )
    smaller_model, smaller_matrix, smaller_names, smaller_scored = _explanation_inputs(3)

    payload = generate_shap_explanations(
        smaller_model,
        smaller_matrix,
        smaller_names,
        smaller_scored,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
    )

    assert not (figure_dir / "shap_dependence_04.png").exists()
    assert not (figure_dir / "shap_dependence_05.png").exists()
    assert payload["files"]["dependence"] == [
        {
            "feature": item["feature"],
            "filename": f"shap_dependence_{rank:02d}.png",
        }
        for rank, item in enumerate(payload["global_top_features"], start=1)
    ]


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


@pytest.mark.parametrize(
    "objective",
    ["regression", _custom_binary_objective],
    ids=["regression", "custom"],
)
def test_generate_shap_explanations_rejects_unsupported_fitted_objective(
    tmp_path: Path,
    objective: object,
) -> None:
    _, matrix, feature_names, scored = _explanation_inputs()
    target = np.array([0, 1] * 15)
    model = LGBMClassifier(
        objective=objective,
        n_estimators=4,
        min_child_samples=1,
        random_state=19,
        n_jobs=1,
        verbosity=-1,
    ).fit(matrix, target)

    with pytest.raises(ValueError, match="objective.*binary"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )


def test_generate_shap_explanations_rejects_raw_score_reconstruction_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix, feature_names, scored = _explanation_inputs()
    monkeypatch.setattr(
        model,
        "predict",
        lambda sample, *, raw_score: np.zeros(len(sample), dtype=float),
    )

    with pytest.raises(ValueError, match="raw-score reconstruction"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )


def test_generate_shap_explanations_rejects_nondefault_binary_sigmoid(
    tmp_path: Path,
) -> None:
    _, matrix, feature_names, scored = _explanation_inputs()
    model = LGBMClassifier(
        objective="binary",
        sigmoid=2.0,
        n_estimators=4,
        min_child_samples=1,
        random_state=19,
        n_jobs=1,
        verbosity=-1,
    ).fit(matrix, np.array([0, 1] * 15))

    with pytest.raises(ValueError, match="sigmoid.*1.0"):
        generate_shap_explanations(
            model,
            matrix,
            feature_names,
            scored,
            artifact_dir=tmp_path / "artifacts",
            figure_dir=tmp_path / "figures",
        )
