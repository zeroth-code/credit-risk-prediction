import json
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

POLICY_ACTIONS = ("approve", "manual_review", "decline")
MAX_SHAP_ROWS = 5000
SHAP_RANDOM_SEED = 42
MAX_DENSE_SHAP_ELEMENTS = 25_000_000
SHAP_IMPORTANCE_FILENAME = "shap_importance.csv"
SHAP_PAYLOAD_FILENAME = "shap_explanations.json"
SHAP_BEESWARM_FILENAME = "shap_beeswarm.png"
SHAP_DEPENDENCE_FILENAMES = tuple(f"shap_dependence_{rank:02d}.png" for rank in range(1, 6))
SHAP_WATERFALL_FILENAMES = {action: f"shap_waterfall_{action}.png" for action in POLICY_ACTIONS}
FILESYSTEM_OPERATION_ATTEMPTS = 2


def _sample_row_positions(
    row_count: int,
    *,
    required_positions: np.ndarray,
    max_rows: int = MAX_SHAP_ROWS,
    random_seed: int = SHAP_RANDOM_SEED,
) -> np.ndarray:
    if row_count <= max_rows:
        return np.arange(row_count, dtype=int)

    required = np.unique(np.asarray(required_positions, dtype=int))
    candidate_positions = np.setdiff1d(
        np.arange(row_count, dtype=int),
        required,
        assume_unique=True,
    )
    random = np.random.default_rng(random_seed)
    sampled = random.choice(
        candidate_positions,
        size=max_rows - len(required),
        replace=False,
    )
    return np.sort(np.concatenate([required, sampled]))


def select_example_indices(scored: pd.DataFrame) -> dict[str, int]:
    required_columns = {"action", "probability"}
    missing_columns = sorted(required_columns.difference(scored.columns))
    if missing_columns:
        raise ValueError(f"scored examples missing required columns: {', '.join(missing_columns)}")
    index_values = scored.index.tolist()
    has_usable_index = scored.index.is_unique and all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
        for value in index_values
    )
    if not has_usable_index:
        raise ValueError("scored examples index must contain unique nonnegative integers")

    actions = scored["action"].to_numpy(dtype=object, copy=True)
    unsupported_actions = sorted(
        {
            str(value)
            for value in actions
            if not isinstance(value, str) or value not in POLICY_ACTIONS
        }
    )
    if unsupported_actions:
        joined_actions = ", ".join(unsupported_actions)
        raise ValueError(f"scored examples contain unsupported actions: {joined_actions}")

    probability_series = scored["probability"]
    raw_probabilities = probability_series.to_numpy(dtype=object, copy=True)
    contains_boolean = any(isinstance(value, (bool, np.bool_)) for value in raw_probabilities)
    try:
        probabilities = probability_series.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scored probability must contain finite numeric values between 0 and 1"
        ) from exc
    if (
        contains_boolean
        or not pd.api.types.is_numeric_dtype(probability_series.dtype)
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ValueError("scored probability must contain finite numeric values between 0 and 1")

    selected: dict[str, int] = {}
    for action in POLICY_ACTIONS:
        group = scored.loc[scored["action"] == action]
        if group.empty:
            continue
        median_probability = group["probability"].median()
        distances = (group["probability"] - median_probability).abs()
        tied = distances == distances.min()
        index = min(distances.index[tied])
        selected[action] = int(index)
    return selected


def _validated_feature_names(
    transformed: object,
    feature_names: Sequence[str],
) -> tuple[int, int, list[str]]:
    shape = getattr(transformed, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("transformed features must be a two-dimensional matrix")
    row_count, feature_count = int(shape[0]), int(shape[1])
    if row_count <= 0 or feature_count <= 0:
        raise ValueError("transformed features must contain at least one row and one column")
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("transformed feature names must be a sequence of unique non-empty strings")
    names = list(feature_names)
    valid_names = (
        len(names) == feature_count
        and all(isinstance(name, str) and bool(name.strip()) for name in names)
        and len(names) == len(set(names))
    )
    if not valid_names:
        raise ValueError(
            "transformed feature names must match the matrix columns and be unique "
            "non-empty strings"
        )
    return row_count, feature_count, names


def _validate_explanation_model(
    estimator: object,
    feature_count: int,
) -> tuple[LGBMClassifier, str, float]:
    if not isinstance(estimator, LGBMClassifier):
        raise ValueError("explanation model must be a fitted LightGBM classifier")
    try:
        check_is_fitted(estimator)
    except NotFittedError as exc:
        raise ValueError("explanation model must be a fitted LightGBM classifier") from exc
    model_feature_count = getattr(estimator, "n_features_in_", None)
    if model_feature_count != feature_count:
        raise ValueError(
            "fitted LightGBM feature count must match transformed feature names and matrix columns"
        )
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.shape != (2,) or not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("explanation model must be a fitted binary LightGBM classifier")
    booster = getattr(estimator, "booster_", None)
    booster_params = getattr(booster, "params", None)
    objective = booster_params.get("objective") if isinstance(booster_params, dict) else None
    if objective != "binary":
        raise ValueError(
            f"fitted LightGBM objective must be binary for log-odds SHAP output, got {objective!r}"
        )
    configured_sigmoid = booster_params.get("sigmoid", 1.0)
    if isinstance(configured_sigmoid, (bool, np.bool_)):
        raise ValueError("fitted binary LightGBM sigmoid must be 1.0 for log-odds SHAP output")
    try:
        sigmoid = float(configured_sigmoid)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fitted binary LightGBM sigmoid must be 1.0 for log-odds SHAP output"
        ) from exc
    if not np.isfinite(sigmoid) or sigmoid != 1.0:
        raise ValueError("fitted binary LightGBM sigmoid must be 1.0 for log-odds SHAP output")
    return estimator, objective, sigmoid


def _dense_sample(
    transformed: object,
    positions: np.ndarray,
    *,
    feature_count: int,
) -> np.ndarray:
    if len(positions) > MAX_SHAP_ROWS:
        raise ValueError(f"SHAP sample must not exceed {MAX_SHAP_ROWS} rows")
    element_count = len(positions) * feature_count
    if element_count > MAX_DENSE_SHAP_ELEMENTS:
        raise ValueError(
            "SHAP sample is too large to densify safely: "
            f"{element_count} elements exceeds {MAX_DENSE_SHAP_ELEMENTS}"
        )
    if sparse.issparse(transformed):
        sampled = transformed[positions].toarray()
    else:
        sampled = np.asarray(transformed)[positions]
    try:
        dense = np.asarray(sampled, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed SHAP sample must contain numeric values") from exc
    if dense.shape != (len(positions), feature_count):
        raise ValueError("transformed SHAP sample shape is not aligned")
    if not np.isfinite(dense).all():
        raise ValueError("transformed SHAP sample must contain only finite values")
    return dense


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("row identifier must be finite")
        return value
    return str(value)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=True, indent=2, sort_keys=True)
        output_file.write("\n")


def _temporary_sibling(path: Path, token: str, role: str) -> Path:
    return path.with_name(f".{path.stem}.{token}.{role}.staging{path.suffix}")


def _validate_staged_files(paths: list[Path]) -> None:
    missing_or_empty = [
        path.name for path in paths if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_or_empty:
        raise RuntimeError(
            f"SHAP staging outputs missing or empty: {', '.join(sorted(missing_or_empty))}"
        )


def _retry_filesystem_operation(operation: Callable[[], None]) -> OSError | None:
    last_error: OSError | None = None
    for _ in range(FILESYSTEM_OPERATION_ATTEMPTS):
        try:
            operation()
            return None
        except OSError as exc:
            last_error = exc
    return last_error


def _cleanup_temporary_files(
    paths: list[Path],
    *,
    preserve: set[Path] | None = None,
) -> list[str]:
    preserved_paths = preserve or set()
    failures: list[str] = []
    for path in paths:
        if path in preserved_paths:
            continue

        def unlink_if_present(path: Path = path) -> None:
            if path.is_file():
                path.unlink()

        error = _retry_filesystem_operation(unlink_if_present)
        if error is not None:
            failures.append(f"cleanup failed for temporary path {path}: {error}")
    return failures


def _restore_published_outputs(
    final_paths: list[Path],
    previous_outputs: dict[Path, Path | None],
) -> tuple[list[str], set[Path]]:
    failures: list[str] = []
    preserved_backups: set[Path] = set()
    for final_path in final_paths:
        backup_path = previous_outputs[final_path]
        if backup_path is None:

            def remove_new_final(final_path: Path = final_path) -> None:
                if final_path.is_file():
                    final_path.unlink()

            error = _retry_filesystem_operation(remove_new_final)
            if error is not None:
                failures.append(f"recovery failed for final {final_path} without a backup: {error}")
            continue

        def restore_backup(
            final_path: Path = final_path,
            backup_path: Path = backup_path,
        ) -> None:
            backup_path.replace(final_path)

        error = _retry_filesystem_operation(restore_backup)
        if error is not None:
            preserved_backups.add(backup_path)
            failures.append(
                f"recovery failed for final {final_path} from backup {backup_path}: {error}"
            )
    return failures, preserved_backups


def _add_exception_notes(exception: BaseException, notes: list[str]) -> None:
    for note in notes:
        exception.add_note(note)


def _publish_staged_outputs(
    *,
    staged_by_final: dict[Path, Path],
    backup_by_final: dict[Path, Path],
    known_final_paths: list[Path],
    current_non_payload_finals: list[Path],
    obsolete_final_paths: list[Path],
    payload_final: Path,
) -> None:
    temporary_paths = [*staged_by_final.values(), *backup_by_final.values()]
    previous_outputs: dict[Path, Path | None] = {}
    publish_started = False
    publication_committed = False

    try:
        for final_path in known_final_paths:
            if final_path.is_file():
                backup_path = backup_by_final[final_path]
                copyfile(final_path, backup_path)
                previous_outputs[final_path] = backup_path
            else:
                previous_outputs[final_path] = None

        publish_started = True
        for final_path in current_non_payload_finals:
            staged_by_final[final_path].replace(final_path)
        for obsolete_path in obsolete_final_paths:
            if obsolete_path.is_file():
                obsolete_path.unlink()
        staged_by_final[payload_final].replace(payload_final)
        publication_committed = True
    except Exception as publication_error:
        recovery_notes: list[str] = []
        preserved_backups: set[Path] = set()
        if publish_started and not publication_committed:
            recovery_notes, preserved_backups = _restore_published_outputs(
                known_final_paths,
                previous_outputs,
            )
        cleanup_notes = _cleanup_temporary_files(
            temporary_paths,
            preserve=preserved_backups,
        )
        _add_exception_notes(publication_error, [*recovery_notes, *cleanup_notes])
        raise
    else:
        _cleanup_temporary_files(temporary_paths)


def _save_figure(plt: object, path: Path, *, width: float, height: float) -> None:
    figure = plt.gcf()  # type: ignore[attr-defined]
    figure.set_size_inches(width, height)
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "credit-risk-prediction"},
    )
    plt.close(figure)  # type: ignore[attr-defined]


@contextmanager
def _deterministic_plotting() -> Iterator[None]:
    random_state = np.random.get_state()
    np.random.seed(SHAP_RANDOM_SEED)
    try:
        yield
    finally:
        np.random.set_state(random_state)


def generate_shap_explanations(
    estimator: object,
    transformed: object,
    feature_names: Sequence[str],
    scored: pd.DataFrame,
    *,
    artifact_dir: str | Path,
    figure_dir: str | Path,
    row_identifier_column: str | None = None,
    model_artifact_name: str = "uncalibrated_model.joblib",
) -> dict[str, object]:
    row_count, feature_count, names = _validated_feature_names(transformed, feature_names)
    model, objective, sigmoid = _validate_explanation_model(estimator, feature_count)
    if len(scored) != row_count:
        raise ValueError("scored examples must align with transformed feature rows")
    if row_identifier_column is not None and row_identifier_column not in scored.columns:
        raise ValueError(f"scored examples missing row identifier column: {row_identifier_column}")

    selected_indices = select_example_indices(scored)
    selected_positions = scored.index.get_indexer(list(selected_indices.values()))
    if np.any(selected_positions < 0):
        raise RuntimeError("selected explanation examples are not aligned with scored rows")
    sample_positions = _sample_row_positions(
        row_count,
        required_positions=selected_positions,
    )
    sample = _dense_sample(transformed, sample_positions, feature_count=feature_count)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        import shap

    try:
        explainer = shap.TreeExplainer(
            model,
            model_output="raw",
            feature_names=names,
        )
        explanations = explainer(sample, check_additivity=False)
    except Exception as exc:
        raise ValueError(f"fitted LightGBM is not suitable for TreeExplainer: {exc}") from exc
    shap_values = np.asarray(explanations.values, dtype=float)
    if shap_values.shape != sample.shape or not np.isfinite(shap_values).all():
        raise ValueError(
            "TreeExplainer must return finite SHAP values aligned with the sampled matrix"
        )
    base_values = np.asarray(explanations.base_values, dtype=float)
    if base_values.ndim == 0:
        base_values = np.full(len(sample), float(base_values), dtype=float)
    if base_values.shape != (len(sample),) or not np.isfinite(base_values).all():
        raise ValueError("TreeExplainer must return finite base values aligned with the sample")
    try:
        raw_scores = np.asarray(model.predict(sample, raw_score=True), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("LightGBM raw scores must be finite and aligned with the sample") from exc
    reconstructed_raw_scores = base_values + shap_values.sum(axis=1)
    if (
        raw_scores.shape != (len(sample),)
        or not np.isfinite(raw_scores).all()
        or not np.isfinite(reconstructed_raw_scores).all()
        or not np.allclose(raw_scores, reconstructed_raw_scores, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(
            "TreeExplainer raw-score reconstruction does not match fitted LightGBM raw scores"
        )

    artifact_path = Path(artifact_dir)
    figure_path = Path(figure_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1, dtype=int))
    top_feature_count = min(5, feature_count)
    ranked_features = importance.head(top_feature_count)
    dependence_files = [
        {
            "feature": str(row.feature),
            "filename": SHAP_DEPENDENCE_FILENAMES[rank - 1],
        }
        for rank, row in enumerate(ranked_features.itertuples(index=False), start=1)
    ]
    waterfall_files = {
        action: SHAP_WATERFALL_FILENAMES[action] for action in selected_indices
    }
    global_top_features = [
        {
            "rank": int(row.rank),
            "feature": str(row.feature),
            "mean_abs_shap": float(row.mean_abs_shap),
        }
        for row in importance.head(5).itertuples(index=False)
    ]

    importance_final = artifact_path / SHAP_IMPORTANCE_FILENAME
    payload_final = artifact_path / SHAP_PAYLOAD_FILENAME
    beeswarm_final = figure_path / SHAP_BEESWARM_FILENAME
    dependence_finals = [figure_path / item["filename"] for item in dependence_files]
    waterfall_finals = [figure_path / filename for filename in waterfall_files.values()]
    known_waterfall_finals = [
        figure_path / filename for filename in SHAP_WATERFALL_FILENAMES.values()
    ]
    current_non_payload_finals = [
        importance_final,
        beeswarm_final,
        *dependence_finals,
        *waterfall_finals,
    ]
    known_final_paths = [
        importance_final,
        beeswarm_final,
        *(figure_path / filename for filename in SHAP_DEPENDENCE_FILENAMES),
        *known_waterfall_finals,
        payload_final,
    ]
    token = uuid4().hex
    staged_by_final = {
        final_path: _temporary_sibling(final_path, token, "new")
        for final_path in [*current_non_payload_finals, payload_final]
    }
    backup_by_final = {
        final_path: _temporary_sibling(final_path, token, "backup")
        for final_path in known_final_paths
    }
    current_dependence_finals = set(dependence_finals)
    obsolete_final_paths = [
        figure_path / filename
        for filename in SHAP_DEPENDENCE_FILENAMES
        if figure_path / filename not in current_dependence_finals
    ] + [
        final_path for final_path in known_waterfall_finals if final_path not in waterfall_finals
    ]

    try:
        importance.to_csv(
            staged_by_final[importance_final],
            index=False,
            encoding="utf-8",
            float_format="%.12g",
        )

        plt.close("all")
        with _deterministic_plotting():
            shap.plots.beeswarm(
                explanations,
                max_display=min(20, feature_count),
                show=False,
                plot_size=(10.0, 6.0),
            )
        _save_figure(
            plt,
            staged_by_final[beeswarm_final],
            width=10.0,
            height=6.0,
        )

        for item, importance_row in zip(
            dependence_files,
            ranked_features.itertuples(index=False),
            strict=True,
        ):
            feature = str(importance_row.feature)
            feature_index = names.index(feature)
            dependence_final = figure_path / str(item["filename"])
            plt.close("all")
            with _deterministic_plotting():
                shap.plots.scatter(
                    explanations[:, feature_index],
                    color="#1E88E5",
                    x_jitter=0,
                    title=f"Dependence: {feature}",
                    show=False,
                )
            _save_figure(
                plt,
                staged_by_final[dependence_final],
                width=8.0,
                height=5.0,
            )

        sample_lookup = {
            int(original): position for position, original in enumerate(sample_positions)
        }
        local_explanations: dict[str, object] = {action: None for action in POLICY_ACTIONS}
        for action, scored_index in selected_indices.items():
            original_position = int(scored.index.get_loc(scored_index))
            explanation_position = sample_lookup[original_position]
            local_values = shap_values[explanation_position]
            contribution_indices = sorted(
                range(feature_count),
                key=lambda index: (-abs(float(local_values[index])), names[index]),
            )[: min(5, feature_count)]
            contributions = [
                {
                    "feature": names[index],
                    "feature_value": float(sample[explanation_position, index]),
                    "shap_value": float(local_values[index]),
                }
                for index in contribution_indices
            ]
            filename = waterfall_files[action]
            waterfall_final = figure_path / filename
            plt.close("all")
            with _deterministic_plotting():
                shap.plots.waterfall(
                    explanations[explanation_position],
                    max_display=min(10, feature_count),
                    show=False,
                )
            _save_figure(
                plt,
                staged_by_final[waterfall_final],
                width=9.0,
                height=6.0,
            )
            row_identifier: dict[str, object] | None = None
            if row_identifier_column is not None:
                row_identifier = {
                    "column": row_identifier_column,
                    "value": _json_scalar(scored.loc[scored_index, row_identifier_column]),
                }
            base_value = float(base_values[explanation_position])
            local_explanations[action] = {
                "policy_action": action,
                "scored_index": scored_index,
                "row_identifier": row_identifier,
                "calibrated_probability": float(scored.loc[scored_index, "probability"]),
                "base_value": base_value,
                "base_model_raw_output": float(base_value + np.sum(local_values)),
                "top_contributions": contributions,
                "waterfall": filename,
            }

        payload: dict[str, object] = {
            "schema_version": "1.0",
            "explanation_model": {
                "artifact": model_artifact_name,
                "source": "frozen_uncalibrated_lightgbm",
                "objective": objective,
                "sigmoid": sigmoid,
                "output_space": "raw_model_output",
                "units": "log_odds",
                "calibrated_probability_source": "frozen_calibrated_model",
                "calibration_note": (
                    "SHAP values explain the frozen base LightGBM score, not the "
                    "post-calibration probability."
                ),
            },
            "sample": {
                "random_seed": SHAP_RANDOM_SEED,
                "maximum_rows": MAX_SHAP_ROWS,
                "test_rows": row_count,
                "explained_rows": int(len(sample_positions)),
            },
            "feature_names": names,
            "global_top_features": global_top_features,
            "local_explanations": local_explanations,
            "files": {
                "importance": SHAP_IMPORTANCE_FILENAME,
                "payload": SHAP_PAYLOAD_FILENAME,
                "beeswarm": SHAP_BEESWARM_FILENAME,
                "dependence": dependence_files,
                "waterfalls": waterfall_files,
            },
        }
        _write_payload(staged_by_final[payload_final], payload)
        _validate_staged_files(list(staged_by_final.values()))
    except Exception as render_error:
        cleanup_notes = _cleanup_temporary_files(list(staged_by_final.values()))
        _add_exception_notes(render_error, cleanup_notes)
        raise
    finally:
        plt.close("all")

    _publish_staged_outputs(
        staged_by_final=staged_by_final,
        backup_by_final=backup_by_final,
        known_final_paths=known_final_paths,
        current_non_payload_finals=current_non_payload_finals,
        obsolete_final_paths=obsolete_final_paths,
        payload_final=payload_final,
    )
    return payload
