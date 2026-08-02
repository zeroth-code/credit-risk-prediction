import json
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

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
            raise ValueError(f"no scored example for action: {action}")
        median_probability = group["probability"].median()
        distances = (group["probability"] - median_probability).abs()
        tied = np.isclose(
            distances.to_numpy(dtype=float),
            float(distances.min()),
            rtol=1e-12,
            atol=1e-15,
        )
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


def _validate_explanation_model(estimator: object, feature_count: int) -> LGBMClassifier:
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
    return estimator


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
    model = _validate_explanation_model(estimator, feature_count)
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
    importance.to_csv(
        artifact_path / SHAP_IMPORTANCE_FILENAME,
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
        figure_path / SHAP_BEESWARM_FILENAME,
        width=10.0,
        height=6.0,
    )

    top_feature_count = min(5, feature_count)
    dependence_files: list[dict[str, object]] = []
    for rank, importance_row in enumerate(
        importance.itertuples(index=False),
        start=1,
    ):
        if rank > top_feature_count:
            break
        feature = str(importance_row.feature)
        feature_index = names.index(feature)
        filename = SHAP_DEPENDENCE_FILENAMES[rank - 1]
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
            figure_path / filename,
            width=8.0,
            height=5.0,
        )
        dependence_files.append({"feature": feature, "filename": filename})

    sample_lookup = {int(original): position for position, original in enumerate(sample_positions)}
    local_explanations: dict[str, object] = {}
    waterfall_files: dict[str, str] = {}
    base_values = np.asarray(explanations.base_values, dtype=float)
    for action in POLICY_ACTIONS:
        scored_index = selected_indices[action]
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
        filename = SHAP_WATERFALL_FILENAMES[action]
        plt.close("all")
        with _deterministic_plotting():
            shap.plots.waterfall(
                explanations[explanation_position],
                max_display=min(10, feature_count),
                show=False,
            )
        _save_figure(
            plt,
            figure_path / filename,
            width=9.0,
            height=6.0,
        )
        waterfall_files[action] = filename
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

    global_top_features = [
        {
            "rank": int(row.rank),
            "feature": str(row.feature),
            "mean_abs_shap": float(row.mean_abs_shap),
        }
        for row in importance.head(5).itertuples(index=False)
    ]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "explanation_model": {
            "artifact": model_artifact_name,
            "source": "frozen_uncalibrated_lightgbm",
            "output_space": "raw_model_output",
            "units": "log_odds",
            "calibrated_probability_source": "frozen_calibrated_model",
            "calibration_note": (
                "SHAP values explain the frozen base LightGBM score, not the post-calibration "
                "probability."
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
    _write_payload(artifact_path / SHAP_PAYLOAD_FILENAME, payload)
    plt.close("all")
    return payload
