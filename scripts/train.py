import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.calibration import (  # noqa: E402
    evaluate_calibration,
    fit_calibrated_model,
)
from credit_risk.config import load_config  # noqa: E402
from credit_risk.features import (  # noqa: E402
    build_feature_frame,
    feature_columns,
    load_feature_dictionary,
    make_logistic_preprocessor,
    make_tree_preprocessor,
)
from credit_risk.metrics import binary_metrics  # noqa: E402
from credit_risk.training import (  # noqa: E402
    make_dummy_model,
    make_lightgbm_model,
    make_logistic_model,
    positive_class_weight,
    random_undersample_indices,
    run_lightgbm_study,
)


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


FEATURE_SETS = ("challenger", "full_underwriting")
CHALLENGER_LIGHTGBM_STRATEGIES = ("natural", "weighted", "undersampled")
BASE_CONFIG_PATH = _project_path("configs/base.yaml")
FEATURE_DICTIONARY_PATH = _project_path("configs/features.yaml")


def required_feature_columns(feature_dictionary: dict[str, object]) -> list[str]:
    required: list[str] = []
    for feature_set in FEATURE_SETS:
        section = feature_dictionary[feature_set]
        for column in section["numeric"] + section["categorical"]:  # type: ignore[index,operator]
            if column not in required:
                required.append(column)
    return required


def load_partitions(
    processed_dir: Path, required_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partitions: dict[str, pd.DataFrame] = {}
    for partition_name in ("train", "validation", "calibration"):
        path = processed_dir / f"{partition_name}.parquet"
        frame = pd.read_parquet(path)
        missing = [column for column in ["bad", *required_columns] if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{partition_name} partition missing required columns: {', '.join(missing)}"
            )
        partitions[partition_name] = frame
    return partitions["train"], partitions["validation"], partitions["calibration"]


def partition_target(frame: pd.DataFrame, *, partition_name: str) -> np.ndarray:
    target = frame["bad"].to_numpy(copy=True)
    try:
        positive_class_weight(target)
    except ValueError as exc:
        raise ValueError(f"{partition_name} bad target invalid: {exc}") from exc
    return target.astype(int, copy=False)


def build_feature_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_dictionary_path: str | Path = FEATURE_DICTIONARY_PATH,
) -> dict[str, dict[str, object]]:
    matrices: dict[str, dict[str, object]] = {}
    for feature_set in FEATURE_SETS:
        numeric_columns, categorical_columns = feature_columns(
            feature_set, path=feature_dictionary_path
        )
        selected_columns = numeric_columns + categorical_columns
        train_frame = build_feature_frame(train, selected_columns, path=feature_dictionary_path)
        validation_frame = build_feature_frame(
            validation, selected_columns, path=feature_dictionary_path
        )

        logistic_preprocessor = make_logistic_preprocessor(numeric_columns, categorical_columns)
        logistic_train = logistic_preprocessor.fit_transform(train_frame)
        logistic_validation = logistic_preprocessor.transform(validation_frame)

        tree_preprocessor = make_tree_preprocessor(numeric_columns, categorical_columns)
        tree_train = tree_preprocessor.fit_transform(train_frame)
        tree_validation = tree_preprocessor.transform(validation_frame)
        matrices[feature_set] = {
            "train_frame": train_frame,
            "validation_frame": validation_frame,
            "logistic_preprocessor": logistic_preprocessor,
            "logistic_train": logistic_train,
            "logistic_validation": logistic_validation,
            "tree_preprocessor": tree_preprocessor,
            "tree_train": tree_train,
            "tree_validation": tree_validation,
        }
    return matrices


def evaluate_model(
    model: object,
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
    imbalance_strategy: str,
) -> dict[str, object]:
    probabilities = model.fit(x_train, y_train).predict_proba(x_validation)[:, 1]  # type: ignore[attr-defined]
    return {
        "model": model_name,
        "feature_set": feature_set,
        "imbalance_strategy": imbalance_strategy,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_validation)),
        "train_prevalence": float(np.mean(y_train)),
        "validation_prevalence": float(np.mean(y_validation)),
        **binary_metrics(y_validation, probabilities, threshold=0.5),
    }


def run_experiments(
    matrices: dict[str, dict[str, object]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    *,
    random_seed: int,
    scale_pos_weight: float,
    undersample_indices: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    challenger = matrices["challenger"]
    records.append(
        evaluate_model(
            make_dummy_model(),
            challenger["logistic_train"],
            y_train,
            challenger["logistic_validation"],
            y_validation,
            model_name="dummy",
            feature_set="challenger",
            imbalance_strategy="natural",
        )
    )

    for feature_set in FEATURE_SETS:
        feature_matrices = matrices[feature_set]
        for class_weight, strategy in ((None, "natural"), ("balanced", "weighted")):
            records.append(
                evaluate_model(
                    make_logistic_model(
                        class_weight=class_weight,
                        random_seed=random_seed,
                    ),
                    feature_matrices["logistic_train"],
                    y_train,
                    feature_matrices["logistic_validation"],
                    y_validation,
                    model_name="logistic_regression",
                    feature_set=feature_set,
                    imbalance_strategy=strategy,
                )
            )

    records.append(
        evaluate_model(
            make_lightgbm_model(random_seed=random_seed, scale_pos_weight=1.0),
            challenger["tree_train"],
            y_train,
            challenger["tree_validation"],
            y_validation,
            model_name="lightgbm",
            feature_set="challenger",
            imbalance_strategy="natural",
        )
    )
    records.append(
        evaluate_model(
            make_lightgbm_model(
                random_seed=random_seed,
                scale_pos_weight=scale_pos_weight,
            ),
            challenger["tree_train"],
            y_train,
            challenger["tree_validation"],
            y_validation,
            model_name="lightgbm",
            feature_set="challenger",
            imbalance_strategy="weighted",
        )
    )
    records.append(
        evaluate_model(
            make_lightgbm_model(random_seed=random_seed, scale_pos_weight=1.0),
            challenger["tree_train"][undersample_indices],  # type: ignore[index]
            y_train[undersample_indices],
            challenger["tree_validation"],
            y_validation,
            model_name="lightgbm",
            feature_set="challenger",
            imbalance_strategy="undersampled",
        )
    )

    full_underwriting = matrices["full_underwriting"]
    records.append(
        evaluate_model(
            make_lightgbm_model(random_seed=random_seed, scale_pos_weight=1.0),
            full_underwriting["tree_train"],
            y_train,
            full_underwriting["tree_validation"],
            y_validation,
            model_name="lightgbm",
            feature_set="full_underwriting",
            imbalance_strategy="natural",
        )
    )
    return records


def select_challenger_lightgbm_strategy(records: list[dict[str, object]]) -> str:
    by_strategy = {
        str(record["imbalance_strategy"]): record
        for record in records
        if record["model"] == "lightgbm" and record["feature_set"] == "challenger"
    }
    missing = [
        strategy for strategy in CHALLENGER_LIGHTGBM_STRATEGIES if strategy not in by_strategy
    ]
    if missing:
        raise ValueError(f"missing challenger LightGBM strategies: {', '.join(missing)}")
    return max(
        CHALLENGER_LIGHTGBM_STRATEGIES,
        key=lambda strategy: float(by_strategy[strategy]["average_precision"]),
    )


def tuning_training_data(
    challenger: dict[str, object],
    y_train: np.ndarray,
    *,
    selected_strategy: str,
    scale_pos_weight: float,
    undersample_indices: np.ndarray,
) -> tuple[object, np.ndarray, float]:
    if selected_strategy == "natural":
        return challenger["tree_train"], y_train, 1.0
    if selected_strategy == "weighted":
        return challenger["tree_train"], y_train, scale_pos_weight
    if selected_strategy == "undersampled":
        return (
            challenger["tree_train"][undersample_indices],  # type: ignore[index]
            y_train[undersample_indices],
            1.0,
        )
    raise ValueError(f"unknown challenger LightGBM strategy: {selected_strategy}")


def _json_native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_training_artifacts(
    artifact_dir: Path,
    *,
    model: object,
    preprocessor: object,
    metrics_payload: dict[str, object],
    study: optuna.Study,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "uncalibrated_model.joblib")
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    with (artifact_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(_json_native(metrics_payload), file, ensure_ascii=False, indent=2)
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials.to_csv(artifact_dir / "tuning_trials.csv", index=False, encoding="utf-8")


def save_calibration_artifacts(
    artifact_dir: Path,
    *,
    calibrated_model: object,
    metrics_payload: dict[str, object],
    curve: pd.DataFrame,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, artifact_dir / "calibrated_model.joblib")
    with (artifact_dir / "calibration_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(_json_native(metrics_payload), file, ensure_ascii=False, indent=2)
    curve.to_csv(
        artifact_dir / "calibration_curve.csv",
        index=False,
        encoding="utf-8",
    )


def main(n_trials: int = 30) -> None:
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        raise ValueError("n_trials must be a positive int and must not be a bool")

    config = load_config(BASE_CONFIG_PATH)
    np.random.seed(config.random_seed)
    feature_dictionary = load_feature_dictionary(FEATURE_DICTIONARY_PATH)
    required_columns = required_feature_columns(feature_dictionary)
    processed_dir = _project_path(config.processed_dir)
    artifact_dir = _project_path(config.artifact_dir)
    train, validation, calibration = load_partitions(processed_dir, required_columns)
    matrices = build_feature_matrices(
        train,
        validation,
        feature_dictionary_path=FEATURE_DICTIONARY_PATH,
    )

    y_train = partition_target(train, partition_name="train")
    y_validation = partition_target(validation, partition_name="validation")
    y_calibration = partition_target(calibration, partition_name="calibration")
    scale_pos_weight = positive_class_weight(y_train)
    undersample_indices = random_undersample_indices(y_train, config.random_seed)
    experiments = run_experiments(
        matrices,
        y_train,
        y_validation,
        random_seed=config.random_seed,
        scale_pos_weight=scale_pos_weight,
        undersample_indices=undersample_indices,
    )
    selected_strategy = select_challenger_lightgbm_strategy(experiments)

    challenger = matrices["challenger"]
    tuning_x_train, tuning_y_train, tuning_scale_pos_weight = tuning_training_data(
        challenger,
        y_train,
        selected_strategy=selected_strategy,
        scale_pos_weight=scale_pos_weight,
        undersample_indices=undersample_indices,
    )
    study = run_lightgbm_study(
        tuning_x_train,
        tuning_y_train,
        challenger["tree_validation"],
        y_validation,
        n_trials=n_trials,
        random_seed=config.random_seed,
        scale_pos_weight=tuning_scale_pos_weight,
    )
    best_params = dict(study.best_params)
    final_model = make_lightgbm_model(
        random_seed=config.random_seed,
        scale_pos_weight=tuning_scale_pos_weight,
        **best_params,
    )
    final_model.fit(tuning_x_train, tuning_y_train)
    tuned_probabilities = final_model.predict_proba(challenger["tree_validation"])[:, 1]
    tuned_metrics = binary_metrics(y_validation, tuned_probabilities, threshold=0.5)

    challenger_config = feature_dictionary["challenger"]
    challenger_columns = challenger_config["numeric"] + challenger_config["categorical"]
    calibration_frame = build_feature_frame(
        calibration,
        challenger_columns,
        path=FEATURE_DICTIONARY_PATH,
    )
    calibration_matrix = challenger["tree_preprocessor"].transform(calibration_frame)  # type: ignore[attr-defined]
    calibration_evaluation = evaluate_calibration(
        final_model,
        calibration_matrix,
        y_calibration,
        methods=config.calibration_methods,
        random_seed=config.random_seed,
    )
    selected_method = calibration_evaluation.selection.method
    calibrated_model = fit_calibrated_model(
        final_model,
        calibration_matrix,
        y_calibration,
        method=selected_method,
    )
    if selected_method == "uncalibrated":
        artifact_metadata = {
            "method": selected_method,
            "fit_protocol": "base_model_train_fit",
            "fit_partition": "train",
        }
    else:
        artifact_metadata = {
            "method": selected_method,
            "fit_protocol": "full_calibration_refit",
            "fit_partition": "calibration",
        }
    calibration_metrics_payload: dict[str, object] = {
        "selected_method": selected_method,
        "methods": calibration_evaluation.metrics,
        "calibration_samples": int(len(y_calibration)),
        "calibration_prevalence": float(np.mean(y_calibration)),
        "ece_bins": 10,
        "evaluation_protocol": "stratified_oof",
        "evaluation_partition": "calibration",
        "folds": calibration_evaluation.folds,
        "random_seed": int(config.random_seed),
        "artifact": artifact_metadata,
    }

    metrics_payload: dict[str, object] = {
        "primary_feature_set": "challenger",
        "experiments": experiments,
        "selected_strategy": selected_strategy,
        "positive_class_weight": scale_pos_weight,
        "tuned_best_params": best_params,
        "tuned_metrics": tuned_metrics,
        "random_seed": int(config.random_seed),
        "n_trials": n_trials,
    }
    save_training_artifacts(
        artifact_dir,
        model=final_model,
        preprocessor=challenger["tree_preprocessor"],
        metrics_payload=metrics_payload,
        study=study,
    )
    save_calibration_artifacts(
        artifact_dir,
        calibrated_model=calibrated_model,
        metrics_payload=calibration_metrics_payload,
        curve=calibration_evaluation.curve,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train uncalibrated credit-risk baselines")
    parser.add_argument(
        "--n-trials",
        type=_positive_int,
        default=30,
        help="number of Optuna trials (default: 30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(n_trials=arguments.n_trials)
