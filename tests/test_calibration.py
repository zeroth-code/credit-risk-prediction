import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import credit_risk.calibration as calibration_module
from credit_risk.calibration import expected_calibration_error, select_calibration


def _load_train_script(module_name: str) -> object:
    script_path = Path("scripts/train.py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_calibration_returns_best_brier_method() -> None:
    y = np.array([0, 0, 1, 1])
    candidates = {
        "uncalibrated": np.array([0.2, 0.3, 0.7, 0.8]),
        "sigmoid": np.array([0.1, 0.2, 0.8, 0.9]),
    }

    result = select_calibration(y, candidates)

    assert result.method == "sigmoid"
    assert result.scores["sigmoid"] < result.scores["uncalibrated"]


def test_expected_calibration_error_weights_equal_width_bins() -> None:
    y = np.array([0, 1, 1])
    probabilities = np.array([0.1, 0.8, 1.0])

    result = expected_calibration_error(y, probabilities, bins=2)

    assert result == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.array([]), "y_true.*non-empty"),
        (np.array([[0, 1]]), "y_true.*one-dimensional"),
        (np.array([0, 2]), "y_true.*0 or 1"),
        (np.array(["0", "1"]), "y_true.*0 or 1"),
    ],
)
def test_expected_calibration_error_rejects_invalid_targets(
    target: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(target, np.array([0.2, 0.8]))


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([]), "probabilities.*non-empty"),
        (np.array([[0.2, 0.8]]), "probabilities.*one-dimensional"),
        (np.array(["not-a-number"]), "probabilities.*numeric"),
        (np.array([np.nan]), "probabilities.*finite"),
        (np.array([np.inf]), "probabilities.*finite"),
        (np.array([-0.1]), "probabilities.*between 0 and 1"),
        (np.array([1.1]), "probabilities.*between 0 and 1"),
    ],
)
def test_expected_calibration_error_rejects_invalid_probabilities(
    probabilities: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(np.array([0]), probabilities)


def test_expected_calibration_error_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error(np.array([0, 1]), np.array([0.2]))


@pytest.mark.parametrize("bins", [0, -1, True, 1.5, "10"])
def test_expected_calibration_error_rejects_invalid_bins(bins: object) -> None:
    with pytest.raises(ValueError, match="bins.*positive int"):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            bins=bins,  # type: ignore[arg-type]
        )


def test_select_calibration_rejects_no_candidates() -> None:
    with pytest.raises(ValueError, match="candidates.*non-empty"):
        select_calibration(np.array([0, 1]), {})


def test_select_calibration_validates_candidate_probabilities() -> None:
    with pytest.raises(ValueError, match="sigmoid probabilities.*same length"):
        select_calibration(
            np.array([0, 1]),
            {"sigmoid": np.array([0.2])},
        )


def test_select_calibration_uses_candidate_order_for_brier_ties() -> None:
    result = select_calibration(
        np.array([0, 1]),
        {
            "sigmoid": np.array([0.25, 0.75]),
            "uncalibrated": np.array([0.25, 0.75]),
        },
    )

    assert result.method == "sigmoid"
    assert list(result.scores) == ["sigmoid", "uncalibrated"]


def test_load_partitions_reads_calibration_without_reading_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_script = _load_train_script("train_calibration_partitions")
    frames = {
        name: pd.DataFrame({"bad": [0, 1], "amount": [100.0, 200.0]})
        for name in ("train", "validation", "calibration")
    }
    observed: list[str] = []

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        partition_name = Path(path).stem
        observed.append(partition_name)
        if partition_name == "test":
            raise AssertionError("test partition must not be read during training")
        return frames[partition_name]

    monkeypatch.setattr(train_script.pd, "read_parquet", fake_read_parquet)

    train, validation, calibration = train_script.load_partitions(tmp_path, ["amount"])

    assert train is frames["train"]
    assert validation is frames["validation"]
    assert calibration is frames["calibration"]
    assert observed == ["train", "validation", "calibration"]


def test_calibration_curve_frame_represents_empty_bins_and_probability_one() -> None:
    curve = calibration_module.calibration_curve_frame(
        np.array([0, 1]),
        {"uncalibrated": np.array([0.0, 1.0])},
        bins=3,
    )

    assert curve.columns.tolist() == [
        "method",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "sample_count",
        "mean_probability",
        "observed_default_rate",
    ]
    assert curve["method"].tolist() == ["uncalibrated"] * 3
    assert curve["bin_index"].tolist() == [0, 1, 2]
    assert curve["sample_count"].tolist() == [1, 0, 1]
    assert curve.loc[0, "mean_probability"] == pytest.approx(0.0)
    assert curve.loc[0, "observed_default_rate"] == pytest.approx(0.0)
    assert np.isnan(curve.loc[1, "mean_probability"])
    assert np.isnan(curve.loc[1, "observed_default_rate"])
    assert curve.loc[2, "bin_upper"] == pytest.approx(1.0)
    assert curve.loc[2, "mean_probability"] == pytest.approx(1.0)
    assert curve.loc[2, "observed_default_rate"] == pytest.approx(1.0)


class _RowProbabilityModel:
    def __init__(self, split: int, low: float = 0.4, high: float = 0.6) -> None:
        self.split = split
        self.low = low
        self.high = high

    def predict_proba(self, features: object) -> np.ndarray:
        row_ids = np.asarray(features)[:, 0].astype(int)
        probabilities = np.where(row_ids < self.split, self.low, self.high)
        return np.column_stack([1.0 - probabilities, probabilities])


class _RecordingCalibrator:
    fit_records: list[tuple[str, set[int]]] = []
    predict_records: list[tuple[str, set[int], list[int]]] = []

    def __init__(self, estimator: object, *, method: str, cv: object) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.labels: dict[int, int] = {}

    def fit(self, features: object, target: np.ndarray) -> "_RecordingCalibrator":
        row_ids = np.asarray(features)[:, 0].astype(int)
        self.labels = dict(
            zip(row_ids.tolist(), np.asarray(target).astype(int).tolist(), strict=True)
        )
        self.fit_records.append((self.method, set(row_ids.tolist())))
        return self

    def predict_proba(self, features: object) -> np.ndarray:
        row_ids = np.asarray(features)[:, 0].astype(int)
        self.predict_records.append((self.method, set(self.labels), row_ids.tolist()))
        if self.method == "isotonic":
            probabilities = np.array([self.labels.get(row_id, 0.5) for row_id in row_ids])
        else:
            probabilities = np.where(row_ids < 10, 0.2, 0.8)
        return np.column_stack([1.0 - probabilities, probabilities])


class _PassThroughCalibrator:
    def __init__(self, estimator: object, *, method: str, cv: object) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, features: object, target: np.ndarray) -> "_PassThroughCalibrator":
        return self

    def predict_proba(self, features: object) -> np.ndarray:
        return self.estimator.predict_proba(features)  # type: ignore[attr-defined,no-any-return]


def test_evaluate_calibration_uses_oof_probabilities_for_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingCalibrator.fit_records = []
    _RecordingCalibrator.predict_records = []
    monkeypatch.setattr(
        calibration_module,
        "CalibratedClassifierCV",
        _RecordingCalibrator,
        raising=False,
    )
    monkeypatch.setattr(calibration_module, "MIN_ISOTONIC_SAMPLES", 20, raising=False)
    monkeypatch.setattr(calibration_module, "MIN_ISOTONIC_CLASS_COUNT", 10, raising=False)
    features = np.arange(20, dtype=float).reshape(-1, 1)
    target = np.array([0] * 10 + [1] * 10)
    base_model = _RowProbabilityModel(split=10)

    evaluation = calibration_module.evaluate_calibration(
        base_model,
        features,
        target,
        methods=["uncalibrated", "sigmoid", "isotonic"],
        random_seed=23,
    )

    assert evaluation.selection.method == "sigmoid"
    assert evaluation.folds == 5
    assert evaluation.metrics["sigmoid"]["brier_score"] == pytest.approx(0.04)
    assert evaluation.metrics["isotonic"]["brier_score"] == pytest.approx(0.25)
    assert np.all(evaluation.probabilities["isotonic"] == 0.5)
    assert evaluation.curve["method"].drop_duplicates().tolist() == [
        "uncalibrated",
        "sigmoid",
        "isotonic",
    ]

    for method in ("sigmoid", "isotonic"):
        records = [record for record in _RecordingCalibrator.predict_records if record[0] == method]
        predicted_rows = [row_id for _, _, row_ids in records for row_id in row_ids]
        assert sorted(predicted_rows) == list(range(len(target)))
        assert len(predicted_rows) == len(set(predicted_rows))
        assert all(train_rows.isdisjoint(holdout_rows) for _, train_rows, holdout_rows in records)

    full_isotonic = _RecordingCalibrator(
        FrozenEstimator(base_model),
        method="isotonic",
        cv=None,
    ).fit(features, target)
    full_fit_probabilities = full_isotonic.predict_proba(features)[:, 1]
    assert brier_score_loss(target, full_fit_probabilities) == pytest.approx(0.0)
    assert evaluation.selection.method != "isotonic"


def test_evaluate_calibration_is_reproducible_and_does_not_modify_base_model() -> None:
    train_features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    train_target = np.array([0, 0, 0, 1, 1, 1])
    features = np.linspace(-4.0, 4.0, 40).reshape(-1, 1)
    target = (features[:, 0] >= 0.0).astype(int)
    base_model = LogisticRegression(random_state=11).fit(train_features, train_target)
    coefficients_before = base_model.coef_.copy()

    first = calibration_module.evaluate_calibration(
        base_model,
        features,
        target,
        methods=["uncalibrated", "sigmoid", "isotonic"],
        random_seed=29,
    )
    second = calibration_module.evaluate_calibration(
        base_model,
        features,
        target,
        methods=["uncalibrated", "sigmoid", "isotonic"],
        random_seed=29,
    )

    assert first.folds == 5
    np.testing.assert_array_equal(first.probabilities["sigmoid"], second.probabilities["sigmoid"])
    assert np.isfinite(first.probabilities["sigmoid"]).all()
    np.testing.assert_array_equal(base_model.coef_, coefficients_before)


def test_evaluate_calibration_skips_fitted_methods_when_minority_class_is_one() -> None:
    features = np.arange(4, dtype=float).reshape(-1, 1)
    target = np.array([0, 0, 0, 1])

    evaluation = calibration_module.evaluate_calibration(
        _RowProbabilityModel(split=3),
        features,
        target,
        methods=["uncalibrated", "sigmoid", "isotonic"],
        random_seed=31,
    )

    assert evaluation.folds == 1
    assert evaluation.selection.method == "uncalibrated"
    assert list(evaluation.probabilities) == ["uncalibrated"]
    assert evaluation.curve["method"].drop_duplicates().tolist() == ["uncalibrated"]
    assert evaluation.metrics["uncalibrated"]["status"] == "evaluated"
    for method in ("sigmoid", "isotonic"):
        assert evaluation.metrics[method]["status"] == "skipped"
        assert "at least 2 samples in each class" in evaluation.metrics[method]["skip_reason"]


@pytest.mark.parametrize(
    ("rows", "positives", "expected_status", "reason"),
    [
        (999, 50, "skipped", "at least 1000 calibration samples"),
        (1000, 49, "skipped", "at least 50 samples in each class"),
        (1000, 50, "evaluated", None),
    ],
)
def test_evaluate_calibration_applies_isotonic_sample_thresholds(
    rows: int,
    positives: int,
    expected_status: str,
    reason: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration_module,
        "CalibratedClassifierCV",
        _PassThroughCalibrator,
        raising=False,
    )
    features = np.arange(rows, dtype=float).reshape(-1, 1)
    target = np.array([0] * (rows - positives) + [1] * positives)

    evaluation = calibration_module.evaluate_calibration(
        _RowProbabilityModel(split=rows - positives, low=0.1, high=0.9),
        features,
        target,
        methods=["uncalibrated", "sigmoid", "isotonic"],
        random_seed=37,
    )

    assert evaluation.metrics["isotonic"]["status"] == expected_status
    if reason is None:
        assert "skip_reason" not in evaluation.metrics["isotonic"]
        assert "isotonic" in evaluation.probabilities
    else:
        assert reason in evaluation.metrics["isotonic"]["skip_reason"]
        assert "isotonic" not in evaluation.probabilities


def test_fit_calibrated_model_refits_selected_method_on_full_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingCalibrator.fit_records = []
    monkeypatch.setattr(
        calibration_module,
        "CalibratedClassifierCV",
        _RecordingCalibrator,
        raising=False,
    )
    features = np.arange(20, dtype=float).reshape(-1, 1)
    target = np.array([0] * 10 + [1] * 10)
    base_model = _RowProbabilityModel(split=10)

    calibrated = calibration_module.fit_calibrated_model(
        base_model,
        features,
        target,
        method="sigmoid",
    )

    assert isinstance(calibrated, _RecordingCalibrator)
    assert calibrated.method == "sigmoid"
    assert isinstance(calibrated.estimator, FrozenEstimator)
    assert calibrated.estimator.estimator is base_model
    assert _RecordingCalibrator.fit_records == [("sigmoid", set(range(20)))]
    train_indices, predict_indices = calibrated.cv[0]
    np.testing.assert_array_equal(train_indices, np.arange(20))
    np.testing.assert_array_equal(predict_indices, np.arange(20))


def test_fit_calibrated_model_returns_base_model_for_uncalibrated() -> None:
    base_model = _RowProbabilityModel(split=2)

    result = calibration_module.fit_calibrated_model(
        base_model,
        np.arange(4, dtype=float).reshape(-1, 1),
        np.array([0, 0, 1, 1]),
        method="uncalibrated",
    )

    assert result is base_model
