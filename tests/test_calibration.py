import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

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
    train_script = _load_train_script("train_calibration_curve")

    curve = train_script.calibration_curve_frame(
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


def test_fit_calibration_candidates_freezes_fitted_model() -> None:
    train_script = _load_train_script("train_fit_calibration")
    x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])
    x_calibration = np.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    y_calibration = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model = LogisticRegression(random_state=13).fit(x_train, y_train)
    coefficients_before = model.coef_.copy()

    estimators, candidates = train_script.fit_calibration_candidates(
        model,
        x_calibration,
        y_calibration,
        methods=["uncalibrated", "sigmoid", "isotonic"],
    )

    assert list(estimators) == ["uncalibrated", "sigmoid", "isotonic"]
    assert list(candidates) == ["uncalibrated", "sigmoid", "isotonic"]
    assert estimators["uncalibrated"] is model
    for method in ("sigmoid", "isotonic"):
        calibrated = estimators[method]
        assert isinstance(calibrated, CalibratedClassifierCV)
        assert isinstance(calibrated.estimator, FrozenEstimator)
        assert calibrated.estimator.estimator is model
        probabilities = candidates[method]
        assert probabilities.shape == (len(y_calibration),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_array_equal(model.coef_, coefficients_before)


def test_fit_calibration_candidates_requires_all_configured_methods() -> None:
    train_script = _load_train_script("train_missing_calibration_method")
    model = LogisticRegression().fit(
        np.array([[-1.0], [1.0]]),
        np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="missing required calibration methods: isotonic"):
        train_script.fit_calibration_candidates(
            model,
            np.array([[-0.5], [0.5]]),
            np.array([0, 1]),
            methods=["uncalibrated", "sigmoid"],
        )
