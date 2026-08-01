import numpy as np
import pytest

from credit_risk.metrics import binary_metrics

EXPECTED_METRIC_KEYS = {
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "ks",
    "precision",
    "recall",
    "specificity",
    "tn",
    "fp",
    "fn",
    "tp",
}


def test_binary_metrics_returns_expected_values_and_python_floats() -> None:
    result = binary_metrics(
        y_true=[0, 0, 1, 1],
        probabilities=[0.1, 0.4, 0.6, 0.9],
        threshold=0.5,
    )

    assert set(result) == EXPECTED_METRIC_KEYS
    assert all(type(value) is float for value in result.values())
    assert result == pytest.approx(
        {
            "roc_auc": 1.0,
            "average_precision": 1.0,
            "brier_score": 0.085,
            "log_loss": 0.30809306971190853,
            "ks": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "specificity": 1.0,
            "tn": 2.0,
            "fp": 0.0,
            "fn": 0.0,
            "tp": 2.0,
        }
    )


def test_binary_metrics_threshold_is_inclusive() -> None:
    result = binary_metrics(
        y_true=[0, 1],
        probabilities=[0.49, 0.5],
        threshold=0.5,
    )

    assert result["tn"] == 1.0
    assert result["tp"] == 1.0


@pytest.mark.parametrize(
    ("y_true", "probabilities", "match"),
    [
        ([[0], [1]], [0.2, 0.8], "y_true.*one-dimensional"),
        ([0, 1], [[0.2], [0.8]], "probabilities.*one-dimensional"),
        ([], [], "non-empty"),
        ([0, 1], [0.2], "same length"),
        ([0, 2], [0.2, 0.8], "0 or 1"),
        ([0, 0], [0.2, 0.8], "both classes"),
    ],
)
def test_binary_metrics_rejects_invalid_shapes_and_targets(
    y_true: list[object], probabilities: list[object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        binary_metrics(y_true, probabilities, threshold=0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_probability", [np.nan, np.inf, -np.inf])
def test_binary_metrics_rejects_non_finite_probabilities(
    invalid_probability: float,
) -> None:
    with pytest.raises(ValueError, match="probabilities.*finite"):
        binary_metrics([0, 1], [0.2, invalid_probability], threshold=0.5)


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_binary_metrics_rejects_probabilities_outside_unit_interval(
    invalid_probability: float,
) -> None:
    with pytest.raises(ValueError, match="probabilities.*0 and 1"):
        binary_metrics([0, 1], [0.2, invalid_probability], threshold=0.5)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_binary_metrics_rejects_non_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold.*finite"):
        binary_metrics([0, 1], [0.2, 0.8], threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_binary_metrics_rejects_threshold_outside_unit_interval(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold.*0 and 1"):
        binary_metrics([0, 1], [0.2, 0.8], threshold=threshold)
