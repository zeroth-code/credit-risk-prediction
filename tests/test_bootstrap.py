import numpy as np
import pandas as pd
import pytest

import credit_risk.metrics as metrics
from credit_risk.metrics import bootstrap_metric


def test_bootstrap_metric_is_reproducible() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    first = bootstrap_metric(y, p, metric_name="roc_auc", samples=100, random_seed=42)
    second = bootstrap_metric(y, p, metric_name="roc_auc", samples=100, random_seed=42)
    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]


@pytest.mark.parametrize("metric_name", ["roc-auc", "pr_auc", "log_loss", "ROC_AUC"])
def test_bootstrap_metric_rejects_unsupported_metric(metric_name: str) -> None:
    with pytest.raises(ValueError, match="metric_name.*roc_auc.*average_precision.*brier_score"):
        bootstrap_metric(
            [0, 1],
            [0.2, 0.8],
            metric_name=metric_name,
            samples=10,
            random_seed=42,
        )


@pytest.mark.parametrize(
    ("y_true", "probabilities", "message"),
    [
        ([[0], [1]], [0.2, 0.8], "y_true.*one-dimensional"),
        ([0, 1], [[0.2], [0.8]], "probabilities.*one-dimensional"),
        ([], [], "non-empty"),
        ([0, 1], [0.2], "same length"),
        ([0, 2], [0.2, 0.8], "y_true.*0 or 1"),
        ([0, 0], [0.2, 0.8], "both classes"),
        ([1, 1], [0.2, 0.8], "both classes"),
    ],
)
def test_bootstrap_metric_rejects_invalid_shapes_and_targets(
    y_true: object,
    probabilities: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_metric(
            y_true,  # type: ignore[arg-type]
            probabilities,  # type: ignore[arg-type]
            metric_name="roc_auc",
            samples=10,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "y_true",
    [
        True,
        np.bool_(False),
        np.array([False, True]),
        np.array([0, True], dtype=object),
        np.array([0, pd.NA], dtype=object),
        np.array([0, None], dtype=object),
        np.array([0, np.nan], dtype=object),
    ],
)
def test_bootstrap_metric_rejects_boolean_or_missing_targets(y_true: object) -> None:
    with pytest.raises(ValueError, match="y_true"):
        bootstrap_metric(
            y_true,  # type: ignore[arg-type]
            [0.2, 0.8],
            metric_name="roc_auc",
            samples=10,
            random_seed=42,
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (True, "probabilities"),
        (np.bool_(False), "probabilities"),
        (np.array([False, True]), "probabilities.*boolean"),
        (np.array([0.2, True], dtype=object), "probabilities.*boolean"),
        (np.array([0.2, pd.NA], dtype=object), "probabilities.*numeric"),
        (np.array([0.2, None], dtype=object), "probabilities.*numeric"),
        (np.array([0.2, np.nan], dtype=object), "probabilities.*finite"),
        ([0.2, np.inf], "probabilities.*finite"),
        ([0.2, -0.01], "probabilities.*between 0 and 1"),
        ([0.2, 1.01], "probabilities.*between 0 and 1"),
    ],
)
def test_bootstrap_metric_rejects_invalid_probabilities(
    probabilities: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_metric(
            [0, 1],
            probabilities,  # type: ignore[arg-type]
            metric_name="roc_auc",
            samples=10,
            random_seed=42,
        )


@pytest.mark.parametrize("samples", [True, False, 0, -1, 1.0, np.int64(10), "10"])
def test_bootstrap_metric_rejects_invalid_samples(samples: object) -> None:
    with pytest.raises(ValueError, match="samples.*positive int.*bool"):
        bootstrap_metric(
            [0, 1],
            [0.2, 0.8],
            metric_name="roc_auc",
            samples=samples,  # type: ignore[arg-type]
            random_seed=42,
        )


@pytest.mark.parametrize("random_seed", [True, False, 1.0, np.int64(42), "42"])
def test_bootstrap_metric_rejects_invalid_random_seed(random_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed.*int.*bool"):
        bootstrap_metric(
            [0, 1],
            [0.2, 0.8],
            metric_name="roc_auc",
            samples=10,
            random_seed=random_seed,  # type: ignore[arg-type]
        )


def test_bootstrap_metric_stratifies_every_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_class_counts: list[tuple[int, int]] = []

    def recording_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        del probabilities
        observed_class_counts.append((int(np.sum(y_true == 0)), int(np.sum(y_true == 1))))
        return 0.5

    monkeypatch.setattr(metrics, "roc_auc_score", recording_roc_auc)

    result = bootstrap_metric(
        [0, 0, 0, 0, 0, 1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
        metric_name="roc_auc",
        samples=25,
        random_seed=7,
    )

    assert observed_class_counts == [(5, 1)] * 26
    assert result == {"estimate": 0.5, "lower": 0.5, "upper": 0.5}


@pytest.mark.parametrize("metric_name", ["roc_auc", "average_precision", "brier_score"])
def test_bootstrap_metric_returns_python_floats(metric_name: str) -> None:
    result = bootstrap_metric(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        metric_name=metric_name,
        samples=10,
        random_seed=42,
    )

    assert set(result) == {"estimate", "lower", "upper"}
    assert all(type(value) is float for value in result.values())
