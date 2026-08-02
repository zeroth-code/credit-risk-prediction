"""Pin the documented probability-range claims to the committed release bundle.

The README and model card state the fitted sigmoid coefficients, the decision-function value
needed to reach the decline threshold, and the fact that nothing clamps the output. Those are
load-bearing explanations for why the policy declines nobody, so they must fail loudly if the
released calibrator is ever retrained into a different shape.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from credit_risk.demo import load_release_artifacts

RELEASE_DIR = Path(__file__).resolve().parents[1] / "artifacts/release"

# Documented in README.md ("Why predicted probabilities stay in a narrow band") and
# docs/model_card.md ("Output probability range").
DOCUMENTED_SLOPE = -1.0195
DOCUMENTED_INTERCEPT = 1.8076
DOCUMENTED_DECLINE_MARGIN = 1.576


@pytest.fixture(scope="module")
def calibrator() -> object:
    artifacts = load_release_artifacts(RELEASE_DIR)
    calibrated = artifacts.model.calibrated_classifiers_
    assert len(calibrated) == 1, "release bundle should hold a single refit sigmoid calibrator"
    return calibrated[0].calibrators[0]


def test_documented_sigmoid_coefficients_match_the_release(calibrator: object) -> None:
    assert calibrator.a_ == pytest.approx(DOCUMENTED_SLOPE, abs=5e-5)
    assert calibrator.b_ == pytest.approx(DOCUMENTED_INTERCEPT, abs=5e-5)


def test_calibrator_is_unbounded_so_the_narrow_range_is_not_a_clamp(calibrator: object) -> None:
    """The band is a property of the score distribution, not a cap in the code."""

    def probability(margin: float) -> float:
        return float(1.0 / (1.0 + np.exp(calibrator.a_ * margin + calibrator.b_)))

    assert probability(-20.0) < 1e-6
    assert probability(20.0) > 0.999
    assert probability(0.0) == pytest.approx(0.1409, abs=1e-3)


def test_decline_threshold_requires_the_documented_margin(calibrator: object) -> None:
    decline_at = json.loads((RELEASE_DIR / "policy.json").read_text())["decline_at"]
    required = (np.log(1.0 / decline_at - 1.0) - calibrator.b_) / calibrator.a_

    assert required == pytest.approx(DOCUMENTED_DECLINE_MARGIN, abs=5e-3)
