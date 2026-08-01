from datetime import date

import pytest
from pydantic import ValidationError

from credit_risk.config import CostScenario, DateWindow, ProjectConfig, load_config


def test_load_config_has_ordered_partitions() -> None:
    config = load_config("configs/base.yaml")
    assert config.random_seed == 42
    assert config.train.start == date(2011, 1, 1)
    assert config.train.end < config.validation.start
    assert config.validation.end < config.calibration.start
    assert config.calibration.end < config.test.start
    assert config.costs.base.lgd == 0.60
    assert config.costs.lgd_values == [0.40, 0.60, 0.80]
    assert config.costs.margin_values == [0.03, 0.05, 0.08]
    assert config.costs.review_cost_values == [15.0, 30.0, 60.0]


def test_date_window_rejects_reversed_dates() -> None:
    with pytest.raises(ValidationError, match="partition start must not be after partition end"):
        DateWindow(start=date(2014, 1, 2), end=date(2014, 1, 1))


def test_project_config_rejects_overlapping_partitions() -> None:
    with pytest.raises(
        ValidationError, match="date partitions must be ordered and non-overlapping"
    ):
        ProjectConfig.model_validate(
            {
                "random_seed": 42,
                "raw_csv": "data/raw/accepted_2007_to_2018Q4.csv",
                "processed_dir": "data/processed",
                "artifact_dir": "artifacts",
                "figure_dir": "reports/figures",
                "train": {"start": "2011-01-01", "end": "2013-12-31"},
                "validation": {"start": "2013-12-31", "end": "2014-06-30"},
                "calibration": {"start": "2014-07-01", "end": "2014-12-31"},
                "test": {"start": "2015-01-01", "end": "2015-12-31"},
                "loan_term": "36 months",
                "good_statuses": ["Fully Paid"],
                "bad_statuses": ["Charged Off", "Default"],
                "unresolved_statuses": ["Current"],
                "calibration_methods": ["sigmoid"],
                "minimum_group_size": 200,
                "costs": {
                    "base": {"lgd": 0.60, "margin": 0.05, "review_cost": 30.0},
                    "lgd_values": [0.60],
                    "margin_values": [0.05],
                    "review_cost_values": [30.0],
                },
            }
        )


def test_cost_scenario_rejects_invalid_costs() -> None:
    with pytest.raises(ValidationError):
        CostScenario(lgd=1.01, margin=0.05, review_cost=30.0)
