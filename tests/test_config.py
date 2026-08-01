from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from credit_risk.config import CostConfig, CostScenario, DateWindow, ProjectConfig, load_config


def _cost_config_payload() -> dict[str, object]:
    return {
        "base": {"lgd": 0.60, "margin": 0.05, "review_cost": 30.0},
        "lgd_values": [0.60],
        "margin_values": [0.05],
        "review_cost_values": [30.0],
    }


def _project_config_payload() -> dict[str, object]:
    return {
        "random_seed": 42,
        "raw_csv": "data/raw/accepted_2007_to_2018Q4.csv",
        "processed_dir": "data/processed",
        "artifact_dir": "artifacts",
        "figure_dir": "reports/figures",
        "train": {"start": "2011-01-01", "end": "2013-12-31"},
        "validation": {"start": "2014-01-01", "end": "2014-06-30"},
        "calibration": {"start": "2014-07-01", "end": "2014-12-31"},
        "test": {"start": "2015-01-01", "end": "2015-12-31"},
        "loan_term": "36 months",
        "good_statuses": ["Fully Paid"],
        "bad_statuses": ["Charged Off", "Default"],
        "unresolved_statuses": ["Current"],
        "calibration_methods": ["sigmoid"],
        "minimum_group_size": 200,
        "costs": _cost_config_payload(),
    }


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
    payload = _project_config_payload()
    payload["validation"] = {"start": "2013-12-31", "end": "2014-06-30"}

    with pytest.raises(
        ValidationError, match="date partitions must be ordered and non-overlapping"
    ) as error:
        ProjectConfig.model_validate(payload)

    assert "2013-12-31" in str(error.value)


def test_cost_scenario_rejects_invalid_costs() -> None:
    with pytest.raises(ValidationError):
        CostScenario(lgd=1.01, margin=0.05, review_cost=30.0)


@pytest.mark.parametrize("field", ["lgd", "margin", "review_cost"])
def test_cost_scenario_rejects_boolean_costs(field: str) -> None:
    payload = {"lgd": 0.60, "margin": 0.05, "review_cost": 30.0}
    payload[field] = True

    with pytest.raises(ValidationError):
        CostScenario.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (DateWindow, {"start": "2011-01-01", "end": "2011-12-31", "extra": True}),
        (CostScenario, {"lgd": 0.60, "margin": 0.05, "review_cost": 30.0, "extra": True}),
        (CostConfig, {**_cost_config_payload(), "extra": True}),
        (ProjectConfig, {**_project_config_payload(), "extra": True}),
    ],
)
def test_config_models_forbid_unknown_fields(
    model: type[BaseModel], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("lgd_values", []),
        ("lgd_values", [-0.01]),
        ("lgd_values", [1.01]),
        ("lgd_values", [float("nan")]),
        ("margin_values", []),
        ("margin_values", [-0.01]),
        ("margin_values", [1.01]),
        ("margin_values", [float("nan")]),
        ("review_cost_values", []),
        ("review_cost_values", [-0.01]),
        ("review_cost_values", [float("nan")]),
    ],
)
def test_cost_config_rejects_invalid_sensitivity_values(field: str, values: list[float]) -> None:
    payload = _cost_config_payload()
    payload[field] = values

    with pytest.raises(ValidationError):
        CostConfig.model_validate(payload)


@pytest.mark.parametrize("field", ["lgd_values", "margin_values", "review_cost_values"])
def test_cost_config_rejects_boolean_sensitivity_values(field: str) -> None:
    payload = _cost_config_payload()
    payload[field] = [True]

    with pytest.raises(ValidationError):
        CostConfig.model_validate(payload)


@pytest.mark.parametrize("field", ["good_statuses", "bad_statuses"])
def test_project_config_requires_nonempty_labeled_statuses(field: str) -> None:
    payload = _project_config_payload()
    payload[field] = []

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize("field", ["good_statuses", "bad_statuses", "unresolved_statuses"])
@pytest.mark.parametrize("status", ["", "   "])
def test_project_config_rejects_blank_statuses(field: str, status: str) -> None:
    payload = _project_config_payload()
    payload[field] = [status]

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "statuses"),
    [
        ("good_statuses", ["Fully Paid", "Fully Paid"]),
        ("bad_statuses", ["Default", "Default"]),
        ("unresolved_statuses", ["Current", "Current"]),
    ],
)
def test_project_config_rejects_duplicate_statuses(field: str, statuses: list[str]) -> None:
    payload = _project_config_payload()
    payload[field] = statuses

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "statuses"),
    [
        ("good_statuses", ["Fully Paid", "Current"]),
        ("bad_statuses", ["Default", "Current"]),
        ("unresolved_statuses", ["Current", "Default"]),
    ],
)
def test_project_config_rejects_overlapping_status_groups(field: str, statuses: list[str]) -> None:
    payload = _project_config_payload()
    payload[field] = statuses

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize("methods", [[], ["invalid"], ["sigmoid", "sigmoid"]])
def test_project_config_rejects_invalid_calibration_methods(methods: list[str]) -> None:
    payload = _project_config_payload()
    payload["calibration_methods"] = methods

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


def test_project_config_rejects_boolean_random_seed() -> None:
    payload = _project_config_payload()
    payload["random_seed"] = True

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)


def test_project_config_rejects_boolean_minimum_group_size() -> None:
    payload = _project_config_payload()
    payload["minimum_group_size"] = True

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)
