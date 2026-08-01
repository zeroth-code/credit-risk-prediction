from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

StrictFiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
Probability = Annotated[StrictFiniteFloat, Field(ge=0, le=1)]
NonnegativeCost = Annotated[StrictFiniteFloat, Field(ge=0)]
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateWindow(ConfigModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_date_order(self) -> "DateWindow":
        if self.start > self.end:
            raise ValueError("partition start must not be after partition end")
        return self


class CostScenario(ConfigModel):
    lgd: Probability
    margin: Probability
    review_cost: NonnegativeCost


class CostConfig(ConfigModel):
    base: CostScenario
    lgd_values: list[Probability] = Field(min_length=1)
    margin_values: list[Probability] = Field(min_length=1)
    review_cost_values: list[NonnegativeCost] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sensitivity_grid(self) -> "CostConfig":
        axes = (
            ("lgd_values", self.lgd_values, "lgd", self.base.lgd),
            ("margin_values", self.margin_values, "margin", self.base.margin),
            (
                "review_cost_values",
                self.review_cost_values,
                "review_cost",
                self.base.review_cost,
            ),
        )
        for values_field, values, base_field, base_value in axes:
            if len(values) != 3:
                raise ValueError(f"{values_field} must contain exactly 3 values")
            if len(values) != len(set(values)):
                raise ValueError(f"{values_field} values must be unique")
            if base_value not in values:
                raise ValueError(f"base.{base_field} must be included in {values_field}")
        return self


class ProjectConfig(ConfigModel):
    random_seed: StrictInt
    raw_csv: Path
    processed_dir: Path
    artifact_dir: Path
    figure_dir: Path
    train: DateWindow
    validation: DateWindow
    calibration: DateWindow
    test: DateWindow
    loan_term: str
    good_statuses: list[NonBlankString] = Field(min_length=1)
    bad_statuses: list[NonBlankString] = Field(min_length=1)
    unresolved_statuses: list[NonBlankString]
    calibration_methods: list[Literal["uncalibrated", "sigmoid", "isotonic"]] = Field(min_length=1)
    minimum_group_size: StrictInt = Field(ge=1)
    costs: CostConfig

    @model_validator(mode="after")
    def validate_partition_order(self) -> "ProjectConfig":
        windows = [self.train, self.validation, self.calibration, self.test]
        for left, right in zip(windows, windows[1:]):  # noqa: B905
            if left.end >= right.start:
                raise ValueError(
                    "date partitions must be ordered and non-overlapping: "
                    f"{left.end.isoformat()} is not before {right.start.isoformat()}"
                )

        status_groups = [self.good_statuses, self.bad_statuses, self.unresolved_statuses]
        if any(len(group) != len(set(group)) for group in status_groups):
            raise ValueError("status groups must not contain duplicate statuses")

        required_calibration_methods = {"uncalibrated", "sigmoid", "isotonic"}
        if (
            len(self.calibration_methods) != len(required_calibration_methods)
            or set(self.calibration_methods) != required_calibration_methods
        ):
            raise ValueError(
                "calibration methods must contain exactly uncalibrated, sigmoid, and isotonic"
            )

        good_statuses = set(self.good_statuses)
        bad_statuses = set(self.bad_statuses)
        unresolved_statuses = set(self.unresolved_statuses)
        if (
            good_statuses & bad_statuses
            or good_statuses & unresolved_statuses
            or bad_statuses & unresolved_statuses
        ):
            raise ValueError("good, bad, and unresolved status groups must be disjoint")
        return self


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    return ProjectConfig.model_validate(payload)
