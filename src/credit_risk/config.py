from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class DateWindow(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_date_order(self) -> "DateWindow":
        if self.start > self.end:
            raise ValueError("partition start must not be after partition end")
        return self


class CostScenario(BaseModel):
    lgd: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    review_cost: float = Field(ge=0)


class CostConfig(BaseModel):
    base: CostScenario
    lgd_values: list[float]
    margin_values: list[float]
    review_cost_values: list[float]


class ProjectConfig(BaseModel):
    random_seed: int
    raw_csv: Path
    processed_dir: Path
    artifact_dir: Path
    figure_dir: Path
    train: DateWindow
    validation: DateWindow
    calibration: DateWindow
    test: DateWindow
    loan_term: str
    good_statuses: list[str]
    bad_statuses: list[str]
    unresolved_statuses: list[str]
    calibration_methods: list[str]
    minimum_group_size: int = Field(ge=1)
    costs: CostConfig

    @model_validator(mode="after")
    def validate_partition_order(self) -> "ProjectConfig":
        windows = [self.train, self.validation, self.calibration, self.test]
        for left, right in zip(windows, windows[1:]):  # noqa: B905
            if left.end >= right.start:
                raise ValueError("date partitions must be ordered and non-overlapping")
        return self


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    return ProjectConfig.model_validate(payload)
