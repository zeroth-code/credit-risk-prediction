from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

DEFAULT_FEATURE_PATH = Path("configs/features.yaml")
FeatureName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, strict=True, min_length=1),
]


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


class _FeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FeatureSection(_FeatureModel):
    numeric: list[FeatureName] = Field(min_length=1)
    categorical: list[FeatureName] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_feature_names(self) -> "_FeatureSection":
        for field_name, values in (
            ("numeric", self.numeric),
            ("categorical", self.categorical),
        ):
            duplicates = _duplicate_values(values)
            if duplicates:
                raise ValueError(
                    f"{field_name} contains duplicate feature names: {', '.join(duplicates)}"
                )

        categorical = set(self.categorical)
        overlap = [value for value in self.numeric if value in categorical]
        if overlap:
            raise ValueError(f"numeric/categorical overlap: {', '.join(overlap)}")
        return self


class _FeatureDictionary(_FeatureModel):
    challenger: _FeatureSection
    full_underwriting: _FeatureSection
    post_origination: list[FeatureName] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_post_origination(self) -> "_FeatureDictionary":
        duplicates = _duplicate_values(self.post_origination)
        if duplicates:
            raise ValueError(
                f"post_origination contains duplicate feature names: {', '.join(duplicates)}"
            )
        return self


def load_feature_dictionary(path: str | Path = DEFAULT_FEATURE_PATH) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as feature_file:
        payload = yaml.safe_load(feature_file)
    return _FeatureDictionary.model_validate(payload).model_dump()


def prohibited_columns(path: str | Path = DEFAULT_FEATURE_PATH) -> set[str]:
    dictionary = load_feature_dictionary(path)
    return set(dictionary["post_origination"])


def feature_columns(
    feature_set: str,
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> tuple[list[str], list[str]]:
    if feature_set not in {"challenger", "full_underwriting"}:
        raise ValueError(f"unknown feature set: {feature_set}")

    configured = load_feature_dictionary(path)[feature_set]
    return list(configured["numeric"]), list(configured["categorical"])


def build_feature_frame(
    frame: pd.DataFrame,
    selected_columns: list[str],
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> pd.DataFrame:
    if not selected_columns:
        raise ValueError("selected_columns must contain at least one column")

    duplicates = _duplicate_values(selected_columns)
    if duplicates:
        raise ValueError(f"duplicate selected columns: {', '.join(duplicates)}")

    blocked = [column for column in selected_columns if column in prohibited_columns(path)]
    if blocked:
        raise ValueError(f"prohibited columns: {', '.join(blocked)}")

    missing = [column for column in selected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    return frame.loc[:, selected_columns].copy()
