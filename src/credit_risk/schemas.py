import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

StrictFiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
NonemptyString = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CreditApplication(StrictSchema):
    loan_amnt: Annotated[StrictFiniteFloat, Field(gt=0, le=100_000)]
    annual_inc: Annotated[StrictFiniteFloat, Field(gt=0)]
    dti: Annotated[StrictFiniteFloat, Field(ge=0, le=100)]
    delinq_2yrs: Annotated[StrictFiniteFloat, Field(ge=0)]
    fico_range_low: Annotated[StrictFiniteFloat, Field(ge=300, le=850)]
    fico_range_high: Annotated[StrictFiniteFloat, Field(ge=300, le=850)]
    inq_last_6mths: Annotated[StrictFiniteFloat, Field(ge=0)]
    open_acc: Annotated[StrictFiniteFloat, Field(ge=0)]
    pub_rec: Annotated[StrictFiniteFloat, Field(ge=0)]
    revol_bal: Annotated[StrictFiniteFloat, Field(ge=0)]
    revol_util: Annotated[StrictFiniteFloat, Field(ge=0, le=200)]
    total_acc: Annotated[StrictFiniteFloat, Field(ge=0)]
    purpose: NonemptyString
    home_ownership: NonemptyString
    verification_status: NonemptyString
    emp_length: NonemptyString
    addr_state: NonemptyString

    @field_validator("addr_state", mode="before")
    @classmethod
    def normalize_state(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().upper()

    @field_validator("addr_state")
    @classmethod
    def validate_state(cls, value: str) -> str:
        if re.fullmatch(r"[A-Z]{2}", value) is None:
            raise ValueError("addr_state must contain exactly two ASCII letters")
        return value

    @model_validator(mode="after")
    def validate_fico_order(self) -> "CreditApplication":
        if self.fico_range_low > self.fico_range_high:
            raise ValueError("fico_range_low must not exceed fico_range_high")
        return self


class CreditPrediction(StrictSchema):
    default_probability: Annotated[StrictFiniteFloat, Field(ge=0, le=1)]
    action: Literal["approve", "manual_review", "decline"]
    explanation: list[tuple[NonemptyString, StrictFiniteFloat]]

    @field_validator("explanation")
    @classmethod
    def validate_explanation(
        cls,
        explanation: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        names = [name for name, _ in explanation]
        if len(names) != len(set(names)):
            raise ValueError("explanation feature names must be unique")
        return explanation
