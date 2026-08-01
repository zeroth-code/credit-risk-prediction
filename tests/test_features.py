import pandas as pd
import pytest

from credit_risk.features import (
    build_feature_frame,
    feature_columns,
    load_feature_dictionary,
    prohibited_columns,
)

CHALLENGER_NUMERIC = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]
CHALLENGER_CATEGORICAL = [
    "purpose",
    "home_ownership",
    "verification_status",
    "emp_length",
    "addr_state",
]
POST_ORIGINATION = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "last_credit_pull_d",
]


def test_prohibited_columns_include_post_origination_leakage() -> None:
    blocked = prohibited_columns()

    assert "recoveries" in blocked
    assert "last_pymnt_amnt" in blocked
    assert "out_prncp" in blocked


def test_build_feature_frame_rejects_recoveries() -> None:
    frame = pd.DataFrame(
        {"loan_amnt": [10_000], "annual_inc": [60_000], "recoveries": [100], "bad": [1]}
    )

    with pytest.raises(ValueError, match="recoveries"):
        build_feature_frame(frame, ["loan_amnt", "recoveries"])


def test_build_feature_frame_reports_all_prohibited_columns() -> None:
    frame = pd.DataFrame({"loan_amnt": [10_000], "out_prncp": [9_000], "recoveries": [100]})

    with pytest.raises(ValueError) as error:
        build_feature_frame(frame, ["loan_amnt", "out_prncp", "recoveries"])

    assert "out_prncp" in str(error.value)
    assert "recoveries" in str(error.value)


def test_build_feature_frame_reports_all_missing_columns() -> None:
    frame = pd.DataFrame({"loan_amnt": [10_000]})

    with pytest.raises(ValueError) as error:
        build_feature_frame(frame, ["loan_amnt", "annual_inc", "purpose"])

    assert "annual_inc" in str(error.value)
    assert "purpose" in str(error.value)


def test_build_feature_frame_preserves_selection_order_and_returns_a_copy() -> None:
    frame = pd.DataFrame(
        {"loan_amnt": [10_000], "annual_inc": [60_000], "purpose": ["credit_card"]}
    )

    result = build_feature_frame(frame, ["purpose", "loan_amnt"])

    assert result.columns.tolist() == ["purpose", "loan_amnt"]
    result.loc[0, "loan_amnt"] = 0
    assert frame.loc[0, "loan_amnt"] == 10_000


def test_load_feature_dictionary_returns_configured_feature_sets() -> None:
    assert load_feature_dictionary() == {
        "challenger": {
            "numeric": CHALLENGER_NUMERIC,
            "categorical": CHALLENGER_CATEGORICAL,
        },
        "full_underwriting": {
            "numeric": [*CHALLENGER_NUMERIC, "int_rate"],
            "categorical": [*CHALLENGER_CATEGORICAL, "grade", "sub_grade"],
        },
        "post_origination": POST_ORIGINATION,
    }


def test_feature_columns_returns_challenger_in_configured_order() -> None:
    numeric, categorical = feature_columns("challenger")

    assert numeric == CHALLENGER_NUMERIC
    assert categorical == CHALLENGER_CATEGORICAL


def test_feature_columns_returns_full_underwriting_in_configured_order() -> None:
    numeric, categorical = feature_columns("full_underwriting")

    assert numeric == [*CHALLENGER_NUMERIC, "int_rate"]
    assert categorical == [*CHALLENGER_CATEGORICAL, "grade", "sub_grade"]


def test_feature_columns_rejects_unknown_feature_set() -> None:
    with pytest.raises(ValueError, match="experimental"):
        feature_columns("experimental")


def test_feature_columns_returns_independent_lists() -> None:
    first_numeric, first_categorical = feature_columns("challenger")
    second_numeric, second_categorical = feature_columns("challenger")

    first_numeric.append("recoveries")
    first_categorical.append("last_pymnt_d")

    assert second_numeric == CHALLENGER_NUMERIC
    assert second_categorical == CHALLENGER_CATEGORICAL
