from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.data import (
    REQUIRED_COLUMNS,
    build_modeling_population,
    load_raw_csv,
    parse_issue_date,
    validate_required_columns,
)


def test_required_columns_include_raw_data_contract() -> None:
    assert {"id", "issue_d", "term", "loan_status", "loan_amnt"} <= set(REQUIRED_COLUMNS)


def test_load_raw_csv_disables_low_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = pd.DataFrame({"id": ["1"]})
    calls: list[tuple[str | Path, bool]] = []

    def fake_read_csv(path: str | Path, *, low_memory: bool) -> pd.DataFrame:
        calls.append((path, low_memory))
        return expected

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    path = Path("data/raw/accepted_2007_to_2018Q4.csv")
    result = load_raw_csv(path)

    assert result is expected
    assert calls == [(path, False)]


def test_validate_required_columns_rejects_missing_loan_status() -> None:
    frame = pd.DataFrame(columns=["id", "issue_d", "term", "loan_amnt"])

    with pytest.raises(ValueError, match="loan_status"):
        validate_required_columns(frame)


def test_parse_issue_date_uses_month_year_format_and_coerces_invalid_values() -> None:
    result = parse_issue_date(pd.Series(["Jan-2013", "not-a-date", None]))

    assert result.iloc[0] == pd.Timestamp("2013-01-01")
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_build_modeling_population_keeps_final_36_month_loans() -> None:
    frame = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "issue_d": ["Jan-2013", "Feb-2013", "Mar-2013", "Apr-2013"],
            "term": [" 36 months", " 36 months", " 60 months", " 36 months"],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Current"],
            "loan_amnt": [10000, 12000, 15000, 8000],
        }
    )

    result, audit = build_modeling_population(
        frame,
        term="36 months",
        good_statuses=["Fully Paid"],
        bad_statuses=["Charged Off", "Default"],
    )

    assert result["id"].tolist() == ["1", "2"]
    assert result["bad"].tolist() == [0, 1]
    assert result["bad"].dtype == "int8"
    assert result["term"].tolist() == ["36 months", "36 months"]
    assert audit["final_rows"] == 2


def test_build_modeling_population_removes_all_duplicates_and_audits_each_stage() -> None:
    frame = pd.DataFrame(
        {
            "id": ["1", "2", "duplicate", "duplicate", "invalid", "60", "current", "0"],
            "issue_d": [
                "Mar-2013",
                "Jan-2013",
                "Feb-2013",
                "Mar-2013",
                "invalid",
                "Apr-2013",
                "May-2013",
                "Feb-2013",
            ],
            "term": [
                " 36 months",
                " 36 months",
                " 36 months",
                " 36 months",
                " 36 months",
                " 60 months",
                " 36 months",
                " 36 months",
            ],
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Fully Paid",
                "Fully Paid",
                "Fully Paid",
                "Current",
                "Fully Paid",
            ],
            "loan_amnt": [10000, 12000, 9000, 9000, 7000, 15000, 8000, 6000],
        }
    )
    original = frame.copy(deep=True)

    result, audit = build_modeling_population(
        frame,
        term="36 months",
        good_statuses=["Fully Paid"],
        bad_statuses=["Charged Off", "Default"],
    )

    assert_frame_equal(frame, original)
    assert "duplicate" not in result["id"].tolist()
    assert result["id"].tolist() == ["2", "0", "1"]
    assert audit == {
        "initial_rows": 8,
        "after_duplicates": 6,
        "after_valid_dates": 5,
        "after_term_filter": 4,
        "final_rows": 3,
    }
