from collections.abc import Collection
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("id", "issue_d", "term", "loan_status", "loan_amnt")


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def validate_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def parse_issue_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def build_modeling_population(
    frame: pd.DataFrame,
    *,
    term: str,
    good_statuses: Collection[str],
    bad_statuses: Collection[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_required_columns(frame)
    result = frame.copy()
    audit = {"initial_rows": len(result)}

    result["term"] = result["term"].astype("string").str.strip()
    result["issue_d"] = parse_issue_date(result["issue_d"])

    result = result.drop_duplicates(subset=["id"], keep=False)
    audit["after_duplicates"] = len(result)

    result = result.loc[result["issue_d"].notna()]
    audit["after_valid_dates"] = len(result)

    result = result.loc[result["term"] == term]
    audit["after_term_filter"] = len(result)

    final_statuses = set(good_statuses) | set(bad_statuses)
    result = result.loc[result["loan_status"].isin(final_statuses)].copy()
    result["bad"] = result["loan_status"].isin(bad_statuses).astype("int8")
    audit["final_rows"] = len(result)

    result = result.sort_values(["issue_d", "id"]).reset_index(drop=True)
    return result, audit
