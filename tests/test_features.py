from pathlib import Path

import pandas as pd
import pytest
import yaml

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


def _valid_feature_dictionary_payload() -> dict[str, object]:
    return {
        "challenger": {"numeric": ["loan_amnt"], "categorical": ["purpose"]},
        "full_underwriting": {
            "numeric": ["loan_amnt", "int_rate"],
            "categorical": ["purpose", "grade"],
        },
        "post_origination": ["recoveries"],
    }


def _write_feature_dictionary(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


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


def test_build_feature_frame_rejects_empty_selection() -> None:
    frame = pd.DataFrame({"loan_amnt": [10_000]})

    with pytest.raises(ValueError, match="at least one|empty"):
        build_feature_frame(frame, [])


def test_build_feature_frame_rejects_duplicate_columns() -> None:
    frame = pd.DataFrame({"loan_amnt": [10_000], "purpose": ["credit_card"]})

    with pytest.raises(ValueError, match="duplicate.*loan_amnt|loan_amnt.*duplicate"):
        build_feature_frame(frame, ["loan_amnt", "purpose", "loan_amnt"])


def test_build_feature_frame_checks_duplicates_before_prohibited_columns() -> None:
    frame = pd.DataFrame({"recoveries": [100]})

    with pytest.raises(ValueError, match="duplicate.*recoveries|recoveries.*duplicate"):
        build_feature_frame(frame, ["recoveries", "recoveries"])


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


def test_load_feature_dictionary_rejects_scalar_post_origination(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["post_origination"] = "recoveries"
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="post_origination"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_missing_top_level_section(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    del payload["full_underwriting"]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="full_underwriting"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["unexpected"] = ["value"]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="unexpected"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_missing_section_list(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    del payload["challenger"]["categorical"]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="categorical"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_unknown_section_key(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"]["unexpected"] = ["value"]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="unexpected"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_scalar_section_list(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"]["numeric"] = "loan_amnt"
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="numeric"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_empty_lists(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"]["numeric"] = []
    payload["post_origination"] = []
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError) as error:
        load_feature_dictionary(path)

    assert "numeric" in str(error.value)
    assert "post_origination" in str(error.value)


def test_load_feature_dictionary_rejects_blank_and_non_string_items(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"]["numeric"] = ["   "]
    payload["post_origination"] = [123]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError) as error:
        load_feature_dictionary(path)

    assert "numeric" in str(error.value)
    assert "post_origination" in str(error.value)


def test_load_feature_dictionary_rejects_duplicate_section_items(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"]["numeric"] = ["loan_amnt", " loan_amnt "]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate.*loan_amnt|loan_amnt.*duplicate"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_duplicate_post_origination_items(
    tmp_path: Path,
) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["post_origination"] = ["recoveries", " recoveries "]
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate.*recoveries|recoveries.*duplicate"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_rejects_section_overlap(tmp_path: Path) -> None:
    payload = _valid_feature_dictionary_payload()
    payload["challenger"] = {
        "numeric": ["loan_amnt", "shared"],
        "categorical": ["purpose", " shared "],
    }
    path = _write_feature_dictionary(tmp_path, payload)

    with pytest.raises(ValueError, match="overlap.*shared|shared.*overlap"):
        load_feature_dictionary(path)


def test_load_feature_dictionary_strips_feature_name_whitespace(tmp_path: Path) -> None:
    payload = {
        "challenger": {"numeric": [" loan_amnt "], "categorical": [" purpose "]},
        "full_underwriting": {
            "numeric": [" int_rate "],
            "categorical": [" grade "],
        },
        "post_origination": [" recoveries "],
    }
    path = _write_feature_dictionary(tmp_path, payload)

    assert load_feature_dictionary(path) == {
        "challenger": {"numeric": ["loan_amnt"], "categorical": ["purpose"]},
        "full_underwriting": {"numeric": ["int_rate"], "categorical": ["grade"]},
        "post_origination": ["recoveries"],
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
