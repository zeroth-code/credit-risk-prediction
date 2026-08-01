import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from credit_risk.config import DateWindow
from credit_risk.splitting import split_by_time


def _windows() -> Mapping[str, DateWindow]:
    return {
        "train": DateWindow(start="2011-01-01", end="2013-12-31"),
        "validation": DateWindow(start="2014-01-01", end="2014-06-30"),
        "calibration": DateWindow(start="2014-07-01", end="2014-12-31"),
        "test": DateWindow(start="2015-01-01", end="2015-12-31"),
    }


def test_split_by_time_creates_ordered_out_of_time_partitions() -> None:
    frame = pd.DataFrame(
        {
            "id": ["outside", "train", "validation", "calibration", "test"],
            "issue_d": pd.to_datetime(
                ["2010-12-01", "2013-12-01", "2014-02-01", "2014-09-01", "2015-04-01"]
            ),
        }
    )

    partitions = split_by_time(frame, _windows())

    assert list(partitions) == ["train", "validation", "calibration", "test"]
    assert {name: part["id"].tolist() for name, part in partitions.items()} == {
        "train": ["train"],
        "validation": ["validation"],
        "calibration": ["calibration"],
        "test": ["test"],
    }
    assert partitions["train"]["issue_d"].max() < partitions["validation"]["issue_d"].min()


def test_split_by_time_requires_issue_date_column() -> None:
    with pytest.raises(ValueError, match="issue_d"):
        split_by_time(pd.DataFrame({"id": ["1"]}), _windows())


def test_split_by_time_rejects_empty_windows() -> None:
    frame = pd.DataFrame({"issue_d": pd.to_datetime(["2013-12-01"])})

    with pytest.raises(ValueError, match="windows"):
        split_by_time(frame, {})


@pytest.mark.parametrize(
    "windows",
    [
        {
            "train": DateWindow(start="2013-01-01", end="2014-03-31"),
            "validation": DateWindow(start="2014-02-01", end="2014-06-30"),
        },
        {
            "train": DateWindow(start="2015-01-01", end="2015-12-31"),
            "validation": DateWindow(start="2014-01-01", end="2014-12-31"),
        },
    ],
)
def test_split_by_time_rejects_overlapping_or_unordered_window_definitions(
    windows: Mapping[str, DateWindow],
) -> None:
    frame = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2013-06-01", "2014-05-01", "2015-05-01"]),
        }
    )

    with pytest.raises(ValueError, match="overlap|unordered"):
        split_by_time(frame, windows)


def test_split_by_time_names_empty_partition_in_error() -> None:
    frame = pd.DataFrame({"issue_d": pd.to_datetime(["2013-12-01"])})
    windows = {
        "train": DateWindow(start="2013-01-01", end="2013-12-31"),
        "validation": DateWindow(start="2014-01-01", end="2014-06-30"),
    }

    with pytest.raises(ValueError, match="validation"):
        split_by_time(frame, windows)


def test_split_by_time_returns_independent_copies_with_reset_indexes() -> None:
    frame = pd.DataFrame(
        {
            "id": ["first", "second"],
            "issue_d": pd.to_datetime(["2013-02-01", "2013-03-01"]),
        },
        index=[5, 9],
    )
    windows = {"train": DateWindow(start="2013-01-01", end="2013-12-31")}

    partition = split_by_time(frame, windows)["train"]

    assert partition.index.tolist() == [0, 1]
    partition.loc[0, "id"] = "changed"
    assert frame.loc[5, "id"] == "first"


def test_split_by_time_includes_both_window_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "id": ["start", "middle", "end"],
            "issue_d": pd.to_datetime(["2013-01-01", "2013-06-01", "2013-12-31"]),
        }
    )
    windows = {"train": DateWindow(start="2013-01-01", end="2013-12-31")}

    partition = split_by_time(frame, windows)["train"]

    assert partition["id"].tolist() == ["start", "middle", "end"]


def test_split_by_time_rejects_actual_partition_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2013-06-01", "2014-03-01"]),
        }
    )
    windows = {
        "train": DateWindow(start="2013-01-01", end="2013-12-31"),
        "validation": DateWindow(start="2014-01-01", end="2014-06-30"),
    }

    def overlapping_mask(
        series: pd.Series,
        left: pd.Timestamp,
        right: pd.Timestamp,
        *,
        inclusive: str,
    ) -> pd.Series:
        assert left < right
        assert inclusive == "both"
        return pd.Series([True, False], index=series.index)

    monkeypatch.setattr(pd.Series, "between", overlapping_mask)

    with pytest.raises(ValueError, match="overlap|unordered"):
        split_by_time(frame, windows)


def test_prepare_data_main_writes_partitions_and_population_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path("scripts/prepare_data.py")
    spec = importlib.util.spec_from_file_location("prepare_data", script_path)
    assert spec is not None
    assert spec.loader is not None
    prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_data)

    processed_dir = tmp_path / "nested" / "processed"
    raw_csv = tmp_path / "raw.csv"
    config = SimpleNamespace(
        raw_csv=raw_csv,
        processed_dir=processed_dir,
        loan_term="36 months",
        good_statuses=["Fully Paid"],
        bad_statuses=["Charged Off", "Default"],
        **_windows(),
    )
    raw = pd.DataFrame(
        {
            "id": ["train", "validation", "calibration", "test"],
            "issue_d": ["Dec-2013", "Feb-2014", "Sep-2014", "Apr-2015"],
            "term": [" 36 months"] * 4,
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Default"],
            "loan_amnt": [10000, 12000, 8000, 9000],
        }
    )
    loaded_config_paths: list[str] = []
    loaded_raw_paths: list[Path] = []

    def fake_load_config(path: str) -> SimpleNamespace:
        loaded_config_paths.append(path)
        return config

    def fake_load_raw_csv(path: Path) -> pd.DataFrame:
        loaded_raw_paths.append(path)
        return raw

    monkeypatch.setattr(prepare_data, "load_config", fake_load_config)
    monkeypatch.setattr(prepare_data, "load_raw_csv", fake_load_raw_csv)

    prepare_data.main()

    assert loaded_config_paths == ["configs/base.yaml"]
    assert loaded_raw_paths == [raw_csv]
    expected_ids = {
        "train": "train",
        "validation": "validation",
        "calibration": "calibration",
        "test": "test",
    }
    for name, expected_id in expected_ids.items():
        parquet_path = processed_dir / f"{name}.parquet"
        partition = pd.read_parquet(parquet_path)
        parquet_pandas_metadata = json.loads(pq.read_metadata(parquet_path).metadata[b"pandas"])

        assert partition["id"].tolist() == [expected_id]
        assert parquet_pandas_metadata["index_columns"] == []

    audit = json.loads((processed_dir / "population_audit.json").read_text(encoding="utf-8"))
    assert audit == {
        "initial_rows": 4,
        "after_valid_ids": 4,
        "after_duplicates": 4,
        "after_valid_dates": 4,
        "after_term_filter": 4,
        "final_rows": 4,
    }
