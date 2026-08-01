import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

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
    audit_windows = {
        "train": DateWindow(start="2011-01-01", end="2013-12-31"),
        "validation": DateWindow(start="2014-03-01", end="2014-06-30"),
        "calibration": DateWindow(start="2014-07-01", end="2014-12-31"),
        "test": DateWindow(start="2015-01-01", end="2015-12-31"),
    }
    config = SimpleNamespace(
        raw_csv=raw_csv,
        processed_dir=processed_dir,
        loan_term="36 months",
        good_statuses=["Fully Paid"],
        bad_statuses=["Charged Off", "Default"],
        **audit_windows,
    )
    raw = pd.DataFrame(
        {
            "id": ["outside", "train", "gap", "validation", "calibration", "test"],
            "issue_d": [
                "Dec-2010",
                "Dec-2013",
                "Jan-2014",
                "Mar-2014",
                "Sep-2014",
                "Apr-2015",
            ],
            "term": [" 36 months"] * 6,
            "loan_status": [
                "Fully Paid",
                "Fully Paid",
                "Charged Off",
                "Charged Off",
                "Fully Paid",
                "Default",
            ],
            "loan_amnt": [7000, 10000, 11000, 12000, 8000, 9000],
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
        "initial_rows": 6,
        "after_valid_ids": 6,
        "after_duplicates": 6,
        "after_valid_dates": 6,
        "after_term_filter": 6,
        "final_rows": 6,
        "partition_rows": {"train": 1, "validation": 1, "calibration": 1, "test": 1},
        "assigned_rows": 4,
        "unassigned_rows": 2,
        "outside_window_rows": 1,
        "window_gap_rows": 1,
    }
    assert audit["assigned_rows"] + audit["unassigned_rows"] == audit["final_rows"]
    assert audit["outside_window_rows"] + audit["window_gap_rows"] == audit["unassigned_rows"]
    assert not (processed_dir / ".prepare_data_staging").exists()


def test_prepare_data_script_runs_directly_from_another_working_directory(tmp_path: Path) -> None:
    script_path = Path("scripts/prepare_data.py").resolve()
    raw_csv = tmp_path / "raw.csv"
    processed_dir = tmp_path / "processed"
    raw = pd.DataFrame(
        {
            "id": ["train", "validation", "calibration", "test"],
            "issue_d": ["Dec-2013", "Feb-2014", "Sep-2014", "Apr-2015"],
            "term": [" 36 months"] * 4,
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Default"],
            "loan_amnt": [10000, 12000, 8000, 9000],
        }
    )
    raw.to_csv(raw_csv, index=False)

    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config["raw_csv"] = str(raw_csv)
    config["processed_dir"] = str(processed_dir)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert {
        name: pd.read_parquet(processed_dir / f"{name}.parquet")["id"].tolist()
        for name in _windows()
    } == {
        "train": ["train"],
        "validation": ["validation"],
        "calibration": ["calibration"],
        "test": ["test"],
    }
    assert (processed_dir / "population_audit.json").is_file()


def test_prepare_data_preserves_published_artifacts_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path("scripts/prepare_data.py")
    spec = importlib.util.spec_from_file_location("prepare_data_transaction_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_data)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    final_paths = {
        name: processed_dir / f"{name}.parquet"
        for name in ("train", "validation", "calibration", "test")
    }
    for name, path in final_paths.items():
        pd.DataFrame({"id": [f"old-{name}"]}).to_parquet(path, index=False)
    audit_path = processed_dir / "population_audit.json"
    audit_path.write_text('{"version": "old"}', encoding="utf-8")
    original_artifacts = {path: path.read_bytes() for path in [*final_paths.values(), audit_path]}

    config = SimpleNamespace(
        raw_csv=tmp_path / "raw.csv",
        processed_dir=processed_dir,
        loan_term="36 months",
        good_statuses=["Fully Paid"],
        bad_statuses=["Charged Off", "Default"],
        **_windows(),
    )
    raw = pd.DataFrame(
        {
            "id": ["new-train", "new-validation", "new-calibration", "new-test"],
            "issue_d": ["Dec-2013", "Feb-2014", "Sep-2014", "Apr-2015"],
            "term": [" 36 months"] * 4,
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Default"],
            "loan_amnt": [10000, 12000, 8000, 9000],
        }
    )
    monkeypatch.setattr(prepare_data, "load_config", lambda path: config)
    monkeypatch.setattr(prepare_data, "load_raw_csv", lambda path: raw)

    original_to_parquet = pd.DataFrame.to_parquet
    attempted_paths: list[Path] = []

    def fail_third_parquet_write(
        frame: pd.DataFrame, path: str | Path, *args: object, **kwargs: object
    ) -> None:
        attempted_paths.append(Path(path))
        if len(attempted_paths) == 3:
            raise RuntimeError("injected parquet failure")
        original_to_parquet(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_third_parquet_write)

    with pytest.raises(RuntimeError, match="injected parquet failure"):
        prepare_data.main()

    assert {path: path.read_bytes() for path in original_artifacts} == original_artifacts
    assert {path.parent for path in attempted_paths} == {processed_dir / ".prepare_data_staging"}


def test_prepare_data_rejects_partition_audit_reconciliation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path("scripts/prepare_data.py")
    spec = importlib.util.spec_from_file_location("prepare_data_audit_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_data)

    config = SimpleNamespace(
        raw_csv=tmp_path / "raw.csv",
        processed_dir=tmp_path / "processed",
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
    monkeypatch.setattr(prepare_data, "load_config", lambda path: config)
    monkeypatch.setattr(prepare_data, "load_raw_csv", lambda path: raw)

    def duplicate_train_row(
        population: pd.DataFrame, windows: Mapping[str, DateWindow]
    ) -> dict[str, pd.DataFrame]:
        return {
            "train": population.iloc[[0, 0]].reset_index(drop=True),
            "validation": population.iloc[[1]].reset_index(drop=True),
            "calibration": population.iloc[[2]].reset_index(drop=True),
            "test": population.iloc[[3]].reset_index(drop=True),
        }

    monkeypatch.setattr(prepare_data, "split_by_time", duplicate_train_row)

    with pytest.raises(ValueError, match="unassigned_rows"):
        prepare_data.main()
