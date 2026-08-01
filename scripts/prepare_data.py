import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import build_modeling_population, load_raw_csv  # noqa: E402
from credit_risk.splitting import split_by_time  # noqa: E402


def main() -> None:
    config = load_config("configs/base.yaml")
    raw = load_raw_csv(config.raw_csv)
    population, audit = build_modeling_population(
        raw,
        term=config.loan_term,
        good_statuses=set(config.good_statuses),
        bad_statuses=set(config.bad_statuses),
    )
    windows = {
        "train": config.train,
        "validation": config.validation,
        "calibration": config.calibration,
        "test": config.test,
    }
    partitions = split_by_time(population, windows)

    partition_rows = {name: len(partition) for name, partition in partitions.items()}
    assigned_rows = sum(partition_rows.values())
    unassigned_rows = len(population) - assigned_rows

    issue_dates = population["issue_d"]
    assigned_mask = pd.Series(False, index=population.index)
    for window in windows.values():
        assigned_mask |= issue_dates.between(
            pd.Timestamp(window.start), pd.Timestamp(window.end), inclusive="both"
        )

    earliest_start = pd.Timestamp(min(window.start for window in windows.values()))
    latest_end = pd.Timestamp(max(window.end for window in windows.values()))
    outside_mask = (issue_dates < earliest_start) | (issue_dates > latest_end)
    window_gap_mask = ~assigned_mask & ~outside_mask
    outside_window_rows = int(outside_mask.sum())
    window_gap_rows = int(window_gap_mask.sum())

    if assigned_rows + unassigned_rows != len(population):
        raise ValueError("assigned_rows + unassigned_rows must equal population rows")
    if outside_window_rows + window_gap_rows != unassigned_rows:
        raise ValueError("outside_window_rows + window_gap_rows must equal unassigned_rows")

    audit.update(
        {
            "partition_rows": partition_rows,
            "assigned_rows": assigned_rows,
            "unassigned_rows": unassigned_rows,
            "outside_window_rows": outside_window_rows,
            "window_gap_rows": window_gap_rows,
        }
    )

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = config.processed_dir / ".prepare_data_staging"
    staging_dir.mkdir(exist_ok=True)
    for name, partition in partitions.items():
        partition.to_parquet(staging_dir / f"{name}.parquet", index=False)

    audit_name = "population_audit.json"
    with (staging_dir / audit_name).open("w", encoding="utf-8") as audit_file:
        json.dump(audit, audit_file, indent=2)

    artifact_names = [f"{name}.parquet" for name in partitions] + [audit_name]
    for artifact_name in artifact_names:
        (staging_dir / artifact_name).replace(config.processed_dir / artifact_name)
    staging_dir.rmdir()


if __name__ == "__main__":
    main()
