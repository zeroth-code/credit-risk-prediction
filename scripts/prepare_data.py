import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import build_modeling_population, load_raw_csv  # noqa: E402
from credit_risk.splitting import split_by_time  # noqa: E402

GENERATION_ARTIFACTS = (
    "train.parquet",
    "validation.parquet",
    "calibration.parquet",
    "test.parquet",
    "population_audit.json",
)


def _ensure_root_artifact_symlinks(processed_dir: Path, generation_id: str) -> None:
    missing_paths: list[tuple[Path, Path]] = []
    for artifact_name in GENERATION_ARTIFACTS:
        artifact_path = processed_dir / artifact_name
        expected_target = Path("CURRENT") / artifact_name
        if artifact_path.is_symlink():
            actual_target = artifact_path.readlink()
            if actual_target != expected_target:
                raise ValueError(
                    f"root artifact symlink {artifact_path} points to {actual_target}, "
                    f"expected {expected_target}"
                )
        elif artifact_path.exists():
            raise FileExistsError(
                f"root artifact path exists and is not a symlink: {artifact_path}"
            )
        else:
            missing_paths.append((artifact_path, expected_target))

    for artifact_path, expected_target in missing_paths:
        temporary_path = processed_dir / f".{artifact_path.name}.{generation_id}.tmp"
        temporary_path.symlink_to(expected_target, target_is_directory=False)
        temporary_path.replace(artifact_path)


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
    partition_assigned_rows = sum(partition_rows.values())

    issue_dates = population["issue_d"]
    assigned_mask = pd.Series(False, index=population.index)
    for window in windows.values():
        assigned_mask |= issue_dates.between(
            pd.Timestamp(window.start), pd.Timestamp(window.end), inclusive="both"
        )
    assigned_rows = int(assigned_mask.sum())
    unassigned_rows = int((~assigned_mask).sum())

    if partition_assigned_rows != assigned_rows:
        raise ValueError(
            "partition_rows total must equal the independently computed assigned mask row count"
        )

    earliest_start = pd.Timestamp(min(window.start for window in windows.values()))
    latest_end = pd.Timestamp(max(window.end for window in windows.values()))
    outside_mask = (issue_dates < earliest_start) | (issue_dates > latest_end)
    window_gap_mask = ~assigned_mask & ~outside_mask
    outside_window_rows = int(outside_mask.sum())
    window_gap_rows = int(window_gap_mask.sum())

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

    generation_id = uuid4().hex
    generations_dir = config.processed_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = generations_dir / f".{generation_id}.staging"
    generation_dir = generations_dir / generation_id
    staging_dir.mkdir()
    for name, partition in partitions.items():
        partition.to_parquet(staging_dir / f"{name}.parquet", index=False)

    audit_name = "population_audit.json"
    with (staging_dir / audit_name).open("w", encoding="utf-8") as audit_file:
        json.dump(audit, audit_file, indent=2)

    staging_dir.replace(generation_dir)
    _ensure_root_artifact_symlinks(config.processed_dir, generation_id)
    temporary_current = config.processed_dir / f".CURRENT.{generation_id}.tmp"
    temporary_current.symlink_to(Path("generations") / generation_id, target_is_directory=True)
    temporary_current.replace(config.processed_dir / "CURRENT")


if __name__ == "__main__":
    main()
