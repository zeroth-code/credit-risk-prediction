import re
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

import pandas as pd

from credit_risk.config import DateWindow

REQUIRED_GENERATION_ARTIFACTS = (
    "train.parquet",
    "validation.parquet",
    "calibration.parquet",
    "test.parquet",
    "population_audit.json",
)


def resolve_current_generation(processed_dir: str | Path) -> Path:
    processed_path = Path(processed_dir)
    current_path = processed_path / "CURRENT"
    try:
        pointer_content = current_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"CURRENT pointer not found: {current_path}") from error
    if re.fullmatch(r"[0-9a-f]{32}\n", pointer_content) is None:
        raise ValueError(
            "CURRENT must contain a 32-character lowercase hexadecimal generation_id and newline"
        )
    generation_id = pointer_content[:-1]
    generation_path = processed_path / "generations" / generation_id
    if not generation_path.is_dir():
        raise FileNotFoundError(f"current generation directory not found: {generation_path}")

    missing_artifacts = [
        artifact_name
        for artifact_name in REQUIRED_GENERATION_ARTIFACTS
        if not (generation_path / artifact_name).is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "current generation is missing required artifacts: " + ", ".join(missing_artifacts)
        )
    return generation_path


def split_by_time(
    frame: pd.DataFrame, windows: Mapping[str, DateWindow]
) -> dict[str, pd.DataFrame]:
    if "issue_d" not in frame.columns:
        raise ValueError("missing required column: issue_d")
    if not windows:
        raise ValueError("windows must not be empty")

    window_items = list(windows.items())
    for (left_name, left), (right_name, right) in pairwise(window_items):
        if left.end >= right.start:
            raise ValueError(
                "windows overlap or are unordered: "
                f"{left_name} ends on {left.end.isoformat()}, "
                f"but {right_name} starts on {right.start.isoformat()}"
            )

    partitions: dict[str, pd.DataFrame] = {}
    for name, window in window_items:
        partition = frame.loc[
            frame["issue_d"].between(
                pd.Timestamp(window.start), pd.Timestamp(window.end), inclusive="both"
            )
        ].copy()
        partition = partition.reset_index(drop=True)
        if partition.empty:
            raise ValueError(f"partition {name} is empty")
        partitions[name] = partition

    for (left_name, left), (right_name, right) in pairwise(partitions.items()):
        left_max = left["issue_d"].max()
        right_min = right["issue_d"].min()
        if left_max >= right_min:
            raise ValueError(
                "partitions overlap or are unordered: "
                f"{left_name} maximum issue_d {left_max} is not before "
                f"{right_name} minimum issue_d {right_min}"
            )

    return partitions
