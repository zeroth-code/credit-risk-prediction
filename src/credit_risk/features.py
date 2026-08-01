from pathlib import Path

import pandas as pd
import yaml

DEFAULT_FEATURE_PATH = Path("configs/features.yaml")


def load_feature_dictionary(path: str | Path = DEFAULT_FEATURE_PATH) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as feature_file:
        return yaml.safe_load(feature_file)


def prohibited_columns(path: str | Path = DEFAULT_FEATURE_PATH) -> set[str]:
    dictionary = load_feature_dictionary(path)
    return set(dictionary["post_origination"])


def feature_columns(
    feature_set: str,
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> tuple[list[str], list[str]]:
    if feature_set not in {"challenger", "full_underwriting"}:
        raise ValueError(f"unknown feature set: {feature_set}")

    configured = load_feature_dictionary(path)[feature_set]
    return list(configured["numeric"]), list(configured["categorical"])


def build_feature_frame(
    frame: pd.DataFrame,
    selected_columns: list[str],
    path: str | Path = DEFAULT_FEATURE_PATH,
) -> pd.DataFrame:
    blocked = [column for column in selected_columns if column in prohibited_columns(path)]
    if blocked:
        raise ValueError(f"prohibited columns: {', '.join(blocked)}")

    missing = [column for column in selected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    return frame.loc[:, selected_columns].copy()
