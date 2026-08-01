import json

from credit_risk.config import load_config
from credit_risk.data import build_modeling_population, load_raw_csv
from credit_risk.splitting import split_by_time


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

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        partition.to_parquet(config.processed_dir / f"{name}.parquet", index=False)

    with (config.processed_dir / "population_audit.json").open("w", encoding="utf-8") as audit_file:
        json.dump(audit, audit_file, indent=2)


if __name__ == "__main__":
    main()
