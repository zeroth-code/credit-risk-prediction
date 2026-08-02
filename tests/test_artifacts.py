import hashlib
import json
from pathlib import Path
from typing import BinaryIO, Literal

import pytest
from pydantic import ValidationError

from credit_risk.artifacts import (
    ReleaseFile,
    ReleaseManifest,
    create_release_bundle,
    load_manifest,
    save_manifest,
    sha256_file,
    validate_release_bundle,
)
from credit_risk.schemas import CreditApplication, CreditPrediction

RELEASE_SOURCE_FILES = (
    "calibrated_model.joblib",
    "preprocessor.joblib",
    "policy.json",
    "validation_metrics.json",
    "calibration_metrics.json",
    "calibration_curve.csv",
    "cost_sensitivity.csv",
    "final_test_metrics.json",
    "confusion_matrix.csv",
    "policy_test_results.json",
    "temporal_metrics.csv",
    "fairness_income.csv",
    "fairness_home_ownership.csv",
    "fairness_region.csv",
    "fairness_employment.csv",
    "fairness_summary.json",
    "shap_importance.csv",
    "shap_explanations.json",
)
PROHIBITED_RELEASE_FILES = {
    "tuning_trials.csv",
    "uncalibrated_model.joblib",
    "scored_test.parquet",
}


def _release_file(path: str, *, key: str = "test") -> ReleaseFile:
    return ReleaseFile(
        key=key,
        path=path,
        sha256="a" * 64,
        size_bytes=1,
        media_type="application/octet-stream",
    )


def _write_release_sources(source_dir: Path, *, prefix: str = "source") -> dict[Path, bytes]:
    source_dir.mkdir(parents=True)
    written: dict[Path, bytes] = {}
    for index, name in enumerate(RELEASE_SOURCE_FILES):
        path = source_dir / name
        payload = f"{prefix}:{index}:{name}\n".encode()
        path.write_bytes(payload)
        written[path] = payload
    return written


def _create_bundle(source_dir: Path, release_dir: Path) -> ReleaseManifest:
    return create_release_bundle(
        source_dir,
        release_dir,
        version="0.1.0",
        feature_set="challenger",
        data_hash="b" * 64,
    )


def _application_payload() -> dict[str, object]:
    return {
        "loan_amnt": 25_000.0,
        "annual_inc": 95_000.0,
        "dti": 18.5,
        "delinq_2yrs": 0.0,
        "fico_range_low": 720.0,
        "fico_range_high": 724.0,
        "inq_last_6mths": 1.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 12_000.0,
        "revol_util": 34.5,
        "total_acc": 16.0,
        "purpose": " debt_consolidation ",
        "home_ownership": " MORTGAGE ",
        "verification_status": " Verified ",
        "emp_length": " 10+ years ",
        "addr_state": " ca ",
    }


def test_release_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ReleaseManifest(
        version="0.1.0",
        feature_set="challenger",
        model_file="calibrated_model.joblib",
        preprocessor_file="preprocessor.joblib",
        policy_file="policy.json",
        data_hash="abc123",
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest
    assert manifest.schema_version == "1.0"
    assert manifest.files == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_file", ""),
        ("model_file", "/tmp/model.joblib"),
        ("model_file", "."),
        ("model_file", "../model.joblib"),
        ("preprocessor_file", "nested/../../preprocessor.joblib"),
        ("policy_file", "nested\\policy.json"),
        ("data_hash", "   "),
    ],
)
def test_release_manifest_rejects_invalid_core_values(field: str, value: str) -> None:
    payload = {
        "version": "0.1.0",
        "feature_set": "challenger",
        "model_file": "calibrated_model.joblib",
        "preprocessor_file": "preprocessor.joblib",
        "policy_file": "policy.json",
        "data_hash": "abc123",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ReleaseManifest.model_validate(payload)


def test_release_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ReleaseManifest.model_validate(
            {
                "version": "0.1.0",
                "feature_set": "challenger",
                "model_file": "calibrated_model.joblib",
                "preprocessor_file": "preprocessor.joblib",
                "policy_file": "policy.json",
                "data_hash": "abc123",
                "created_at": "now",
            }
        )


@pytest.mark.parametrize(
    "files",
    [
        [_release_file("a.json"), _release_file("a.json", key="other")],
        [_release_file("a.json"), _release_file("b.json")],
        [_release_file("b.json", key="b"), _release_file("a.json", key="a")],
    ],
    ids=["duplicate-path", "duplicate-key", "unsorted"],
)
def test_release_manifest_rejects_duplicate_or_unsorted_inventory(
    files: list[ReleaseFile],
) -> None:
    with pytest.raises(ValidationError):
        ReleaseManifest(
            version="0.1.0",
            feature_set="challenger",
            model_file="calibrated_model.joblib",
            preprocessor_file="preprocessor.joblib",
            policy_file="policy.json",
            data_hash="abc123",
            files=files,
        )


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"version": 1}'])
def test_load_manifest_rejects_invalid_json_or_schema(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises((ValueError, ValidationError)):
        load_manifest(path)


class _RecordingBinaryReader:
    def __init__(self, file: BinaryIO, reads: list[int]) -> None:
        self._file = file
        self._reads = reads

    def __enter__(self) -> "_RecordingBinaryReader":
        self._file.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._file.__exit__(*args)

    def read(self, size: int = -1) -> bytes:
        self._reads.append(size)
        return self._file.read(size)


def test_sha256_file_is_deterministic_and_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "payload.bin"
    payload = b"0123456789abcdef"
    path.write_bytes(payload)
    original_open = Path.open
    reads: list[int] = []

    def recording_open(
        self: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> _RecordingBinaryReader | BinaryIO:
        opened = original_open(self, mode, *args, **kwargs)
        if self == path and mode == "rb":
            return _RecordingBinaryReader(opened, reads)  # type: ignore[arg-type]
        return opened  # type: ignore[return-value]

    monkeypatch.setattr(Path, "open", recording_open)

    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(path, chunk_size=3) == expected
    assert sha256_file(path, chunk_size=3) == expected
    assert reads and -1 not in reads and max(reads) <= 3


def test_credit_application_accepts_valid_values_and_normalizes_strings() -> None:
    application = CreditApplication.model_validate(_application_payload())

    assert application.addr_state == "CA"
    assert application.purpose == "debt_consolidation"
    assert application.home_ownership == "MORTGAGE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amnt", 0.0),
        ("loan_amnt", 100_000.1),
        ("annual_inc", 0.0),
        ("dti", -0.1),
        ("dti", 100.1),
        ("delinq_2yrs", -1.0),
        ("fico_range_low", 299.0),
        ("fico_range_high", 851.0),
        ("inq_last_6mths", -1.0),
        ("open_acc", -1.0),
        ("pub_rec", -1.0),
        ("revol_bal", -1.0),
        ("revol_util", 200.1),
        ("total_acc", -1.0),
        ("loan_amnt", True),
        ("loan_amnt", "25000"),
        ("annual_inc", float("nan")),
        ("dti", float("inf")),
        ("purpose", "   "),
        ("addr_state", "C"),
        ("addr_state", "C1"),
        ("addr_state", "ÉC"),
    ],
)
def test_credit_application_rejects_invalid_field_values(field: str, value: object) -> None:
    payload = _application_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        CreditApplication.model_validate(payload)


def test_credit_application_rejects_reversed_fico_range_and_extra_fields() -> None:
    reversed_payload = _application_payload()
    reversed_payload["fico_range_low"] = 750.0
    reversed_payload["fico_range_high"] = 700.0
    with pytest.raises(ValidationError, match="fico_range_low"):
        CreditApplication.model_validate(reversed_payload)

    extra_payload = _application_payload()
    extra_payload["grade"] = "A"
    with pytest.raises(ValidationError, match="extra"):
        CreditApplication.model_validate(extra_payload)


def test_credit_prediction_accepts_strict_valid_payload() -> None:
    prediction = CreditPrediction(
        default_probability=0.25,
        action="manual_review",
        explanation=[("income", -0.4), ("dti", 0.2)],
    )

    assert prediction.explanation == [("income", -0.4), ("dti", 0.2)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_probability", -0.1),
        ("default_probability", 1.1),
        ("default_probability", True),
        ("default_probability", "0.25"),
        ("default_probability", float("nan")),
        ("action", "hold"),
        ("explanation", [("", 0.2)]),
        ("explanation", [("income", 0.2), ("income", -0.1)]),
        ("explanation", [("income", float("inf"))]),
        ("explanation", [("income", True)]),
        ("explanation", [("income", "0.2")]),
    ],
)
def test_credit_prediction_rejects_invalid_values(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "default_probability": 0.25,
        "action": "approve",
        "explanation": [("income", -0.4)],
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        CreditPrediction.model_validate(payload)


def test_create_and_validate_release_bundle_with_deterministic_inventory(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    original_sources = _write_release_sources(source_dir)

    manifest = _create_bundle(source_dir, release_dir)

    assert validate_release_bundle(release_dir) == manifest
    assert manifest.data_hash == "b" * 64
    assert [item.path for item in manifest.files] == sorted(RELEASE_SOURCE_FILES)
    assert {item.path for item in manifest.files} == set(RELEASE_SOURCE_FILES)
    assert {item.key for item in manifest.files} == set(RELEASE_SOURCE_FILES)
    assert {path.name for path in release_dir.iterdir()} == {
        *RELEASE_SOURCE_FILES,
        "release_manifest.json",
    }
    assert PROHIBITED_RELEASE_FILES.isdisjoint(path.name for path in release_dir.iterdir())
    assert all(path.read_bytes() == payload for path, payload in original_sources.items())
    for item in manifest.files:
        assert item.sha256 == sha256_file(release_dir / item.path)
        assert item.size_bytes == (release_dir / item.path).stat().st_size


def test_release_bundle_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)

    _create_bundle(source_dir, release_dir)
    first = {path.name: path.read_bytes() for path in release_dir.iterdir()}
    _create_bundle(source_dir, release_dir)
    second = {path.name: path.read_bytes() for path in release_dir.iterdir()}

    assert second == first


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra"])
def test_validate_release_bundle_rejects_changed_file_set_or_bytes(
    tmp_path: Path,
    mutation: Literal["tamper", "missing", "extra"],
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)

    if mutation == "tamper":
        (release_dir / "policy.json").write_bytes(b"tampered\n")
    elif mutation == "missing":
        (release_dir / "policy.json").unlink()
    else:
        (release_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_release_bundle(release_dir)


@pytest.mark.parametrize("unexpected_type", ["directory", "symlink"])
def test_validate_release_bundle_rejects_unexpected_non_regular_entries(
    tmp_path: Path,
    unexpected_type: Literal["directory", "symlink"],
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)
    unexpected = release_dir / "unexpected"
    if unexpected_type == "directory":
        unexpected.mkdir()
    else:
        unexpected.symlink_to(release_dir / "policy.json")

    with pytest.raises(ValueError, match="regular file"):
        validate_release_bundle(release_dir)


def test_validate_release_bundle_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)
    manifest_path = release_dir / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_release_bundle(release_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_file", "policy.json"),
        ("key", "wrong-role"),
        ("media_type", "text/plain"),
    ],
)
def test_validate_release_bundle_rejects_tampered_core_names_or_inventory_metadata(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)
    manifest_path = release_dir / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "model_file":
        payload[field] = value
    else:
        payload["files"][0][field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_release_bundle(release_dir)


def test_validate_release_bundle_rejects_swapped_model_and_preprocessor_roles(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)
    manifest_path = release_dir / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_file"] = "preprocessor.joblib"
    payload["preprocessor_file"] = "calibrated_model.joblib"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_release_bundle(release_dir)


def test_create_release_bundle_rejects_invalid_data_hash_and_symlink_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)

    with pytest.raises(ValueError, match="data_hash"):
        create_release_bundle(
            source_dir,
            release_dir,
            version="0.1.0",
            feature_set="challenger",
            data_hash="abc123",
        )

    model_path = source_dir / "calibrated_model.joblib"
    model_path.unlink()
    model_path.symlink_to(source_dir / "policy.json")
    with pytest.raises(ValueError, match="regular file"):
        _create_bundle(source_dir, release_dir)


def test_create_release_bundle_preserves_previous_bundle_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir, prefix="old")
    _create_bundle(source_dir, release_dir)
    previous = {path.name: path.read_bytes() for path in release_dir.iterdir()}
    original_source_paths = list(source_dir.iterdir())
    for path in original_source_paths:
        if path.is_file():
            path.write_bytes(f"new:{path.name}\n".encode())

    original_replace = Path.replace
    failed = False

    def fail_mid_publish(self: Path, target: str | Path) -> Path:
        nonlocal failed
        destination = Path(target)
        if destination.name == "fairness_region.csv" and not failed:
            failed = True
            raise OSError("injected publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_mid_publish)

    with pytest.raises(OSError, match="injected publication failure"):
        _create_bundle(source_dir, release_dir)

    assert {path.name: path.read_bytes() for path in release_dir.iterdir()} == previous
    assert not [path for path in release_dir.iterdir() if path.name.startswith(".")]
    assert original_source_paths


def test_create_release_bundle_retries_backup_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir, prefix="old")
    _create_bundle(source_dir, release_dir)
    previous = {path.name: path.read_bytes() for path in release_dir.iterdir()}
    for path in source_dir.iterdir():
        if path.is_file():
            path.write_bytes(f"new:{path.name}\n".encode())

    original_replace = Path.replace
    publish_failed = False
    restore_failures = 0

    def fail_publish_and_first_restore(self: Path, target: str | Path) -> Path:
        nonlocal publish_failed, restore_failures
        destination = Path(target)
        if destination.name == "fairness_region.csv" and not publish_failed:
            publish_failed = True
            raise OSError("injected publication failure")
        if (
            destination.name == "calibrated_model.joblib"
            and "recovery.backup" in self.name
            and restore_failures == 0
        ):
            restore_failures += 1
            raise OSError("transient restore failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_first_restore)

    with pytest.raises(OSError, match="injected publication failure"):
        _create_bundle(source_dir, release_dir)

    assert restore_failures == 1
    assert {path.name: path.read_bytes() for path in release_dir.iterdir()} == previous
    assert not [path for path in release_dir.iterdir() if path.name.startswith(".")]


def test_create_release_bundle_preserves_unresolved_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir, prefix="old")
    _create_bundle(source_dir, release_dir)
    previous_model = (release_dir / "calibrated_model.joblib").read_bytes()
    for path in source_dir.iterdir():
        if path.is_file():
            path.write_bytes(f"new:{path.name}\n".encode())

    original_replace = Path.replace
    publish_failed = False

    def fail_publish_and_model_restore(self: Path, target: str | Path) -> Path:
        nonlocal publish_failed
        destination = Path(target)
        if destination.name == "fairness_region.csv" and not publish_failed:
            publish_failed = True
            raise OSError("injected publication failure")
        if destination.name == "calibrated_model.joblib" and "recovery.backup" in self.name:
            raise OSError("persistent restore failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_model_restore)

    with pytest.raises(OSError, match="injected publication failure") as exc_info:
        _create_bundle(source_dir, release_dir)

    backups = [
        path
        for path in source_dir.iterdir()
        if "calibrated_model.joblib" in path.name and path.name.endswith("recovery.backup")
    ]
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous_model
    assert not [path for path in release_dir.iterdir() if path.name.endswith("recovery.backup")]
    assert any("release recovery failed" in note for note in exc_info.value.__notes__)


def test_create_release_bundle_publishes_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    original_replace = Path.replace
    published: list[str] = []

    def record_publish(self: Path, target: str | Path) -> Path:
        destination = Path(target)
        if destination.parent == release_dir:
            published.append(destination.name)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", record_publish)

    _create_bundle(source_dir, release_dir)

    assert published[-1] == "release_manifest.json"
    assert set(published[:-1]) == set(RELEASE_SOURCE_FILES)


def test_create_release_bundle_removes_stale_regular_file_after_commit(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir)
    _create_bundle(source_dir, release_dir)
    stale = release_dir / "old_release_note.txt"
    stale.write_text("stale", encoding="utf-8")

    _create_bundle(source_dir, release_dir)

    assert not stale.exists()


def test_create_release_bundle_rolls_back_when_stale_file_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir, prefix="old")
    _create_bundle(source_dir, release_dir)
    stale = release_dir / "old_release_note.txt"
    stale.write_text("stale", encoding="utf-8")
    previous = {path.name: path.read_bytes() for path in release_dir.iterdir()}
    for path in source_dir.iterdir():
        if path.is_file():
            path.write_bytes(f"new:{path.name}\n".encode())

    original_unlink = Path.unlink

    def fail_stale_removal(self: Path, *args: object, **kwargs: object) -> None:
        if self == stale:
            raise OSError("injected stale removal failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stale_removal)

    with pytest.raises(OSError, match="injected stale removal failure"):
        _create_bundle(source_dir, release_dir)

    assert {path.name: path.read_bytes() for path in release_dir.iterdir()} == previous
    assert not [path for path in release_dir.iterdir() if path.name.startswith(".")]


def test_create_release_bundle_does_not_report_failure_after_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "artifacts"
    release_dir = source_dir / "release"
    _write_release_sources(source_dir, prefix="old")
    _create_bundle(source_dir, release_dir)
    previous_model = (release_dir / "calibrated_model.joblib").read_bytes()
    for path in source_dir.iterdir():
        if path.is_file():
            path.write_bytes(f"new:{path.name}\n".encode())

    original_unlink = Path.unlink

    def fail_one_backup_cleanup(self: Path, *args: object, **kwargs: object) -> None:
        if "calibrated_model.joblib" in self.name and self.name.endswith("recovery.backup"):
            raise OSError("persistent post-commit cleanup failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one_backup_cleanup)

    manifest = _create_bundle(source_dir, release_dir)

    assert validate_release_bundle(release_dir) == manifest
    assert not [path for path in release_dir.iterdir() if path.name.endswith("recovery.backup")]
    backups = [
        path
        for path in source_dir.iterdir()
        if "calibrated_model.joblib" in path.name and path.name.endswith("recovery.backup")
    ]
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous_model
