import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from shutil import copyfile
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

RELEASE_MANIFEST_FILENAME = "release_manifest.json"
RELEASE_ARTIFACT_MEDIA_TYPES = {
    "calibrated_model.joblib": "application/octet-stream",
    "preprocessor.joblib": "application/octet-stream",
    "policy.json": "application/json",
    "validation_metrics.json": "application/json",
    "calibration_metrics.json": "application/json",
    "calibration_curve.csv": "text/csv",
    "cost_sensitivity.csv": "text/csv",
    "final_test_metrics.json": "application/json",
    "confusion_matrix.csv": "text/csv",
    "policy_test_results.json": "application/json",
    "temporal_metrics.csv": "text/csv",
    "fairness_income.csv": "text/csv",
    "fairness_home_ownership.csv": "text/csv",
    "fairness_region.csv": "text/csv",
    "fairness_employment.csv": "text/csv",
    "fairness_summary.json": "application/json",
    "shap_importance.csv": "text/csv",
    "shap_explanations.json": "application/json",
}
RELEASE_ARTIFACT_NAMES = tuple(sorted(RELEASE_ARTIFACT_MEDIA_TYPES))
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FILESYSTEM_OPERATION_ATTEMPTS = 2


def _nonempty_string(value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def _safe_relative_path(value: str, field: str) -> str:
    path_text = _nonempty_string(value, field)
    if "\\" in path_text or "\x00" in path_text:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(path_text)
    if (
        not path.parts
        or path_text == "."
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != path_text
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path_text


class ReleaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str
    path: str
    sha256: str
    size_bytes: StrictInt = Field(gt=0)
    media_type: str

    @field_validator("key", "media_type")
    @classmethod
    def validate_nonempty_strings(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _nonempty_string(value, str(field_name))

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value, "path")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must be a 64-character lowercase SHA-256 digest")
        return value


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    version: str
    feature_set: str
    model_file: str
    preprocessor_file: str
    policy_file: str
    data_hash: str
    files: list[ReleaseFile] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("schema_version must be 1.0")
        return value

    @field_validator("version", "feature_set", "data_hash")
    @classmethod
    def validate_nonempty_strings(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _nonempty_string(value, str(field_name))

    @field_validator("model_file", "preprocessor_file", "policy_file")
    @classmethod
    def validate_core_paths(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "path")
        return _safe_relative_path(value, str(field_name))

    @model_validator(mode="after")
    def validate_inventory(self) -> "ReleaseManifest":
        paths = [item.path for item in self.files]
        keys = [item.key for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("release inventory paths must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("release inventory keys must be unique")
        if paths != sorted(paths):
            raise ValueError("release inventory must be sorted by path")
        return self


def save_manifest(manifest: ReleaseManifest, path: str | Path) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            manifest.model_dump(mode="json"),
            output_file,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


def load_manifest(path: str | Path) -> ReleaseManifest:
    input_path = Path(path)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid release manifest JSON at {input_path}: {exc}") from exc
    return ReleaseManifest.model_validate(payload)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise ValueError(f"SHA-256 input must be a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_nonempty_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{description} must be non-empty: {path}")


def _validate_data_hash(data_hash: str) -> str:
    if not isinstance(data_hash, str) or SHA256_PATTERN.fullmatch(data_hash) is None:
        raise ValueError("data_hash must be a 64-character lowercase SHA-256 digest")
    return data_hash


def _temporary_sibling(path: Path, token: str, role: str) -> Path:
    return path.with_name(f".{path.name}.{token}.{role}")


def _recovery_backup_path(release_dir: Path, final_path: Path, token: str) -> Path:
    return release_dir.parent / (f".{release_dir.name}.{final_path.name}.{token}.recovery.backup")


def _retry_unlink(path: Path) -> OSError | None:
    last_error: OSError | None = None
    for _ in range(FILESYSTEM_OPERATION_ATTEMPTS):
        try:
            if path.is_symlink():
                raise OSError(f"refusing to unlink symlink temporary path: {path}")
            if path.is_file():
                path.unlink()
            return None
        except OSError as exc:
            last_error = exc
    return last_error


def _retry_replace(source: Path, destination: Path) -> OSError | None:
    last_error: OSError | None = None
    for _ in range(FILESYSTEM_OPERATION_ATTEMPTS):
        try:
            source.replace(destination)
            return None
        except OSError as exc:
            last_error = exc
    return last_error


def _cleanup_temporary_files(paths: list[Path], *, preserve: set[Path] | None = None) -> list[str]:
    preserved = preserve or set()
    failures: list[str] = []
    for path in paths:
        if path in preserved:
            continue
        error = _retry_unlink(path)
        if error is not None:
            failures.append(f"cleanup failed for temporary path {path}: {error}")
    return failures


def _add_exception_notes(exception: BaseException, notes: list[str]) -> None:
    for note in notes:
        exception.add_note(note)


def _validate_manifest_inventory(
    manifest: ReleaseManifest,
    paths_by_release_path: dict[str, Path],
) -> None:
    if not manifest.files:
        raise ValueError("release manifest inventory must not be empty")
    expected_names = set(RELEASE_ARTIFACT_NAMES)
    inventory_names = {item.path for item in manifest.files}
    if inventory_names != expected_names:
        missing = sorted(expected_names - inventory_names)
        extra = sorted(inventory_names - expected_names)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError(f"release inventory does not match contract ({'; '.join(details)})")
    core_paths = {
        "model_file": (manifest.model_file, "calibrated_model.joblib"),
        "preprocessor_file": (manifest.preprocessor_file, "preprocessor.joblib"),
        "policy_file": (manifest.policy_file, "policy.json"),
    }
    for field, (actual, expected) in core_paths.items():
        if actual != expected:
            raise ValueError(f"release manifest {field} must be {expected}")
    if set(paths_by_release_path) != inventory_names:
        raise ValueError("release files do not match manifest inventory")
    for item in manifest.files:
        if item.key != item.path:
            raise ValueError(f"release inventory key must match path for {item.path}")
        if item.media_type != RELEASE_ARTIFACT_MEDIA_TYPES[item.path]:
            raise ValueError(f"release inventory media type mismatch for {item.path}")
        path = paths_by_release_path[item.path]
        _require_regular_nonempty_file(path, f"release artifact {item.path}")
        size_bytes = path.stat().st_size
        if size_bytes != item.size_bytes:
            raise ValueError(
                f"release artifact size mismatch for {item.path}: "
                f"expected {item.size_bytes}, got {size_bytes}"
            )
        digest = sha256_file(path)
        if digest != item.sha256:
            raise ValueError(f"release artifact SHA-256 mismatch for {item.path}")


def _release_directory_entries(release_dir: Path) -> list[Path]:
    if release_dir.is_symlink() or not release_dir.is_dir():
        raise ValueError(f"release directory must be a regular directory: {release_dir}")
    entries = sorted(release_dir.iterdir(), key=lambda path: path.name)
    invalid = [path for path in entries if path.is_symlink() or not path.is_file()]
    if invalid:
        names = ", ".join(path.name for path in invalid)
        raise ValueError(f"release directory entries must be regular files: {names}")
    return entries


def validate_release_bundle(release_dir: str | Path) -> ReleaseManifest:
    release_path = Path(release_dir)
    entries = _release_directory_entries(release_path)
    manifest_path = release_path / RELEASE_MANIFEST_FILENAME
    _require_regular_nonempty_file(manifest_path, "release manifest")
    manifest = load_manifest(manifest_path)
    _validate_data_hash(manifest.data_hash)
    expected_names = {item.path for item in manifest.files} | {RELEASE_MANIFEST_FILENAME}
    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unreferenced: {', '.join(extra)}")
        raise ValueError(f"release directory does not match manifest ({'; '.join(details)})")
    paths_by_release_path = {item.path: release_path / item.path for item in manifest.files}
    _validate_manifest_inventory(manifest, paths_by_release_path)
    return manifest


def _restore_previous_release(
    known_finals: list[Path],
    previous_outputs: dict[Path, Path | None],
) -> tuple[list[str], set[Path]]:
    failures: list[str] = []
    preserved_backups: set[Path] = set()
    for final_path in known_finals:
        backup_path = previous_outputs[final_path]
        try:
            if backup_path is None:
                error = _retry_unlink(final_path)
                if error is not None:
                    raise error
            else:
                error = _retry_replace(backup_path, final_path)
                if error is not None:
                    raise error
        except OSError as exc:
            if backup_path is not None and backup_path.is_file():
                preserved_backups.add(backup_path)
            failures.append(f"release recovery failed for {final_path}: {exc}")
    return failures, preserved_backups


def _remove_stale_release_files(paths: list[Path]) -> None:
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"stale release entry must be a regular file: {path}")
        error = _retry_unlink(path)
        if error is not None:
            raise error


def create_release_bundle(
    source_dir: str | Path,
    release_dir: str | Path,
    *,
    version: str,
    feature_set: str,
    data_hash: str,
) -> ReleaseManifest:
    source_path = Path(source_dir)
    release_path = Path(release_dir)
    _validate_data_hash(data_hash)
    if source_path.is_symlink() or not source_path.is_dir():
        raise ValueError(f"release source must be a regular directory: {source_path}")
    source_paths = {name: source_path / name for name in RELEASE_ARTIFACT_NAMES}
    for name, path in source_paths.items():
        _require_regular_nonempty_file(path, f"release source artifact {name}")

    expected_names = {*RELEASE_ARTIFACT_NAMES, RELEASE_MANIFEST_FILENAME}
    if release_path.exists() or release_path.is_symlink():
        existing_entries = _release_directory_entries(release_path)
    else:
        release_path.mkdir(parents=True)
        existing_entries = []
    stale_final_paths = [path for path in existing_entries if path.name not in expected_names]

    token = uuid4().hex
    final_paths = {name: release_path / name for name in RELEASE_ARTIFACT_NAMES}
    manifest_final = release_path / RELEASE_MANIFEST_FILENAME
    staged_paths = {
        name: _temporary_sibling(final_paths[name], token, "new.staging")
        for name in RELEASE_ARTIFACT_NAMES
    }
    manifest_staged = _temporary_sibling(manifest_final, token, "new.staging")
    known_finals = [*(final_paths[name] for name in RELEASE_ARTIFACT_NAMES), manifest_final]
    previous_paths = [*known_finals, *stale_final_paths]
    backup_paths = {
        final_path: _recovery_backup_path(release_path, final_path, token)
        for final_path in previous_paths
    }
    temporary_paths = [*staged_paths.values(), manifest_staged, *backup_paths.values()]
    previous_outputs: dict[Path, Path | None] = {}
    publish_started = False
    manifest_committed = False

    try:
        for name in RELEASE_ARTIFACT_NAMES:
            copyfile(source_paths[name], staged_paths[name], follow_symlinks=False)
            _require_regular_nonempty_file(staged_paths[name], f"staged release artifact {name}")

        inventory = [
            ReleaseFile(
                key=name,
                path=name,
                sha256=sha256_file(staged_paths[name]),
                size_bytes=staged_paths[name].stat().st_size,
                media_type=RELEASE_ARTIFACT_MEDIA_TYPES[name],
            )
            for name in RELEASE_ARTIFACT_NAMES
        ]
        manifest = ReleaseManifest(
            version=version,
            feature_set=feature_set,
            model_file="calibrated_model.joblib",
            preprocessor_file="preprocessor.joblib",
            policy_file="policy.json",
            data_hash=data_hash,
            files=inventory,
        )
        save_manifest(manifest, manifest_staged)
        _require_regular_nonempty_file(manifest_staged, "staged release manifest")
        if load_manifest(manifest_staged) != manifest:
            raise RuntimeError("staged release manifest did not round trip")
        _validate_manifest_inventory(manifest, staged_paths)

        for final_path in previous_paths:
            if final_path.is_symlink() or (final_path.exists() and not final_path.is_file()):
                raise ValueError(f"existing release output must be a regular file: {final_path}")
            if final_path.is_file():
                backup_path = backup_paths[final_path]
                copyfile(final_path, backup_path, follow_symlinks=False)
                previous_outputs[final_path] = backup_path
            else:
                previous_outputs[final_path] = None

        publish_started = True
        for name in RELEASE_ARTIFACT_NAMES:
            staged_paths[name].replace(final_paths[name])
        _remove_stale_release_files(stale_final_paths)
        manifest_staged.replace(manifest_final)
        manifest_committed = True

        _cleanup_temporary_files(list(backup_paths.values()))
        _require_regular_nonempty_file(manifest_final, "published release manifest")
        if load_manifest(manifest_final) != manifest:
            raise RuntimeError("published release manifest did not match staged manifest")
        _validate_manifest_inventory(manifest, final_paths)
        return validate_release_bundle(release_path)
    except Exception as publication_error:
        recovery_notes: list[str] = []
        preserved_backups: set[Path] = set()
        if publish_started and not manifest_committed:
            recovery_notes, preserved_backups = _restore_previous_release(
                previous_paths,
                previous_outputs,
            )
        cleanup_notes = _cleanup_temporary_files(
            temporary_paths,
            preserve=preserved_backups,
        )
        _add_exception_notes(publication_error, [*recovery_notes, *cleanup_notes])
        raise
