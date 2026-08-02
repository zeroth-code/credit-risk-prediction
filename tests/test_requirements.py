"""Guard the deployment dependency file against drift from uv.lock.

Streamlit Community Cloud installs requirements.txt and ignores uv.lock. The release model is
a joblib pickle, so a scikit-learn or lightgbm version that differs from the training
environment can fail to load or silently change predictions.
"""

import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
LOCK_PATH = PROJECT_ROOT / "uv.lock"

# Imported by app/streamlit_app.py and src/credit_risk/demo.py at runtime, plus the
# libraries required to unpickle the release model.
REQUIRED_PACKAGES = frozenset(
    {
        "joblib",
        "lightgbm",
        "numpy",
        "pandas",
        "plotly",
        "pydantic",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "streamlit",
    }
)


def parse_requirements(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise AssertionError(f"requirements.txt entry is not an exact pin: {line}")
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()
    return pins


def locked_versions() -> dict[str, str]:
    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {
        package["name"].lower(): package["version"]
        for package in lock["package"]
        if "version" in package
    }


@pytest.fixture(scope="module")
def requirements() -> dict[str, str]:
    return parse_requirements(REQUIREMENTS_PATH.read_text(encoding="utf-8"))


def test_requirements_cover_every_runtime_package(requirements: dict[str, str]) -> None:
    missing = sorted(REQUIRED_PACKAGES - set(requirements))
    assert not missing, f"requirements.txt is missing runtime packages: {', '.join(missing)}"


def test_requirements_match_the_locked_versions(requirements: dict[str, str]) -> None:
    locked = locked_versions()
    drifted = {
        name: (pinned, locked.get(name))
        for name, pinned in requirements.items()
        if locked.get(name) != pinned
    }
    assert not drifted, (
        f"requirements.txt drifted from uv.lock; rerun scripts/export_requirements.sh: {drifted}"
    )
