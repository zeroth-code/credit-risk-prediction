#!/usr/bin/env bash
# Print the uv.lock pins for the packages the deployed application imports.
#
# Streamlit Community Cloud does not read pyproject.toml or uv.lock, so requirements.txt is
# the deployment dependency file. The release model is a joblib pickle, so the deployed
# scikit-learn and lightgbm versions must match the versions that trained it.
#
# Compare this output against requirements.txt after changing dependencies;
# tests/test_requirements.py fails when the two drift apart.
set -euo pipefail

cd "$(dirname "$0")/.."

uv export --no-dev --no-emit-project --format requirements-txt --no-hashes \
  | grep -vE '^\s*#' \
  | grep -iE '^(joblib|lightgbm|numpy|pandas|plotly|pydantic|pyyaml|scikit-learn|scipy|streamlit)=='
