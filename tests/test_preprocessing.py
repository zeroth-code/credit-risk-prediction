from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.features import make_logistic_preprocessor, make_tree_preprocessor

PreprocessorFactory = Callable[[list[str], list[str]], ColumnTransformer]
PREPROCESSOR_FACTORIES = [make_logistic_preprocessor, make_tree_preprocessor]


def test_make_logistic_preprocessor_returns_column_transformer() -> None:
    preprocessor = make_logistic_preprocessor(["loan_amnt"], ["purpose"])

    assert isinstance(preprocessor, ColumnTransformer)


def test_make_logistic_preprocessor_builds_expected_pipelines() -> None:
    numeric_columns = ["annual_inc", "loan_amnt"]
    categorical_columns = ["grade", "purpose"]

    preprocessor = make_logistic_preprocessor(numeric_columns, categorical_columns)
    transformers = {
        name: (transformer, columns) for name, transformer, columns in preprocessor.transformers
    }

    assert preprocessor.remainder == "drop"
    assert list(transformers) == ["numeric", "categorical"]

    numeric_pipeline, configured_numeric = transformers["numeric"]
    assert isinstance(numeric_pipeline, Pipeline)
    assert list(numeric_pipeline.named_steps) == ["imputer", "scaler"]
    assert isinstance(numeric_pipeline.named_steps["imputer"], SimpleImputer)
    assert numeric_pipeline.named_steps["imputer"].strategy == "median"
    assert isinstance(numeric_pipeline.named_steps["scaler"], StandardScaler)
    assert configured_numeric == numeric_columns

    categorical_pipeline, configured_categorical = transformers["categorical"]
    assert isinstance(categorical_pipeline, Pipeline)
    assert list(categorical_pipeline.named_steps) == ["imputer", "encoder"]
    assert isinstance(categorical_pipeline.named_steps["imputer"], SimpleImputer)
    assert categorical_pipeline.named_steps["imputer"].strategy == "most_frequent"
    assert isinstance(categorical_pipeline.named_steps["encoder"], OneHotEncoder)
    assert categorical_pipeline.named_steps["encoder"].handle_unknown == "ignore"
    assert categorical_pipeline.named_steps["encoder"].min_frequency == 25
    assert configured_categorical == categorical_columns


def test_logistic_preprocessor_handles_missing_and_future_values() -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, np.nan],
            "purpose": ["debt_consolidation", None],
            "ignored": [1.0, 2.0],
        }
    )
    future = pd.DataFrame(
        {
            "loan_amnt": [12_000.0],
            "purpose": ["medical"],
            "ignored": [np.inf],
        }
    )
    preprocessor = make_logistic_preprocessor(["loan_amnt"], ["purpose"])

    preprocessor.fit(train)
    transformed = preprocessor.transform(future)
    numeric_train = preprocessor.named_transformers_["numeric"].transform(train[["loan_amnt"]])
    categorical_train = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["imputer"]
        .transform(train[["purpose"]])
    )

    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()
    assert numeric_train[:, 0].tolist() == [0.0, 0.0]
    assert categorical_train[:, 0].tolist() == ["debt_consolidation", "debt_consolidation"]


def test_make_tree_preprocessor_builds_expected_pipelines() -> None:
    numeric_columns = ["annual_inc", "loan_amnt"]
    categorical_columns = ["grade", "purpose"]

    preprocessor = make_tree_preprocessor(numeric_columns, categorical_columns)
    transformers = {
        name: (transformer, columns) for name, transformer, columns in preprocessor.transformers
    }

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.remainder == "drop"
    assert list(transformers) == ["numeric", "categorical"]

    numeric_pipeline, configured_numeric = transformers["numeric"]
    assert isinstance(numeric_pipeline, Pipeline)
    assert list(numeric_pipeline.named_steps) == ["imputer"]
    assert isinstance(numeric_pipeline.named_steps["imputer"], SimpleImputer)
    assert numeric_pipeline.named_steps["imputer"].strategy == "median"
    assert "scaler" not in numeric_pipeline.named_steps
    assert configured_numeric == numeric_columns

    categorical_pipeline, configured_categorical = transformers["categorical"]
    assert isinstance(categorical_pipeline, Pipeline)
    assert list(categorical_pipeline.named_steps) == ["imputer", "encoder"]
    assert isinstance(categorical_pipeline.named_steps["imputer"], SimpleImputer)
    assert categorical_pipeline.named_steps["imputer"].strategy == "most_frequent"
    assert isinstance(categorical_pipeline.named_steps["encoder"], OneHotEncoder)
    assert categorical_pipeline.named_steps["encoder"].handle_unknown == "ignore"
    assert categorical_pipeline.named_steps["encoder"].min_frequency == 25
    assert configured_categorical == categorical_columns


def test_tree_preprocessor_handles_missing_and_future_values_without_scaling() -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, np.nan],
            "purpose": ["debt_consolidation", None],
            "ignored": [1.0, 2.0],
        }
    )
    future = pd.DataFrame(
        {
            "loan_amnt": [12_000.0],
            "purpose": ["medical"],
            "ignored": [np.inf],
        }
    )
    preprocessor = make_tree_preprocessor(["loan_amnt"], ["purpose"])

    preprocessor.fit(train)
    transformed = preprocessor.transform(future)
    numeric_train = preprocessor.named_transformers_["numeric"].transform(train[["loan_amnt"]])
    categorical_train = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["imputer"]
        .transform(train[["purpose"]])
    )

    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] == 12_000.0
    assert numeric_train[:, 0].tolist() == [10_000.0, 10_000.0]
    assert categorical_train[:, 0].tolist() == ["debt_consolidation", "debt_consolidation"]


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_factory_rejects_empty_column_groups(
    factory: PreprocessorFactory,
) -> None:
    with pytest.raises(ValueError, match="at least one|empty"):
        factory([], [])


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
@pytest.mark.parametrize(
    ("numeric_columns", "categorical_columns", "duplicate"),
    [
        (["loan_amnt", "loan_amnt"], ["purpose"], "loan_amnt"),
        (["loan_amnt"], ["purpose", "purpose"], "purpose"),
    ],
)
def test_preprocessor_factory_rejects_duplicate_columns(
    factory: PreprocessorFactory,
    numeric_columns: list[str],
    categorical_columns: list[str],
    duplicate: str,
) -> None:
    with pytest.raises(ValueError) as error:
        factory(numeric_columns, categorical_columns)

    message = str(error.value)
    assert "duplicate" in message
    assert duplicate in message


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_factory_rejects_overlapping_columns(
    factory: PreprocessorFactory,
) -> None:
    with pytest.raises(ValueError) as error:
        factory(["loan_amnt", "shared"], ["purpose", "shared"])

    message = str(error.value)
    assert "overlap" in message
    assert "shared" in message


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
@pytest.mark.parametrize(
    ("numeric_columns", "categorical_columns"),
    [(["loan_amnt"], []), ([], ["purpose"])],
)
def test_preprocessor_factory_accepts_one_nonempty_column_group(
    factory: PreprocessorFactory,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    assert isinstance(factory(numeric_columns, categorical_columns), ColumnTransformer)


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_factory_preserves_input_lists(
    factory: PreprocessorFactory,
) -> None:
    numeric_columns = ["annual_inc", "loan_amnt"]
    categorical_columns = ["grade", "purpose"]
    original_numeric = numeric_columns.copy()
    original_categorical = categorical_columns.copy()

    factory(numeric_columns, categorical_columns)

    assert numeric_columns == original_numeric
    assert categorical_columns == original_categorical
