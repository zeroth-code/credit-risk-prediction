from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.base import clone
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
    assert numeric_pipeline.named_steps["imputer"].keep_empty_features is True
    assert np.isnan(numeric_pipeline.named_steps["imputer"].missing_values)
    assert numeric_pipeline.named_steps["imputer"].get_params(deep=False) == {}
    assert isinstance(numeric_pipeline.named_steps["scaler"], StandardScaler)
    assert configured_numeric == numeric_columns

    categorical_pipeline, configured_categorical = transformers["categorical"]
    assert isinstance(categorical_pipeline, Pipeline)
    assert list(categorical_pipeline.named_steps) == ["imputer", "encoder"]
    assert isinstance(categorical_pipeline.named_steps["imputer"], SimpleImputer)
    assert categorical_pipeline.named_steps["imputer"].strategy == "most_frequent"
    assert categorical_pipeline.named_steps["imputer"].keep_empty_features is True
    assert categorical_pipeline.named_steps["imputer"].missing_values is pd.NA
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
    assert numeric_pipeline.named_steps["imputer"].keep_empty_features is True
    assert np.isnan(numeric_pipeline.named_steps["imputer"].missing_values)
    assert numeric_pipeline.named_steps["imputer"].get_params(deep=False) == {}
    assert "scaler" not in numeric_pipeline.named_steps
    assert configured_numeric == numeric_columns

    categorical_pipeline, configured_categorical = transformers["categorical"]
    assert isinstance(categorical_pipeline, Pipeline)
    assert list(categorical_pipeline.named_steps) == ["imputer", "encoder"]
    assert isinstance(categorical_pipeline.named_steps["imputer"], SimpleImputer)
    assert categorical_pipeline.named_steps["imputer"].strategy == "most_frequent"
    assert categorical_pipeline.named_steps["imputer"].keep_empty_features is True
    assert categorical_pipeline.named_steps["imputer"].missing_values is pd.NA
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
def test_preprocessor_preserves_single_all_missing_numeric_column(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame({"all_missing": [np.nan, np.nan]})
    future = pd.DataFrame({"all_missing": [np.nan]})
    preprocessor = factory(["all_missing"], [])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)

    assert train_result.shape == (2, 1)
    assert future_result.shape == (1, 1)
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()
    assert np.unique(train_result).tolist() == [0.0]
    assert future_result[0, 0] == 0.0
    assert preprocessor.get_feature_names_out().tolist() == ["numeric__all_missing"]


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_preserves_all_missing_numeric_column_alongside_observed_column(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 12_000.0],
            "all_missing": [np.nan, np.nan],
        }
    )
    future = pd.DataFrame({"loan_amnt": [14_000.0], "all_missing": [np.nan]})
    preprocessor = factory(["loan_amnt", "all_missing"], [])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)

    assert train_result.shape == (2, 2)
    assert future_result.shape == (1, 2)
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()
    assert preprocessor.get_feature_names_out().tolist() == [
        "numeric__loan_amnt",
        "numeric__all_missing",
    ]


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_preserves_single_all_missing_categorical_column(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame({"all_missing": [np.nan, np.nan]}, dtype="object")
    future = pd.DataFrame({"all_missing": [np.nan]}, dtype="object")
    preprocessor = factory([], ["all_missing"])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert train_result.shape[0] == 2
    assert future_result.shape[0] == 1
    assert train_result.shape[1] == future_result.shape[1] == len(feature_names) == 1
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()
    assert "all_missing" in feature_names[0]


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_preserves_all_missing_categorical_column_alongside_observed_column(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {
            "purpose": ["debt_consolidation", "debt_consolidation"],
            "all_missing": [np.nan, np.nan],
        },
        dtype="object",
    )
    future = pd.DataFrame(
        {"purpose": ["medical"], "all_missing": [np.nan]},
        dtype="object",
    )
    preprocessor = factory([], ["purpose", "all_missing"])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert train_result.shape[0] == 2
    assert future_result.shape[0] == 1
    assert train_result.shape[1] == future_result.shape[1] == len(feature_names) == 2
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()
    assert any("purpose" in name for name in feature_names)
    assert any("all_missing" in name for name in feature_names)


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_handles_nullable_string_missing_values(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {
            "purpose": pd.Series(
                ["debt_consolidation", pd.NA, "credit_card"],
                dtype="string",
            )
        }
    )
    future = pd.DataFrame({"purpose": pd.Series([pd.NA, "medical"], dtype="string")})
    preprocessor = factory([], ["purpose"])

    preprocessor.fit(train)
    imputed_train = (
        preprocessor.named_transformers_["categorical"].named_steps["imputer"].transform(train)
    )
    future_result = preprocessor.transform(future)

    assert not pd.isna(imputed_train).any()
    assert future_result.shape[0] == 2
    assert np.isfinite(future_result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_handles_mixed_object_categorical_missing_values(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {
            "purpose": [
                "debt_consolidation",
                None,
                np.nan,
                pd.NA,
                "debt_consolidation",
            ]
        },
        dtype="object",
    )
    future = pd.DataFrame(
        {"purpose": [None, np.nan, pd.NA, "medical"]},
        dtype="object",
    )
    preprocessor = factory([], ["purpose"])

    preprocessor.fit(train)
    imputed_train = (
        preprocessor.named_transformers_["categorical"].named_steps["imputer"].transform(train)
    )
    future_result = preprocessor.transform(future)

    assert imputed_train[:, 0].tolist() == ["debt_consolidation"] * len(train)
    assert future_result.shape[0] == len(future)
    assert np.isfinite(future_result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_handles_mixed_object_numeric_missing_values(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {"loan_amnt": [10_000.0, pd.NA, None, np.nan, 12_000.0]},
        dtype="object",
    )
    future = pd.DataFrame(
        {"loan_amnt": [pd.NA, None, np.nan, 14_000.0]},
        dtype="object",
    )
    preprocessor = factory(["loan_amnt"], [])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)

    assert train_result.shape == (len(train), 1)
    assert future_result.shape == (len(future), 1)
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_parses_numeric_strings_and_percent_literals(
    factory: PreprocessorFactory,
) -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": ["10000", " 12000.5 "],
            "int_rate": ["13.5%", " 7 % "],
        },
        dtype="object",
    )
    future = pd.DataFrame(
        {"loan_amnt": ["14000"], "int_rate": [" 9.25% "]},
        dtype="object",
    )
    preprocessor = factory(["loan_amnt", "int_rate"], [])

    preprocessor.fit(train)
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    normalized_train = imputer.transform(train)
    future_result = preprocessor.transform(future)

    np.testing.assert_allclose(
        normalized_train,
        [[10_000.0, 13.5], [12_000.5, 7.0]],
    )
    assert future_result.shape == (1, 2)
    assert np.isfinite(future_result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_rejects_unparseable_numeric_values_with_column_context(
    factory: PreprocessorFactory,
) -> None:
    frame = pd.DataFrame(
        {
            "loan_amnt": ["10000", "not-a-number"],
            "int_rate": ["13.5%", "14.0%"],
        },
        dtype="object",
    )
    preprocessor = factory(["loan_amnt", "int_rate"], [])

    with pytest.raises(ValueError) as error:
        preprocessor.fit(frame)

    message = str(error.value)
    assert "loan_amnt" in message
    assert "not-a-number" in message


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf, "inf", "-inf"])
def test_preprocessor_rejects_non_finite_numeric_values_with_column_context(
    factory: PreprocessorFactory,
    invalid_value: object,
) -> None:
    frame = pd.DataFrame(
        {"loan_amnt": [10_000.0, invalid_value]},
        dtype="object",
    )
    preprocessor = factory(["loan_amnt"], [])

    with pytest.raises(ValueError) as error:
        preprocessor.fit(frame)

    message = str(error.value)
    assert "loan_amnt" in message
    assert str(invalid_value) in message


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
    if numeric_columns:
        train = pd.DataFrame({"loan_amnt": [10_000.0, np.nan]})
        future = pd.DataFrame({"loan_amnt": [12_000.0]})
    else:
        train = pd.DataFrame({"purpose": ["debt_consolidation", None]})
        future = pd.DataFrame({"purpose": ["medical"]})

    preprocessor = factory(numeric_columns, categorical_columns)
    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)

    assert isinstance(preprocessor, ColumnTransformer)
    assert train_result.shape[0] == 2
    assert future_result.shape[0] == 1
    assert train_result.shape[1] == future_result.shape[1] > 0
    assert np.isfinite(train_result).all()
    assert np.isfinite(future_result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_preserves_sparse_column_transformer_output(
    factory: PreprocessorFactory,
) -> None:
    categories = np.repeat([f"purpose_{index}" for index in range(8)], 25)
    train = pd.DataFrame(
        {
            "loan_amnt": np.arange(1.0, len(categories) + 1.0),
            "purpose": categories,
        }
    )
    future = pd.DataFrame({"loan_amnt": [250.0], "purpose": ["unseen"]})
    preprocessor = factory(["loan_amnt"], ["purpose"])

    train_result = preprocessor.fit_transform(train)
    future_result = preprocessor.transform(future)

    assert issparse(train_result)
    assert issparse(future_result)
    assert train_result.format == "csr"
    assert future_result.format == "csr"
    assert np.isfinite(train_result.data).all()
    assert np.isfinite(future_result.data).all()


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


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_factory_snapshots_column_selectors(
    factory: PreprocessorFactory,
) -> None:
    numeric_columns = ["loan_amnt"]
    categorical_columns = ["purpose"]
    preprocessor = factory(numeric_columns, categorical_columns)

    numeric_columns.append("unused_numeric")
    categorical_columns.clear()
    transformers = {name: columns for name, _transformer, columns in preprocessor.transformers}

    assert transformers["numeric"] == ["loan_amnt"]
    assert transformers["categorical"] == ["purpose"]
    assert transformers["numeric"] is not numeric_columns
    assert transformers["categorical"] is not categorical_columns

    result = preprocessor.fit_transform(
        pd.DataFrame(
            {
                "loan_amnt": [10_000.0, 12_000.0],
                "purpose": ["debt_consolidation", "credit_card"],
            }
        )
    )
    assert result.shape[0] == 2


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_fitted_preprocessor_survives_joblib_round_trip(
    factory: PreprocessorFactory,
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "loan_amnt": ["10000", pd.NA, "12000"],
            "int_rate": ["13.5%", " 7 % ", None],
            "purpose": ["debt_consolidation", None, "credit_card"],
        },
        dtype="object",
    )
    future = pd.DataFrame(
        {
            "loan_amnt": ["14000"],
            "int_rate": ["9.25%"],
            "purpose": ["medical"],
        },
        dtype="object",
    )
    preprocessor = factory(["loan_amnt", "int_rate"], ["purpose"])
    preprocessor.fit(train)
    expected = preprocessor.transform(future)
    expected_names = preprocessor.get_feature_names_out()
    artifact_path = tmp_path / "preprocessor.joblib"

    joblib.dump(preprocessor, artifact_path)
    loaded = joblib.load(artifact_path)
    actual = loaded.transform(future)

    np.testing.assert_allclose(actual, expected)
    assert loaded.get_feature_names_out().tolist() == expected_names.tolist()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_and_numeric_imputer_are_sklearn_clone_compatible(
    factory: PreprocessorFactory,
) -> None:
    preprocessor = factory(["loan_amnt"], ["purpose"])
    cloned = clone(preprocessor)
    frame = pd.DataFrame(
        {
            "loan_amnt": ["10000", pd.NA, "12000"],
            "purpose": ["debt_consolidation", None, "credit_card"],
        },
        dtype="object",
    )

    result = cloned.fit_transform(frame)

    assert result.shape[0] == len(frame)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("factory", PREPROCESSOR_FACTORIES)
def test_preprocessor_handles_large_vectorized_numeric_input(
    factory: PreprocessorFactory,
) -> None:
    row_count = 10_000
    loan_amnt = pd.Series(np.arange(1_000, 1_000 + row_count), dtype="string")
    int_rate = pd.Series(np.arange(row_count) % 20 + 1, dtype="string") + "%"
    loan_amnt.iloc[::17] = pd.NA
    int_rate.iloc[::19] = pd.NA
    frame = pd.DataFrame({"loan_amnt": loan_amnt, "int_rate": int_rate})
    preprocessor = factory(["loan_amnt", "int_rate"], [])

    result = preprocessor.fit_transform(frame)

    assert result.shape == (row_count, 2)
    assert np.isfinite(result).all()
