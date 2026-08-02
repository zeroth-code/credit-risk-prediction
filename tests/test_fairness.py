import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.fairness import (
    FAIRNESS_COLUMNS,
    STATE_TO_REGION,
    build_fairness_diagnostics,
    census_region,
    income_band,
    subgroup_metrics,
    suppress_small_groups,
)


def test_income_band_assigns_ordered_categories() -> None:
    values = pd.Series([20_000, 40_000, 60_000, 80_000, 100_000])

    bands = income_band(values, quantiles=5)

    assert bands.notna().all()
    assert bands.nunique() == 5
    assert bands.cat.ordered


def test_suppress_small_groups_marks_unstable_metrics() -> None:
    metrics = pd.DataFrame({"group": ["small", "large"], "count": [50, 500], "auc": [0.8, 0.7]})

    result = suppress_small_groups(metrics, minimum_size=200)

    assert pd.isna(result.loc[result["group"] == "small", "auc"]).all()
    assert result.loc[result["group"] == "small", "suppressed"].item()


def test_income_band_handles_duplicate_edges_and_missing_values_deterministically() -> None:
    values = pd.Series([10_000.0, 10_000.0, 20_000.0, 20_000.0, 30_000.0, np.nan])

    first = income_band(values, quantiles=5)
    second = income_band(values, quantiles=5)

    pd.testing.assert_series_equal(first, second)
    assert first.cat.ordered
    assert first.iloc[-1] == "Unknown"
    assert first.astype("string").tolist() == [
        "Income Q1",
        "Income Q1",
        "Income Q2",
        "Income Q2",
        "Income Q3",
        "Unknown",
    ]


def test_income_band_assigns_constant_nonmissing_values_to_one_band() -> None:
    bands = income_band(pd.Series([50_000.0, 50_000.0, np.nan]), quantiles=5)

    assert bands.astype("string").tolist() == ["Income Q1", "Income Q1", "Unknown"]


def test_income_band_preserves_duplicate_indexes_without_affecting_assignment() -> None:
    values = pd.Series([10_000.0, 20_000.0, np.nan, 30_000.0], index=[0, 0, 1, 1])

    bands = income_band(values, quantiles=3)

    assert bands.index.tolist() == [0, 0, 1, 1]
    assert bands.astype("string").tolist() == [
        "Income Q1",
        "Income Q2",
        "Unknown",
        "Income Q3",
    ]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([20_000, True], dtype=object),
        pd.Series(["20000", "40000"]),
        pd.Series([20_000.0, np.inf]),
    ],
    ids=["boolean", "numeric-strings", "infinite"],
)
def test_income_band_rejects_non_real_or_nonfinite_income(values: pd.Series) -> None:
    with pytest.raises(ValueError, match="income"):
        income_band(values)


@pytest.mark.parametrize("quantiles", [True, 0, -1, 2.5])
def test_income_band_rejects_invalid_quantile_count(quantiles: object) -> None:
    with pytest.raises(ValueError, match="quantiles"):
        income_band(pd.Series([10_000, 20_000]), quantiles=quantiles)  # type: ignore[arg-type]


def test_census_region_normalizes_states_and_unknowns() -> None:
    states = pd.Series([" ny ", "ca", "DC", "zz", "", None, pd.NA])

    regions = census_region(states)

    assert regions.tolist() == [
        "Northeast",
        "West",
        "South",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
    ]
    assert len(STATE_TO_REGION) == 51
    assert set(STATE_TO_REGION.values()) == {"Northeast", "Midwest", "South", "West"}


def test_census_region_rejects_non_string_nonmissing_states() -> None:
    with pytest.raises(ValueError, match="addr_state"):
        census_region(pd.Series(["NY", 123], dtype=object))


def test_suppress_small_groups_preserves_identity_and_does_not_mutate_input() -> None:
    metrics = pd.DataFrame(
        {
            "group": ["large", "small"],
            "count": [200, 199],
            "bad_rate": [0.1, 0.2],
            "selection_rate": [0.9, 0.8],
        }
    )
    original = metrics.copy(deep=True)

    result = suppress_small_groups(metrics, minimum_size=200)

    pd.testing.assert_frame_equal(metrics, original)
    assert result.loc[0].to_dict() == {
        "group": "large",
        "count": 200,
        "bad_rate": 0.1,
        "selection_rate": 0.9,
        "suppressed": False,
    }
    assert result.loc[1, ["bad_rate", "selection_rate"]].isna().all()
    assert bool(result.loc[1, "suppressed"])


@pytest.mark.parametrize(
    ("metrics", "minimum_size", "message"),
    [
        (pd.DataFrame({"group": ["a"]}), 1, "count"),
        (pd.DataFrame({"count": [1]}), 1, "group"),
        (pd.DataFrame({"group": ["a", "a"], "count": [1, 2]}), 1, "unique"),
        (pd.DataFrame({"group": ["a"], "count": [-1]}), 1, "count"),
        (pd.DataFrame({"group": ["a"], "count": [1.5]}), 1, "count"),
        (pd.DataFrame({"group": ["a"], "count": [1]}), True, "minimum_size"),
        (pd.DataFrame({"group": ["a"], "count": [1]}), 0, "minimum_size"),
    ],
    ids=[
        "missing-count",
        "missing-group",
        "duplicate-group",
        "negative-count",
        "fractional-count",
        "boolean-minimum",
        "zero-minimum",
    ],
)
def test_suppress_small_groups_validates_inputs(
    metrics: pd.DataFrame,
    minimum_size: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        suppress_small_groups(metrics, minimum_size=minimum_size)  # type: ignore[arg-type]


def test_subgroup_metrics_uses_good_outcome_and_approval_as_favorable() -> None:
    y_bad = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.7, 0.9, 0.2, 0.3, 0.6, 0.8])
    actions = np.array(
        [
            "approve",
            "manual_review",
            "approve",
            "decline",
            "approve",
            "approve",
            "manual_review",
            "decline",
        ]
    )
    groups = pd.Series(["A"] * 4 + ["B"] * 4)

    table = subgroup_metrics(y_bad, probabilities, actions, groups, minimum_size=1)

    assert table.columns.tolist() == FAIRNESS_COLUMNS
    assert table["group"].tolist() == ["A", "B"]
    group_a = table.loc[table["group"] == "A"].iloc[0]
    group_b = table.loc[table["group"] == "B"].iloc[0]
    assert group_a["count"] == 4
    assert group_a["bad_rate"] == pytest.approx(0.5)
    assert group_a["selection_rate"] == pytest.approx(0.5)
    assert group_a["true_positive_rate"] == pytest.approx(0.5)
    assert group_a["false_positive_rate"] == pytest.approx(0.5)
    assert group_a["roc_auc"] == pytest.approx(1.0)
    assert group_a["brier_score"] == pytest.approx(0.0675)
    assert group_b["selection_rate"] == pytest.approx(0.5)
    assert group_b["true_positive_rate"] == pytest.approx(1.0)
    assert group_b["false_positive_rate"] == pytest.approx(0.0)
    assert group_b["roc_auc"] == pytest.approx(1.0)
    assert group_b["brier_score"] == pytest.approx(0.0825)


def test_subgroup_metrics_handles_one_class_groups_without_warnings() -> None:
    table = subgroup_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array(["approve", "decline", "approve", "decline"]),
        pd.Series(["good_only", "good_only", "bad_only", "bad_only"]),
        minimum_size=1,
    )

    good_only = table.loc[table["group"] == "good_only"].iloc[0]
    bad_only = table.loc[table["group"] == "bad_only"].iloc[0]
    assert pd.isna(good_only["false_positive_rate"])
    assert pd.isna(good_only["roc_auc"])
    assert good_only["brier_score"] == pytest.approx(0.025)
    assert pd.isna(bad_only["true_positive_rate"])
    assert pd.isna(bad_only["roc_auc"])
    assert bad_only["brier_score"] == pytest.approx(0.025)


@pytest.mark.parametrize(
    ("y_bad", "probabilities", "actions", "groups", "message"),
    [
        ([0, 1], [0.1], ["approve", "decline"], ["A", "B"], "align"),
        ([0, 2], [0.1, 0.2], ["approve", "decline"], ["A", "B"], "0 and 1"),
        ([False, True], [0.1, 0.2], ["approve", "decline"], ["A", "B"], "boolean"),
        ([0, 1], [0.1, np.nan], ["approve", "decline"], ["A", "B"], "finite"),
        ([0, 1], [0.1, 1.1], ["approve", "decline"], ["A", "B"], "between"),
        ([0, 1], [0.1, 0.2], ["approve", "refer"], ["A", "B"], "action"),
        ([0, 1], [0.1, 0.2], ["approve", "decline"], ["A", None], "group"),
        ([0, 1], [0.1, 0.2], ["approve", "decline"], ["A", " "], "group"),
    ],
    ids=[
        "misaligned",
        "nonbinary-target",
        "boolean-target",
        "nonfinite-probability",
        "out-of-range-probability",
        "unknown-action",
        "missing-group",
        "blank-group",
    ],
)
def test_subgroup_metrics_validates_inputs(
    y_bad: object,
    probabilities: object,
    actions: object,
    groups: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        subgroup_metrics(y_bad, probabilities, actions, pd.Series(groups), minimum_size=1)


def _diagnostic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_inc": [20_000.0, 40_000.0, 60_000.0, 80_000.0, 100_000.0],
            "home_ownership": [" RENT ", "RENT", "MORTGAGE", "", None],
            "addr_state": ["ny", " CA ", "TX", "zz", None],
            "emp_length": ["1 year", " 1 year ", "10+ years", "", None],
        }
    )


def test_build_fairness_diagnostics_is_deterministic_and_self_describing() -> None:
    frame = _diagnostic_frame()
    original = frame.copy(deep=True)
    y_bad = np.array([0, 1, 0, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.3])
    actions = np.array(["approve", "decline", "approve", "manual_review", "approve"])

    first_tables, first_summary = build_fairness_diagnostics(
        frame,
        y_bad,
        probabilities,
        actions,
        minimum_group_size=1,
    )
    second_tables, second_summary = build_fairness_diagnostics(
        frame,
        y_bad,
        probabilities,
        actions,
        minimum_group_size=1,
    )

    pd.testing.assert_frame_equal(frame, original)
    assert list(first_tables) == ["income", "home_ownership", "region", "employment"]
    for name in first_tables:
        pd.testing.assert_frame_equal(first_tables[name], second_tables[name])
        assert first_tables[name].columns.tolist() == FAIRNESS_COLUMNS
        assert first_tables[name]["group"].tolist() == sorted(first_tables[name]["group"])
    assert first_summary == second_summary
    json.dumps(first_summary, sort_keys=True, allow_nan=False)
    assert first_summary["schema_version"] == "1.0"
    assert first_summary["minimum_group_size"] == 1
    assert first_summary["metric_semantics"] == {
        "target": "bad=1 (default)",
        "favorable_ground_truth_outcome": "good/repaid (1 - bad)",
        "favorable_decision": "action == approve",
        "not_selected_actions": ["manual_review", "decline"],
        "probability": "frozen calibrated default probability",
        "selection_rate": "overall approval rate",
        "true_positive_rate": "approval rate among actually good/repaid loans",
        "false_positive_rate": "approval rate among actually bad/defaulted loans",
        "roc_auc_and_brier_score_target": "bad=1 (default)",
    }
    assert "proxy subgroup reliability diagnostics" in first_summary["limitations"]
    assert "not a statutory fair-lending audit" in first_summary["limitations"]
    assert set(first_summary["attributes"]) == {
        "income",
        "home_ownership",
        "region",
        "employment",
    }
    assert first_summary["attributes"]["income"]["output_file"] == "fairness_income.csv"
    assert first_summary["attributes"]["home_ownership"]["group_definition"].startswith(
        "home_ownership"
    )
    assert first_tables["home_ownership"]["group"].tolist() == [
        "MORTGAGE",
        "RENT",
        "Unknown",
    ]
    assert first_tables["employment"]["group"].tolist() == [
        "1 year",
        "10+ years",
        "Unknown",
    ]
    assert first_tables["region"]["group"].tolist() == sorted(
        ["Northeast", "West", "South", "Unknown"]
    )


@pytest.mark.parametrize("column", ["home_ownership", "emp_length"])
def test_build_fairness_diagnostics_rejects_non_string_text_groups(column: str) -> None:
    frame = _diagnostic_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = 123

    with pytest.raises(ValueError, match=column):
        build_fairness_diagnostics(
            frame,
            np.array([0, 1, 0, 1, 0]),
            np.array([0.1, 0.8, 0.2, 0.7, 0.3]),
            np.array(["approve", "decline", "approve", "manual_review", "approve"]),
            minimum_group_size=1,
        )


def test_suppressed_groups_do_not_influence_summary_disparities() -> None:
    frame = pd.DataFrame(
        {
            "annual_inc": np.arange(9, dtype=float),
            "home_ownership": ["A"] * 4 + ["B"] * 4 + ["C"],
            "addr_state": ["NY"] * 4 + ["CA"] * 4 + ["TX"],
            "emp_length": ["A"] * 4 + ["B"] * 4 + ["C"],
        }
    )
    y_bad = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.2, 0.7, 0.8, 0.05])
    actions = np.array(
        [
            "approve",
            "decline",
            "decline",
            "decline",
            "approve",
            "approve",
            "decline",
            "decline",
            "decline",
        ]
    )

    tables, summary = build_fairness_diagnostics(
        frame,
        y_bad,
        probabilities,
        actions,
        minimum_group_size=2,
    )

    home = tables["home_ownership"]
    assert home.loc[home["group"] == "C", "suppressed"].item()
    home_summary = summary["attributes"]["home_ownership"]
    assert home_summary["equal_opportunity_difference"] == {
        "status": "defined",
        "value": pytest.approx(0.5),
        "reason": None,
    }
    assert home_summary["selection_rate_ratio"] == {
        "status": "defined",
        "value": pytest.approx(0.5),
        "reason": None,
    }


def test_summary_reports_too_few_usable_groups_and_zero_ratio_denominator() -> None:
    frame = pd.DataFrame(
        {
            "annual_inc": [10_000.0, 20_000.0, 30_000.0, 40_000.0],
            "home_ownership": ["A", "A", "B", "B"],
            "addr_state": ["NY", "NY", "CA", "CA"],
            "emp_length": ["A", "A", "B", "B"],
        }
    )
    y_bad = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    _, too_small = build_fairness_diagnostics(
        frame,
        y_bad,
        probabilities,
        np.array(["approve", "decline", "decline", "decline"]),
        minimum_group_size=3,
    )
    _, zero_selection = build_fairness_diagnostics(
        frame,
        y_bad,
        probabilities,
        np.array(["decline"] * 4),
        minimum_group_size=1,
    )

    too_small_home = too_small["attributes"]["home_ownership"]
    assert too_small_home["equal_opportunity_difference"] == {
        "status": "undefined",
        "value": None,
        "reason": "fewer_than_two_usable_unsuppressed_groups",
    }
    zero_home = zero_selection["attributes"]["home_ownership"]
    assert zero_home["selection_rate_ratio"] == {
        "status": "undefined",
        "value": None,
        "reason": "maximum_selection_rate_is_zero",
    }
