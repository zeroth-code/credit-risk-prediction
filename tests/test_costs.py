import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from credit_risk.costs import assign_actions, policy_cost, search_policy


def test_assign_actions_uses_two_thresholds() -> None:
    actions = assign_actions(np.array([0.1, 0.4, 0.8]), approve_below=0.2, decline_at=0.7)
    assert actions.tolist() == ["approve", "manual_review", "decline"]


def test_assign_actions_uses_exact_threshold_boundaries() -> None:
    actions = assign_actions(
        np.array([0.199, 0.2, 0.699, 0.7]),
        approve_below=0.2,
        decline_at=0.7,
    )
    assert actions.tolist() == [
        "approve",
        "manual_review",
        "manual_review",
        "decline",
    ]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[0.2]]), "one-dimensional"),
        (np.array([]), "non-empty"),
        (np.array(["invalid"]), "numeric"),
        (np.array([np.nan]), "finite"),
        (np.array([np.inf]), "finite"),
        (np.array([-0.01]), "between 0 and 1"),
        (np.array([1.01]), "between 0 and 1"),
    ],
)
def test_assign_actions_rejects_invalid_probabilities(
    probabilities: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        assign_actions(probabilities, approve_below=0.2, decline_at=0.7)


@pytest.mark.parametrize(
    "probabilities",
    [
        True,
        np.bool_(False),
        np.array([True]),
        np.array([np.bool_(False)]),
        np.array([0.2, True], dtype=object),
    ],
)
def test_assign_actions_rejects_boolean_probabilities(probabilities: object) -> None:
    with pytest.raises(ValueError, match="probabilities.*boolean"):
        assign_actions(probabilities, approve_below=0.2, decline_at=0.7)


@pytest.mark.parametrize(
    ("approve_below", "decline_at"),
    [
        (-0.01, 0.7),
        (0.2, 0.2),
        (0.8, 0.7),
        (0.2, 1.01),
        (np.nan, 0.7),
        (0.2, np.inf),
    ],
)
def test_assign_actions_rejects_invalid_thresholds(approve_below: float, decline_at: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        assign_actions(
            np.array([0.5]),
            approve_below=approve_below,
            decline_at=decline_at,
        )


@pytest.mark.parametrize(
    ("approve_below", "decline_at"),
    [(False, 0.7), (np.bool_(False), 0.7), (0.2, True), (0.2, np.bool_(True))],
)
def test_assign_actions_rejects_boolean_thresholds(
    approve_below: object, decline_at: object
) -> None:
    with pytest.raises(ValueError, match="thresholds.*boolean"):
        assign_actions(
            np.array([0.5]),
            approve_below=approve_below,  # type: ignore[arg-type]
            decline_at=decline_at,  # type: ignore[arg-type]
        )


def test_policy_cost_applies_loan_level_costs() -> None:
    cost = policy_cost(
        y_true=np.array([1, 0, 0]),
        loan_amount=np.array([10000.0, 10000.0, 10000.0]),
        actions=np.array(["approve", "decline", "manual_review"]),
        lgd=0.60,
        margin=0.05,
        review_cost=30.0,
    )
    assert cost == pytest.approx(6530.0)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("y_true", np.array([[0, 1]])),
        ("loan_amount", np.array([[1000.0, 2000.0]])),
        ("actions", np.array([["approve", "decline"]])),
    ],
)
def test_policy_cost_rejects_non_one_dimensional_inputs(
    field: str, invalid_value: np.ndarray
) -> None:
    inputs = {
        "y_true": np.array([0, 1]),
        "loan_amount": np.array([1000.0, 2000.0]),
        "actions": np.array(["approve", "decline"]),
    }
    inputs[field] = invalid_value

    with pytest.raises(ValueError, match=f"{field}.*one-dimensional"):
        policy_cost(**inputs, lgd=0.6, margin=0.05, review_cost=30.0)


@pytest.mark.parametrize("field", ["y_true", "loan_amount", "actions"])
def test_policy_cost_rejects_empty_inputs(field: str) -> None:
    inputs = {
        "y_true": np.array([0]),
        "loan_amount": np.array([1000.0]),
        "actions": np.array(["approve"]),
    }
    inputs[field] = np.array([])

    with pytest.raises(ValueError, match=f"{field}.*non-empty"):
        policy_cost(**inputs, lgd=0.6, margin=0.05, review_cost=30.0)


@pytest.mark.parametrize("field", ["loan_amount", "actions"])
def test_policy_cost_rejects_misaligned_inputs(field: str) -> None:
    inputs = {
        "y_true": np.array([0, 1]),
        "loan_amount": np.array([1000.0, 2000.0]),
        "actions": np.array(["approve", "decline"]),
    }
    inputs[field] = inputs[field][:-1]

    with pytest.raises(ValueError, match="same length"):
        policy_cost(**inputs, lgd=0.6, margin=0.05, review_cost=30.0)


@pytest.mark.parametrize("target", [np.array([2]), np.array(["invalid"])])
def test_policy_cost_rejects_non_binary_target(target: np.ndarray) -> None:
    with pytest.raises(ValueError, match="y_true.*0 or 1"):
        policy_cost(
            target,
            np.array([1000.0]),
            np.array(["approve"]),
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize(
    "target", [True, np.bool_(False), np.array([True]), np.array([np.bool_(False)])]
)
def test_policy_cost_rejects_boolean_target(target: object) -> None:
    with pytest.raises(ValueError, match="y_true.*boolean"):
        policy_cost(
            target,
            np.array([1000.0]),
            np.array(["approve"]),
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize(
    ("amounts", "message"),
    [
        (np.array(["invalid"]), "numeric"),
        (np.array([np.nan]), "finite"),
        (np.array([np.inf]), "finite"),
        (np.array([-1.0]), "nonnegative"),
    ],
)
def test_policy_cost_rejects_invalid_loan_amounts(amounts: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=f"loan_amount.*{message}"):
        policy_cost(
            np.array([0]),
            amounts,
            np.array(["approve"]),
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize(
    "amounts", [True, np.bool_(False), np.array([True]), np.array([np.bool_(False)])]
)
def test_policy_cost_rejects_boolean_loan_amounts(amounts: object) -> None:
    with pytest.raises(ValueError, match="loan_amount.*boolean"):
        policy_cost(
            np.array([0]),
            amounts,
            np.array(["approve"]),
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize("actions", [np.array(["refer"]), np.array([1])])
def test_policy_cost_rejects_unknown_actions(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="actions.*approve.*manual_review.*decline"):
        policy_cost(
            np.array([0]),
            np.array([1000.0]),
            actions,
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize("action", [pd.NA, None, np.nan])
def test_policy_cost_rejects_missing_actions_with_field_name(action: object) -> None:
    with pytest.raises(ValueError, match="actions"):
        policy_cost(
            np.array([0]),
            np.array([1000.0]),
            np.array([action], dtype=object),
            lgd=0.6,
            margin=0.05,
            review_cost=30.0,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("lgd", -0.01),
        ("lgd", 1.01),
        ("lgd", np.nan),
        ("margin", -0.01),
        ("margin", 1.01),
        ("margin", np.inf),
        ("review_cost", -0.01),
        ("review_cost", np.nan),
        ("review_cost", np.inf),
    ],
)
def test_policy_cost_rejects_invalid_cost_parameters(field: str, invalid_value: float) -> None:
    costs = {"lgd": 0.6, "margin": 0.05, "review_cost": 30.0}
    costs[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        policy_cost(
            np.array([0]),
            np.array([1000.0]),
            np.array(["approve"]),
            **costs,
        )


@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("field", ["lgd", "margin", "review_cost"])
def test_policy_cost_rejects_boolean_cost_parameters(field: str, boolean: object) -> None:
    costs = {"lgd": 0.6, "margin": 0.05, "review_cost": 30.0}
    costs[field] = boolean

    with pytest.raises(ValueError, match=f"{field}.*boolean"):
        policy_cost(
            np.array([0]),
            np.array([1000.0]),
            np.array(["approve"]),
            **costs,
        )


def test_search_policy_evaluates_all_ordered_grid_thresholds() -> None:
    result = search_policy(
        y_true=np.array([0, 1, 0]),
        loan_amount=np.array([1000.0, 2000.0, 3000.0]),
        probabilities=np.array([0.1, 0.5, 0.9]),
        lgd=0.6,
        margin=0.05,
        review_cost=30.0,
    )

    assert result.columns.tolist() == [
        "approve_below",
        "decline_at",
        "cost",
        "approval_rate",
        "review_rate",
        "decline_rate",
    ]
    assert len(result) == 171
    assert result["cost"].is_monotonic_increasing
    assert np.allclose(
        result[["approval_rate", "review_rate", "decline_rate"]].sum(axis=1),
        1.0,
    )


def test_search_policy_uses_stable_grid_order_for_cost_ties() -> None:
    first = search_policy(
        y_true=np.array([0]),
        loan_amount=np.array([0.0]),
        probabilities=np.array([0.5]),
        lgd=0.0,
        margin=0.0,
        review_cost=0.0,
    )
    second = search_policy(
        y_true=np.array([0]),
        loan_amount=np.array([0.0]),
        probabilities=np.array([0.5]),
        lgd=0.0,
        margin=0.0,
        review_cost=0.0,
    )
    canonical_grid = np.arange(5, 100, 5, dtype=float) / 100.0
    expected_pairs = np.array(list(combinations(canonical_grid, 2)))

    assert first.equals(second)
    assert np.array_equal(
        first[["approve_below", "decline_at"]].to_numpy(),
        expected_pairs,
    )
    threshold_values = set(first[["approve_below", "decline_at"]].to_numpy().ravel())
    assert 0.4 in threshold_values
    assert 0.5 in threshold_values
    threshold_records = first[["approve_below", "decline_at"]].to_dict(orient="records")
    assert "0.39999999999999997" not in json.dumps(threshold_records)
    assert "0.39999999999999997" not in first.to_csv(index=False)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("y_true", np.array([[0]]), "y_true.*one-dimensional"),
        ("loan_amount", np.array([np.nan]), "loan_amount.*finite"),
        ("probabilities", np.array([1.1]), "probabilities.*between 0 and 1"),
        ("lgd", np.nan, "lgd"),
        ("margin", -0.01, "margin"),
        ("review_cost", np.inf, "review_cost"),
    ],
)
def test_search_policy_rejects_invalid_inputs(
    field: str, invalid_value: object, message: str
) -> None:
    inputs = {
        "y_true": np.array([0]),
        "loan_amount": np.array([1000.0]),
        "probabilities": np.array([0.5]),
        "lgd": 0.6,
        "margin": 0.05,
        "review_cost": 30.0,
    }
    inputs[field] = invalid_value

    with pytest.raises(ValueError, match=message):
        search_policy(**inputs)


@pytest.mark.parametrize(
    ("field", "boolean_value"),
    [
        ("y_true", np.array([True])),
        ("loan_amount", np.array([np.bool_(False)])),
        ("probabilities", np.array([True])),
        ("lgd", np.bool_(True)),
        ("margin", False),
        ("review_cost", True),
    ],
)
def test_search_policy_rejects_boolean_inputs(field: str, boolean_value: object) -> None:
    inputs = {
        "y_true": np.array([0]),
        "loan_amount": np.array([1000.0]),
        "probabilities": np.array([0.5]),
        "lgd": 0.6,
        "margin": 0.05,
        "review_cost": 30.0,
    }
    inputs[field] = boolean_value

    with pytest.raises(ValueError, match=f"{field}.*boolean"):
        search_policy(**inputs)


@pytest.mark.parametrize("field", ["loan_amount", "probabilities"])
def test_search_policy_rejects_misaligned_inputs(field: str) -> None:
    inputs = {
        "y_true": np.array([0, 1]),
        "loan_amount": np.array([1000.0, 2000.0]),
        "probabilities": np.array([0.2, 0.8]),
    }
    inputs[field] = inputs[field][:-1]

    with pytest.raises(ValueError, match="same length"):
        search_policy(**inputs, lgd=0.6, margin=0.05, review_cost=30.0)
