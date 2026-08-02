# Fairness and Subgroup Reliability Report

## Scope

This report evaluates whether the frozen test predictions and policy actions behave
consistently across four available operational groupings. It is a proxy reliability analysis,
not evidence that the model satisfies legal or ethical fairness requirements.

**This is not a statutory fair-lending audit.**

## Available Groups

| Attribute | Group construction | Usable groups |
| --- | --- | ---: |
| Income | Five `annual_inc` quantile bands computed on the frozen test partition; duplicate edges dropped; missing would be `Unknown` | 5 |
| Home ownership | Stripped `home_ownership`; missing/blank would be `Unknown` | 3 unsuppressed, 1 suppressed |
| Employment | Stripped `emp_length`; missing/blank becomes `Unknown` | 12 |
| Region | `addr_state` mapped to US Census regions; missing/unrecognized becomes `Unknown` | 4 |

These are model features or transformations of model features. They are useful for checking
reliability and action concentration, but they are not protected-class labels.

## Unavailable Protected Groups

The source does not provide the protected attributes required for a fair-lending audit,
including race, ethnicity, sex/gender, age, and other legally protected classes. The analysis
also cannot measure intersectional outcomes, compare applicant and accepted-loan populations,
or infer treatment of rejected applicants.

Income, home ownership, employment, and region must not be presented as substitutes for
protected attributes. Their results cannot establish disparate treatment, disparate impact,
causation, compliance, or absence of discrimination.

## Decision and Metric Definitions

The model target is `bad = 1` for default. The policy's favorable decision is `approve`.
`manual_review` and `decline` both count as not selected for these diagnostics.

| Metric | Definition in this report |
| --- | --- |
| Selection rate | Fraction of all group members assigned `approve` |
| True-positive rate | Approval rate among actually good/repaid loans |
| False-positive rate | Approval rate among actually bad/defaulted loans |
| Selection-rate ratio | Lowest defined group selection rate divided by the highest defined group selection rate |
| Equal-opportunity difference | Maximum minus minimum group true-positive rate among usable groups |
| ROC AUC | Within-group ranking performance for `bad = 1`; undefined for a single-class group |
| Brier score | Within-group mean squared error of calibrated default probability for `bad = 1` |

The true-positive/false-positive labels above follow favorable-decision semantics, not the
model's default-positive target semantics. This distinction is important when interpreting
the tables.

## Minimum Group Size and Suppression

The minimum reportable group size is 200. Groups below 200 are suppressed and excluded from
cross-group ratio/difference calculations. Suppression protects against unstable point
estimates; it does not make the remaining large-group estimates legally sufficient.

One home-ownership value, `ANY`, contains one test row and is suppressed. No other recorded
group is suppressed.

## Results

| Available grouping | Selection-rate ratio | Equal-opportunity difference | Suppressed groups |
| --- | ---: | ---: | ---: |
| Income band | 0.037063 | 0.230278 | 0 |
| Home ownership | 0.316112 | 0.103537 | 1 |
| Employment length | 0.525137 | 0.051544 | 0 |
| Region | 0.901025 | 0.010746 | 0 |

### Income

Income has the largest observed disparity across the available groupings. Selection rates rise
from 0.8227% in Income Q1 to 22.1977% in Income Q5, producing the 0.037063 ratio. The true
positive rate ranges from 0.9741% to 24.0018%, producing the 0.230278 difference. Bad rates,
ROC AUC, and Brier scores also vary across income bands. Because income is a model feature and
the bands are derived on the test partition, this result is descriptive and policy-sensitive,
not causal.

### Home Ownership

Among unsuppressed groups, selection rates are 14.1551% for `MORTGAGE`, 8.4120% for `OWN`, and
4.4746% for `RENT`. The one-row `ANY` group is not interpreted. Differences may reflect model
inputs, correlations, historical selection, or outcome mix; the report cannot distinguish
these mechanisms.

### Employment Length

Selection rates range from 5.5802% for `Unknown` to 10.6261% for `10+ years`. The `Unknown`
group also has the highest observed bad rate, 21.5770%, and the highest Brier score, 0.164636.
This is both a subgroup reliability signal and a missing-data monitoring concern.

### Region

Regional selection rates are closer together, from 8.8961% in the Midwest to 9.8733% in the
West. The 0.901025 ratio and 0.010746 equal-opportunity difference are more stable than the
other available grouping results, but they do not demonstrate protected-class fairness.

## Instability and Interpretation Warnings

- The policy approves only 9.3553% of test loans, routes 90.6447% to manual review, and declines
  none. Fairness metrics therefore describe an unusual, review-heavy demonstration policy.
- Selection-rate ratios can become extreme when one group has a very low approval rate, as in
  the income result.
- Point estimates have no confidence intervals in this release. Differences may reflect
  sampling variation, especially near the minimum group size.
- Income bands are computed from the frozen test population, so their boundaries are specific
  to this sample and may shift in another population.
- Group outcome prevalence differs. Selection differences do not by themselves identify
  unfair treatment or a causal mechanism.
- Historical approved-loan selection prevents analysis of the full applicant funnel and can
  distort both performance and fairness conclusions.
- Missing protected attributes prevent statutory comparisons and intersectional analysis.
- SHAP and subgroup metrics describe associations only and must not be used as legal
  explanations.

## Responsible Next Steps

A real lending review would require protected-attribute data collected and governed for the
appropriate legal purpose, counsel-approved metrics, confidence intervals, intersectional
analysis, applicant-funnel coverage, adverse-action testing, policy-capacity analysis,
independent validation, and ongoing monitoring. None of those requirements is satisfied by
this portfolio report.
