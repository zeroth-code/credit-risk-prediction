# Case Study: Temporal Credit Risk Prediction

## Situation

A historical consumer-loan portfolio is materially imbalanced: only 14.8863% of the 2015
out-of-time test loans default. A superficially strong accuracy result can therefore hide a
useless classifier. At the conventional 0.5 threshold, the frozen calibrated model predicts
every test loan as non-default, producing 240,894 true negatives, 42,132 false negatives, and
no positive predictions.

The data adds a second challenge. It contains accepted LendingClub loans with observed final
statuses, not all applicants, so prior underwriting and outcome maturity shape the sample.

## Task

Build a reproducible application-time risk demonstration that estimates calibrated default
probabilities under temporal drift, reserves the 2015 holdout for reported final evaluation,
and connects model evidence to a transparent business-cost policy. The result must remain
honest about modest discrimination, operational workload, selection bias, and missing
protected attributes.

## Action

The work followed a leakage-first sequence:

1. Audit the 2,260,701-row raw snapshot and filter to 1,020,768 eligible 36-month final-status
   loans. Assign 603,589 rows to chronological train, validation, calibration, and test windows.
2. Exclude post-origination payments, balances, recoveries, collections, and later credit-pull
   fields. Use a 17-field challenger that also omits lender-decision variables `int_rate`,
   `grade`, and `sub_grade`.
3. Fit preprocessing and models on 2011-2013 train data; compare dummy, logistic, LightGBM,
   natural weighting, class weighting, and undersampling on first-half 2014 validation data.
4. Select weighted challenger LightGBM by validation average precision and tune it with 30
   Optuna trials. The selected model uses 894 estimators, learning rate 0.0107234, and 33 leaves.
5. Compare uncalibrated, sigmoid, and isotonic probabilities on second-half 2014 calibration
   data using 5-fold stratified out-of-fold predictions. Select sigmoid and refit it on the full
   calibration partition.
6. Select approve/review/decline thresholds on calibration probabilities under explicit loss,
   margin, and review-cost assumptions. Freeze approve below 5% PD and decline at 45% PD.
7. Report final evaluation on the reserved 2015 test holdout with bootstrap intervals, monthly
   temporal slices, SHAP associations, and subgroup reliability diagnostics.

## Result

On 283,026 test loans, ROC AUC is 0.662707 (nominal 95% percentile CI
0.660260-0.665304), average precision is 0.241568 (nominal 95% percentile CI
0.238800-0.244326), and Brier score is 0.121539 (nominal 95% percentile CI
0.121380-0.121700). Log loss is 0.399637, KS is 0.236654, and expected calibration error is
0.011108. ECE uses 10 equal-width probability bins and includes the upper endpoint in the final
bin. These figures show measurable ranking signal and usable probability evidence, but only
modest discrimination.

The implemented equation is:

```text
proxy_cost = LGD * sum(loan_amnt for approved bad loans)
           + margin * sum(loan_amnt for declined good loans)
           + review_cost * count(manual_review)
```

Under the stated base scenario, the partial scenario-cost proxy is USD 16.272 million for USD
3.624 billion of exposure, or USD 57,493.53 per 1,000 applications. The 27 sensitivity proxies
range from USD 29,985.60 to USD 87,387.52 per 1,000. No validated comparator policy is present,
so the case study makes no unsupported cost-improvement claim.

The operating result is intentionally visible: 9.3553% approve, 90.6447% manual review, and 0%
decline. SHAP identifies annual income, `fico_range_low` (the lower bound of the reported FICO
range), recent inquiries, loan amount, and DTI as leading mean-absolute associations; this does
not establish direction. Subgroup diagnostics expose large income and home-ownership
selection-rate differences, while explicitly stopping short of a legal audit.

## Limitation

The project is trained on a historical selected sample of approved loans and cannot observe
rejected-applicant outcomes. Protected attributes are absent, preventing statutory
fair-lending analysis and intersectional validation.

Manual review is modeled as a terminal USD 30 fee only. The downstream reviewer approval or
decline, credit loss, and foregone margin are omitted, so the cost result is not a complete
operating cost. This is a material limitation because 90.6% of test applications route to
review.

The result is a portfolio demonstration of disciplined temporal evaluation, calibration, cost
policy, explainability, and reliability reporting. It is not a causal model, a legal compliance
assessment, or a production lending decision system.
