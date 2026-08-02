# Model Card: Calibrated LightGBM Credit Risk Challenger

## Model Details

| Field | Value |
| --- | --- |
| Release | `0.1.0` |
| Feature set | Challenger, 17 application/underwriting-time fields |
| Preprocessor | scikit-learn `ColumnTransformer` |
| Base estimator | Weighted LightGBM binary classifier |
| Calibration | Sigmoid `CalibratedClassifierCV` |
| Manifest contents | 18 files with SHA-256 and byte-size checks |
| Raw data hash | `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a` |

The model predicts `bad = 1`, where `Charged Off` and `Default` are bad and `Fully Paid` is
good. It is a historical portfolio demonstration, not an approved lending model.

## Intended Use

- Demonstrate leakage-aware temporal credit-risk modeling.
- Estimate historical default probabilities for records with the release schema.
- Compare discrimination, calibration, policy cost scenarios, SHAP associations, and subgroup
  reliability.
- Support portfolio review, reproducibility, and technical discussion with human oversight.

## Prohibited Use

- Do not use the model for autonomous approval, denial, pricing, credit-limit, or collections
  decisions.
- Do not use outputs as adverse-action reasons or a substitute for legally compliant reason
  codes.
- Do not use this release for statutory fair-lending conclusions.
- Do not deploy it to a new institution, product, geography, population, or time period without
  independent validation and governance approval.
- Do not interpret model scores or SHAP values as causal effects.
- Do not load joblib artifacts from an untrusted source; they use Python pickle semantics.

## Models Compared

The validation study includes:

- a prior-probability dummy classifier;
- natural and class-balanced logistic regression on challenger and full-underwriting fields;
- natural, positive-class-weighted, and randomly undersampled LightGBM on challenger fields;
- natural LightGBM on the full-underwriting fields.

The release is intentionally tied to the challenger feature set. Among challenger LightGBM
strategies, validation average precision is 0.215795 for natural training, 0.215990 for class
weighting, and 0.208898 for random undersampling. The weighted strategy is selected and tuned
for average precision using 30 Optuna trials. The fitted positive-class weight is 6.987513.

The best trial uses 894 estimators, learning rate 0.0107234, 33 leaves, minimum child size 21,
subsample 0.807274, column sample 0.818077, L1 regularization 4.951069, and L2 regularization
0.059330. Its validation average precision is 0.217576.

Natural full-underwriting LightGBM reaches validation average precision 0.226573, but its
feature set contains `int_rate`, `grade`, and `sub_grade`, which encode lender underwriting
decisions and are excluded from the release challenger.

An offline probe on the validation partition measured the headroom in the unused
application-time bureau columns, which are neither post-origination nor part of LendingClub's
own assessment. Adding roughly 35 of them lifts validation ROC AUC from 0.6575 to 0.6779,
a gain of +0.0204 with a 95% bootstrap interval of [+0.0176, +0.0234]. Restricting training to
2013, where those columns are 99% populated rather than 76% across 2011-2013, preserves the
lift at +0.0197, so it is not an imputation artifact. This is documented headroom only; the
released model does not use these features, and adopting them would require re-running
calibration, policy selection, fairness, and explainability.

## Preprocessing

The release preprocessor is fitted on train only:

- numeric strings are stripped, optional percent suffixes are removed, and values are
  validated as finite numbers;
- numeric fields are median imputed; logistic experiments additionally standardize them;
- categorical fields are most-frequent imputed and one-hot encoded;
- unknown categories are ignored and categories with frequency below 25 are grouped;
- all unlisted columns are dropped.

The frozen LightGBM uses the tree preprocessor, so its numeric columns are not standardized.

## Validation Design and Partition Roles

| Partition | Dates | Rows | Reported role |
| --- | --- | ---: | --- |
| Train | 2011-01 through 2013-12 | 157,993 | Fit preprocessors and base estimators |
| Validation | 2014-01 through 2014-06 | 71,955 | Select experiments, imbalance strategy, and Optuna parameters |
| Calibration | 2014-07 through 2014-12 | 90,615 | Compare/refit calibration and select cost thresholds |
| Test | 2015-01 through 2015-12 | 283,026 | Reported final evaluation and post-selection diagnostics |

The split is chronological by issue date. The documented protocol reserves test for final
evaluation rather than model, calibration, or threshold selection.

## Calibration

Uncalibrated, sigmoid, and isotonic probabilities are evaluated on calibration. Calibrated
methods use 5-fold stratified out-of-fold predictions for method comparison and policy
threshold selection. Sigmoid is selected, then refit on the full calibration partition.

| Calibration method | Brier score | Log loss | ECE | Probability source |
| --- | ---: | ---: | ---: | --- |
| Uncalibrated | 0.231948 | 0.652289 | 0.329800 | Base-model calibration holdout |
| Sigmoid | 0.116482 | 0.386624 | 0.000448 | 5-fold stratified OOF |
| Isotonic | 0.116514 | 0.387128 | 0.000405 | 5-fold stratified OOF |

Sigmoid is selected by the calibration selection rule and is the method recorded in the
frozen policy and release manifest.

ECE is the sample-weighted absolute gap between mean predicted PD and observed default rate
across 10 equal-width probability bins, with the final bin including its upper endpoint. The
calibration table uses 5-fold OOF predictions at 14.1257% prevalence; the final test metrics
use a single sigmoid calibrator refit on all calibration rows and a holdout at 14.8863%
prevalence. Brier score is prevalence sensitive, and the evaluation protocols differ, so its
increase on test is not proof of calibration drift. The later test ECE of 0.011108, compared
with calibration OOF ECE 0.000448, is consistent with degraded temporal calibration but is
not definitive evidence by itself.

### Output probability range

The fitted sigmoid is `P = 1 / (1 + exp(-1.0195 * z + 1.8076))`, where `z` is the LightGBM
decision function in log-odds units. The calibrator is unbounded and no code clamps the
output, but `z` spans only -3.553 to +1.519 across the 283,026 test applications, so published
probabilities span 0.4362% to 43.5550%. The positive tail is the binding constraint: reaching
the 0.45 decline threshold requires `z >= 1.576`, which no real test application attains.

This is a consequence of modest discrimination. At ROC AUC 0.662707 the features do not
separate defaulters sharply enough to produce large positive margins, and sigmoid calibration
anchors the output near the 14.8863% base rate rather than manufacturing unearned confidence.
Test ECE 0.011108 indicates the compressed probabilities are accurate. Deliberately
constructed extreme applications do leave the band, scoring 0.62% and 49.73%, so the range
describes realistic applications rather than a hard bound.

## Selected Thresholds

Thresholds are grid-searched on sigmoid calibration out-of-fold probabilities under the base
cost scenario:

- approve: PD < 0.05;
- manual review: 0.05 <= PD < 0.45;
- decline: PD >= 0.45.

On test, the policy approves 9.3553%, reviews 90.6447%, and declines 0%. This workload is not
operationally ready and should not be presented as a deployable decision policy.

## Final Test Metrics

The test partition is reserved for final evaluation, and the reported final evaluation uses
the 2015 holdout of 283,026 loans, including 42,132 bad loans (14.8863%). The nominal 95%
percentile confidence intervals use 1,000 stratified bootstrap samples and random seed 42.

| Metric | Estimate | 95% confidence interval |
| --- | ---: | --- |
| ROC AUC | 0.662707 | [0.660260, 0.665304] |
| Average precision | 0.241568 | [0.238800, 0.244326] |
| Brier score | 0.121539 | [0.121380, 0.121700] |
| Log loss | 0.399637 | Not bootstrapped |
| KS | 0.236654 | Not bootstrapped |
| Expected calibration error | 0.011108 | Not bootstrapped |

At the conventional 0.5 threshold, TN = 240,894, FN = 42,132, FP = 0, and TP = 0. The model
therefore predicts no defaults at 0.5 despite an apparently high accuracy. Discrimination is
modest, and the model must be assessed as a probability-ranking system rather than a default
0.5 classifier.

Because calibrated probabilities peak at 0.4355, no application can be classified positive at
0.5, and precision and recall there carry no information. Published threshold-dependent
metrics use the policy approve boundary of 0.05 instead: precision 0.160426, recall 0.976858,
specificity 0.105868, with TP = 41,157, FP = 215,391, FN = 975, and TN = 25,503. The
`classification_threshold_source` field in `final_test_metrics.json` records this binding. The
metrics above the table are threshold independent and unchanged.

## Cost Scenarios

The base policy assumes LGD 60%, foregone margin 5%, and USD 30 per manual review. The
implemented equation is:

```text
proxy_cost = LGD * sum(loan_amnt for approved bad loans)
           + margin * sum(loan_amnt for declined good loans)
           + review_cost * count(manual_review)
```

`manual_review` is terminal in this equation and incurs only the USD 30 fee. Reviewer
downstream approval or decline, credit loss, and foregone margin are omitted. On USD
3,624,300,800 of test exposure, the resulting partial scenario-cost proxy is USD 16,272,165,
or USD 57,493.53 per 1,000 applications. It is not a complete operating-cost estimate,
especially because 90.6447% of test rows end at `manual_review` in the proxy.

The calibration sensitivity grid contains 27 combinations of LGD 40%-80%, margin 3%-8%, and
review cost USD 15-60. Recorded partial scenario-cost proxies range from USD 29,985.60 to USD
87,387.52 per 1,000. Every recorded scenario selects the same 0.05/0.45 threshold pair and
produces no declines, so margin does not affect these scenario totals. There is no validated
comparator policy in the release artifacts; no cost-improvement claim is made.

## SHAP Interpretation

SHAP uses a reproducible 5,000-row test sample and the frozen uncalibrated LightGBM explanation
model. The five leading mean-absolute SHAP features are annual income, `fico_range_low` (the
lower bound of the reported FICO range), recent inquiries, loan amount, and DTI. Mean-absolute
importance does not establish feature direction. Local examples exist for approve and manual
review. No decline example exists because the test policy produced no declines.

Values are raw base-model log-odds contributions. They do not decompose the calibrated PD,
prove causation, establish fairness, or provide legally sufficient adverse-action reasons.

## Temporal Stability

Across the 12 monthly test slices in 2015:

| Measure | Minimum | Maximum |
| --- | ---: | ---: |
| Prevalence | 14.1445% | 15.5881% |
| ROC AUC | 0.654242 | 0.674247 |
| Average precision | 0.224612 | 0.257107 |
| Brier score | 0.116677 | 0.126097 |
| ECE | 0.002351 | 0.019466 |
| Approval rate | 7.8160% | 10.5622% |
| Partial scenario-cost proxy per 1,000 | USD 49,472.88 | USD 63,314.54 |

Train-to-test prevalence also rises from 12.5195% to 14.8863%. These retrospective slices show
variation within one historical test year; they do not demonstrate stability after 2015 or in
a live portfolio.

## Retraining and Review Triggers

No automated monitoring or retraining system is included. A governed deployment should define
its own control limits and, at minimum, trigger investigation and possible retraining when:

- the input schema, source definition, data hash, outcome definition, or decision process
  changes;
- missingness or category coverage materially departs from the training baseline;
- rolling discrimination, calibration, or cost leaves approved control limits, with the
  historical test metrics above used only as initial reference points;
- approval, review, or decline rates create an unmanageable workload or materially change;
- subgroup sample sizes, selection-rate ratios, equal-opportunity differences, or calibration
  indicate worsening reliability;
- macroeconomic, product, underwriting, geography, or applicant-population changes make the
  historical sample no longer representative;
- an independent validation, model-risk, legal, compliance, or security review requires a
  rebuild.

Retraining must repeat temporal validation, calibration, policy selection, fairness analysis,
documentation, and independent approval. It must not tune against previously disclosed test
results without establishing a new reserved holdout.

## Limitations and Ethical Considerations

- Historical accepted loans create selection bias and provide no rejected-applicant outcomes.
- Protected attributes are absent, so statutory fair-lending performance cannot be assessed.
- The release offers no causal interpretation and no legal compliance conclusion.
- Modest discrimination, zero default classifications at 0.5, and a 90.6% review rate are major
  operational limitations. Reported precision and recall use the 0.05 policy boundary, because
  0.5 lies above the calibrated probability range and yields no positive predictions at all.
- The cost result is a partial proxy: reviewed rows receive a terminal fee without modeled
  downstream decisions, credit loss, or foregone margin.
- External validity beyond LendingClub's historical 2011-2015 population is unproven.
- This is a demonstration only, not a production lending decision system.
