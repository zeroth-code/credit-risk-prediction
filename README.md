# Credit Risk Prediction

An end-to-end portfolio project for estimating 36-month consumer-loan default risk from
application-time information. The frozen release combines a `ColumnTransformer`, a weighted
LightGBM challenger, sigmoid probability calibration, an approve/review/decline cost policy,
SHAP diagnostics, and subgroup reliability checks.

This is a demonstration only, not a production lending decision system. The final model has
modest discrimination, and its frozen policy sends 90.6% of the test population to manual
review while declining no applications.

## Live Demo and Screenshot

The public deployment URL will be added after deployment. No URL is invented here.

> Screenshot pending: a current image must be captured from the genuine
> `artifacts/release/` bundle. No old fixture or `test-1` screenshot is used.

The local Streamlit application verifies the release manifest before loading artifacts and
labels the experience as a demonstration rather than a lending decision tool.

## Business Problem

For a lender, a default probability is useful only when it is estimated from information
available at decision time, holds up on later borrowers, and can support an explicit operating
policy. The project therefore treats credit risk as four connected problems:

- rank applicants by default risk;
- produce probabilities that correspond to observed default frequencies;
- translate probabilities into approve, manual-review, and decline actions under stated costs;
- expose model associations and subgroup reliability limits for human review.

The target is `bad = 1` for a 36-month loan that ultimately reached `Charged Off` or `Default`.
This is historical risk estimation, not an automated eligibility determination.

## Why Accuracy Is Not Enough

The out-of-time test set contains 42,132 bad loans among 283,026 loans, a prevalence of
14.8863%. Predicting every loan as good would therefore appear to be 85.1137% accurate while
detecting no defaults. That is exactly what the frozen calibrated model does at the
conventional 0.5 classification threshold: TN 240,894, FN 42,132, FP 0, and TP 0.

The project emphasizes ROC AUC, average precision, Brier score, log loss, calibration error,
the Kolmogorov-Smirnov statistic, temporal slices, and scenario costs. The 0.5 result is kept
prominent because it demonstrates why a probability model and an operating policy must not be
judged by accuracy alone.

## Dataset and Outcome Window

The source is the Kaggle LendingClub accepted-loans CSV covering 2007 through 2018 Q4. Access
requires a personal Kaggle account and acceptance of the dataset terms. The verified raw file
is 1,675,133,810 bytes with SHA-256
`3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`.

The population pipeline starts with 2,260,701 accepted-loan rows. After date, 36-month term,
and final-outcome filters, 1,020,768 rows are eligible. Of these, 603,589 fall into the defined
2011-2015 modeling windows and 417,179 are outside the horizon; there are no window-gap rows.
See [the data card](docs/data_card.md) for the complete measured audit, missingness, outcome
mapping, and selection effects.

## Leakage Prevention

The release uses the 17-feature challenger set: 12 numeric and 5 categorical variables that
are available at application or underwriting time. It excludes repayment and collection
fields such as total payments, recovered amounts, outstanding principal, last-payment data,
next-payment dates, and later credit pulls. It also excludes `int_rate`, `grade`, and
`sub_grade` from the release challenger because those fields encode lender underwriting
decisions.

Temporal ownership is enforced:

- train fits preprocessors and candidate models;
- validation compares feature sets, models, imbalance strategies, and 30 Optuna trials;
- calibration compares calibration methods, fits the selected calibrator, and selects policy
  thresholds;
- test is reserved for final metrics, policy evaluation, SHAP sampling, and subgroup
  diagnostics.

No test result participates in model, calibration-method, or threshold selection.

## Temporal Validation Design

| Partition | Issue dates | Rows | Ownership |
| --- | --- | ---: | --- |
| Train | 2011-01-01 to 2013-12-31 | 157,993 | Preprocessing and model fitting |
| Validation | 2014-01-01 to 2014-06-30 | 71,955 | Model, feature-set, imbalance, and tuning selection |
| Calibration | 2014-07-01 to 2014-12-31 | 90,615 | Calibration evaluation/refit and policy thresholds |
| Test | 2015-01-01 to 2015-12-31 | 283,026 | One-way final evaluation and diagnostics |

The partitions follow issue date rather than a random split. Test prevalence rises from
12.5195% in train to 14.8863% in test, making temporal drift part of the evaluation rather
than something hidden by shuffled folds.

## Models and Imbalance Strategies

Experiments compare a prior dummy classifier, logistic regression, and LightGBM. Logistic
baselines use natural and balanced class weights on challenger and full-underwriting feature
sets. LightGBM experiments compare natural, positive-class-weighted, and randomly
undersampled challenger training, plus a natural full-underwriting diagnostic.

The weighted challenger LightGBM is selected by validation average precision among the
challenger LightGBM strategies. Its positive-class weight is 6.9875. A 30-trial Optuna study
selects 894 estimators, learning rate 0.0107234, 33 leaves, and the remaining regularization
parameters recorded in `artifacts/release/validation_metrics.json`.

## Probability Calibration

Uncalibrated, sigmoid, and isotonic methods are compared on the calibration partition.
Sigmoid and isotonic evaluation probabilities come from 5-fold stratified out-of-fold
predictions; the selected method is then refit on the full calibration partition. Sigmoid is
selected with calibration Brier score 0.116482, log loss 0.386624, and expected calibration
error 0.000448.

The frozen scoring artifact is a `CalibratedClassifierCV` around the selected LightGBM model.
Calibration improves probability interpretation on its selection partition, but the higher
test Brier score and ECE show that calibration still changes over time.

## Business Cost Policy

The base scenario assumes 60% loss given default, 5% foregone margin on a declined good loan,
and USD 30 per manual review. Thresholds are selected on calibration out-of-fold probabilities:

- approve when PD < 0.05;
- manual review when 0.05 <= PD < 0.45;
- decline when PD >= 0.45.

On the test set, this policy approves 9.3553%, sends 90.6447% to manual review, and declines
0%. Across USD 3,624,300,800 of test exposure, the stated base scenario costs USD 16,272,165,
or USD 57,493.53 per 1,000 applications.

The 27 calibration sensitivity scenarios span LGD 40%-80%, margin 3%-8%, and manual-review
cost USD 15-60. Scenario cost ranges from USD 29,985.60 to USD 87,387.52 per 1,000. No valid
comparator policy is present in the release artifacts, so this project does not claim an
unsupported cost improvement.

## Results

Final results are measured once on the 2015 out-of-time test partition. Confidence intervals
use 1,000 stratified bootstrap samples, percentile intervals, a 95% confidence level, and
random seed 42.

| Metric | Test estimate | 95% CI |
| --- | ---: | --- |
| ROC AUC | 0.662707 | [0.660260, 0.665304] |
| Average precision | 0.241568 | [0.238800, 0.244326] |
| Brier score | 0.121539 | [0.121380, 0.121700] |
| Log loss | 0.399637 | Not bootstrapped |
| KS | 0.236654 | Not bootstrapped |
| Expected calibration error | 0.011108 | Not bootstrapped |

Discrimination is modest. Monthly 2015 ROC AUC ranges from 0.6542 to 0.6742 and monthly ECE
from 0.00235 to 0.01947; these are retrospective stability diagnostics, not evidence of live
production robustness.

## SHAP Explainability

Global SHAP evidence uses a reproducible 5,000-row test sample. The leading associations are
annual income, lower FICO score, recent inquiries, loan amount, and debt-to-income ratio.
Local examples are available for approve and manual-review actions. A decline example is not
shown because the frozen test policy produced no declines.

The SHAP values explain the uncalibrated base LightGBM output in raw log-odds units. They are
directional statistical associations, not causal effects, adverse-action reasons, or an
additive decomposition of the calibrated probability.

## Fairness and Subgroup Reliability

The release reports proxy reliability diagnostics for income bands, home ownership,
employment length, and US Census region, with a minimum group size of 200. Selection-rate
ratios are 0.037063, 0.316112, 0.525137, and 0.901025 respectively; equal-opportunity
differences are 0.230278, 0.103537, 0.051544, and 0.010746. One one-row home-ownership group is
suppressed.

Protected attributes are absent, so the analysis cannot assess outcomes by race, ethnicity,
sex, age, or other legally protected classes. These are policy-sensitive proxy reliability
checks only. This is not a statutory fair-lending audit. See the
[fairness report](docs/fairness_report.md) for definitions and interpretation limits.

## Repository Structure

```text
.
|-- app/                    # Read-only Streamlit demonstration
|-- configs/                # Data windows, costs, and feature dictionaries
|-- data/                   # Raw/processed locations; generated data is ignored
|-- docs/                   # Data, model, fairness, and portfolio documentation
|-- scripts/                # Prepare, train, and final-evaluation entry points
|-- src/credit_risk/        # Reusable data, modeling, policy, and diagnostic code
|-- tests/                  # Unit, integration, artifact, and app tests
|-- artifacts/release/      # Genuine local frozen release bundle
`-- reports/figures/        # Generated SHAP figures
```

The genuine `artifacts/release/` directory is preserved locally and is not added by this
documentation change.

## Reproduce Locally

Python 3.12 and access to the Kaggle source CSV are required. Place
`accepted_2007_to_2018Q4.csv` in `data/raw/`, verify its hash against the data card, then run:

```bash
uv sync --dev
uv run python scripts/prepare_data.py
uv run python scripts/train.py
uv run python scripts/evaluate.py
uv run streamlit run app/streamlit_app.py
```

Training runs the real 30-trial Optuna study and may take substantial time and memory. The
scripts regenerate ignored processed data and model artifacts; they do not download the raw
dataset.

## Run Tests

```bash
uv run pytest
```

Static checks use `uv run ruff check .`.

## Docker

Docker packaging is pending Task 17. This commit does not contain a `Dockerfile`, so the
following planned commands are not runnable yet. After Task 17 adds the packaging, the
intended interface is:

```bash
docker build -t credit-risk-prediction .
docker run --rm -p 8501:8501 credit-risk-prediction
```

When packaging exists, the runtime must contain the frozen release bundle expected by the
application. Do not load untrusted joblib files; Python pickle semantics can execute code
during deserialization.

## Limitations and Responsible Use

- The model has modest discrimination and is not suitable for autonomous credit decisions.
- The conventional 0.5 classifier predicts no defaults on test data.
- The cost policy sends 90.6% to manual review and declines none, so it is not operationally
  ready without capacity analysis and policy redesign.
- LendingClub accepted loans are a historically selected sample, not the full applicant
  population; rejected-applicant risk and reject inference are unavailable.
- Protected attributes are absent. The project provides no causal interpretation, no
  statutory fair-lending audit, and no legal compliance conclusion.
- Historical results do not establish current or production performance. Deployment would
  require independent validation, governance, monitoring, human oversight, security review,
  accessibility review, and documented adverse-action processes.
- This repository is a portfolio demonstration only, not a production lending decision
  system.

Additional evidence is documented in the [data card](docs/data_card.md),
[model card](docs/model_card.md), [fairness report](docs/fairness_report.md), and
[case study](docs/case_study.md).
