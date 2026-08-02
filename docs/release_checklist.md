# Release Checklist

Verification evidence for release `0.1.0`. Every entry below was produced by running the
pipeline and the test suite locally against the committed `artifacts/release/` bundle. Items
that are not met are marked `NOT MET` rather than omitted.

Raw data snapshot SHA-256:
`3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`
(`accepted_2007_to_2018Q4.csv`, 1,675,133,810 bytes, not redistributed).

## Local verification run

| Command | Result |
| --- | --- |
| `uv sync --locked --dev` | environment resolves from the committed lock file |
| `uv run ruff check .` | `All checks passed!` |
| `uv run pytest --cov=credit_risk --cov-report=term-missing` | 703 passed, 90% statement coverage |
| `uv run python scripts/prepare_data.py` | four temporal Parquet partitions plus `population_audit.json` |
| `uv run python scripts/train.py` | preprocessor, uncalibrated and calibrated models, tuning trials, policy |
| `uv run python scripts/evaluate.py` | final test metrics, SHAP, fairness, and release manifest |
| `docker build -t credit-risk-prediction:local .` | image builds from the locked runtime dependencies |
| `docker run --rm -p 8501:8501 credit-risk-prediction:local` | Streamlit serves; `/_stcore/health` returns `ok` |

A complete re-run of all three pipeline stages reproduced every file in `artifacts/release/`
byte-for-byte (`git diff artifacts/release/` was empty afterwards), so the release is
reproducible from the recorded seeds and the pinned data snapshot.

## No prohibited fields in the model matrix

- MET. `configs/features.yaml` lists the prohibited `post_origination` fields, and
  `build_feature_frame` raises on any intersection with the selected columns.
- The 90 encoded features in `shap_importance.csv` were checked against the prohibited list
  and returned zero matches.
- `tests/test_features.py` fails the build if a prohibited field enters a training matrix.

## Test set untouched until `scripts/evaluate.py`

- MET. `scripts/train.py` contains no read of `test.parquet`; it loads only the train,
  validation, and calibration partitions.
- Hyperparameter search uses the 2014 H1 validation partition; calibration-method selection and
  both policy thresholds use the 2014 H2 calibration partition.
- `scripts/evaluate.py` is the only entry point that opens `data/processed/test.parquet`.

## Dummy, Logistic, and LightGBM results

MET. Nine validation experiments are recorded in `validation_metrics.json`
(average precision on the validation partition, prevalence 0.1322):

| Model | Imbalance | Feature set | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- |
| Dummy (prior) | natural | challenger | 0.1322 | 0.5000 |
| Logistic Regression | natural | challenger | 0.2060 | 0.6453 |
| Logistic Regression | weighted | challenger | 0.2050 | 0.6448 |
| LightGBM | natural | challenger | 0.2158 | 0.6577 |
| LightGBM | weighted | challenger | 0.2160 | 0.6571 |
| LightGBM | undersampled | challenger | 0.2089 | 0.6515 |
| Logistic Regression | natural | full_underwriting | 0.2202 | 0.6651 |
| Logistic Regression | weighted | full_underwriting | 0.2193 | 0.6649 |
| LightGBM | natural | full_underwriting | 0.2266 | 0.6718 |

The weighted LightGBM challenger was selected and tuned with 30 Optuna trials under seed 42.
`full_underwriting` is reported only as a secondary benchmark, as it consumes LendingClub's own
grade, sub-grade, and pricing outputs.

## Calibration selected on the calibration partition

MET. Selection used stratified five-fold out-of-fold probabilities on the 2014 H2 partition
(90,615 loans, prevalence 0.1413), scored by Brier score with log loss and ECE reported:

| Method | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| Uncalibrated | 0.23195 | 0.65229 | 0.32980 |
| Sigmoid (selected) | 0.11648 | 0.38662 | 0.00045 |
| Isotonic | 0.11651 | 0.38713 | 0.00040 |

Sigmoid won on Brier score and log loss; isotonic was marginally better on ECE. The released
calibrator is a full refit on the calibration partition. The application displays calibrated
probability, never an uncalibrated score.

## PR-AUC, ROC-AUC, Brier, log loss, KS, and confidence intervals

MET, on the untouched 2015 out-of-time test partition (283,026 loans, prevalence 0.148863).
Intervals are 1,000-sample stratified percentile bootstraps under seed 42 and include the point
estimate:

| Metric | Estimate | 95% CI |
| --- | --- | --- |
| ROC-AUC | 0.66271 | 0.66026 – 0.66530 |
| PR-AUC | 0.24157 | 0.23880 – 0.24433 |
| Brier score | 0.12154 | 0.12138 – 0.12170 |

Log loss 0.39964, KS 0.23665, and ECE 0.01111 are also recorded. `temporal_metrics.csv` holds
the monthly test slices. At the conventional 0.5 threshold the model predicts no defaults
(TP 0, FP 0, FN 42,132, TN 240,894), which the README keeps prominent as evidence that accuracy
is the wrong headline metric.

## Cost per 1,000 applications and 27 sensitivity scenarios

MET. `cost_sensitivity.csv` contains all 27 combinations of LGD {0.40, 0.60, 0.80}, margin
{0.03, 0.05, 0.08}, and review cost {15, 30, 60} USD, reporting both the re-optimized policy
and the frozen base policy for each scenario.

Thresholds were selected on the calibration partition: approve below 0.05, decline at or above
0.45. On the base scenario the frozen thresholds equal the re-optimized ones
(51,832.48 USD per 1,000 applications on the calibration partition).

Frozen policy on the test partition: 57,493.53 USD per 1,000 applications, approval rate
9.36%, review rate 90.64%, decline rate 0.00%, total exposure 3,624,300,800 USD.

These cost assumptions are illustrative and are not estimates of LendingClub economics. Manual
review is modeled as a terminal fee only, so the figure is a partial cost proxy rather than a
complete operating cost.

## SHAP global and local explanations

PARTIALLY MET. Global artifacts are complete: `shap_importance.csv` (90 features), a beeswarm
plot, and five dependence plots for the most influential continuous variables
(`annual_inc`, `fico_range_low`, `inq_last_6mths`, `loan_amnt`, `dti`).

Local waterfall plots exist for `approve` and `manual_review` only. NOT MET: no high-risk
`decline` waterfall is published, because the frozen policy declines no test applications, so no
genuine declined example exists. The pipeline records the absent action rather than substituting
a fabricated example. The application labels all SHAP output as association, not causation.

## Four subgroup reports and the compliance limitation

MET. `fairness_summary.json` plus four CSVs cover income quintiles (5 groups), home ownership
(4 groups, 1 suppressed), US census region (4 groups), and employment length (12 groups).
Groups below the configured minimum of 200 observations are suppressed rather than reported.
Equal Opportunity Difference and Selection Rate Ratio are reported per attribute.

`docs/fairness_report.md`, the README, and the application state that LendingClub data lacks
race, ethnicity, sex, age, and other protected attributes, so this is a subgroup reliability
diagnostic and not a statutory fair-lending audit under ECOA, Regulation B, or the FHA.

## Docker and Streamlit smoke tests

MET. The image builds and serves. `libgomp1` is installed because LightGBM links against the
OpenMP runtime that `python:3.12-slim` omits; without it the app raised `StartupError` while the
Streamlit health endpoint still returned 200, so the health check alone is not sufficient
evidence. Verified inside the running container: the manifest loads, all hashes match, and the
synthetic application scores 19.26% and routes to manual review. `tests/test_app.py` covers
import and startup in CI.

## Documentation

MET: `README.md` (all 18 required sections), `docs/data_card.md`, `docs/model_card.md`,
`docs/fairness_report.md`, `docs/case_study.md`, and this checklist. The README screenshot was
captured from the container running the committed release bundle.

NOT MET: the demonstration video and the three resume bullets are not in the repository, and no
public deployment URL exists yet. The README states the deployment URL is pending rather than
inventing one. These remain open before the portfolio is presented externally.
