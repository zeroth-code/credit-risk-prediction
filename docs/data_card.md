# Data Card: LendingClub Accepted Loans

## Summary

This project uses the LendingClub accepted-loans dataset to build a historical 36-month loan
default model. The data describes loans that passed LendingClub's prior selection process; it
does not represent all credit applicants.

## Source and Access

- Underlying source: LendingClub.
- Kaggle publisher and slug: `wordsforthewise/lending-club`.
- Kaggle page: [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
- Local copy receipt/retrieval date: 2026-08-02.
- Local file: `data/raw/accepted_2007_to_2018Q4.csv`.
- Local snapshot size: 1,675,133,810 bytes.
- SHA-256: `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`.

The Kaggle version ID and license metadata were not persisted with the local copy. This
repository does not redistribute the raw CSV. Users must obtain it from the Kaggle page under
the page's current license and terms; a personal Kaggle account and acceptance of those terms
may be required.

Verify the local copy with:

```bash
shasum -a 256 data/raw/accepted_2007_to_2018Q4.csv
```

The raw CSV is ignored by Git. A different size or hash is a different snapshot and requires
rerunning data preparation, training, calibration, threshold selection, and evaluation.

## Population Filters

Counts below come from `data/processed/population_audit.json`. Removed counts are exact
differences between adjacent recorded stages, not estimates.

| Audit stage | Remaining rows | Removed at stage |
| --- | ---: | ---: |
| Raw accepted-loan rows | 2,260,701 | - |
| Valid nonblank IDs | 2,260,701 | 0 |
| IDs unique under the pipeline rule | 2,260,701 | 0 |
| Valid issue dates | 2,260,668 | 33 |
| 36-month term | 1,609,754 | 650,914 |
| Explicit final outcome labels | 1,020,768 | 588,986 |

The pipeline strips ID and term strings, drops every duplicated ID if a duplicate exists,
parses `issue_d`, keeps only `36 months`, and retains only the explicit good and bad outcomes
defined below.

## Outcome Mapping

The binary target is `bad`:

- `Fully Paid` maps to `bad = 0`;
- `Charged Off` and `Default` map to `bad = 1`;
- every other status is excluded from the labeled population.

Configured unresolved statuses include `Current`, `In Grace Period`, `Issued`,
`Late (16-30 days)`, and `Late (31-120 days)`. The exclusion rule is broader than that list:
any status outside the three explicit outcome labels is not used for supervised learning.

The label is a historical final-status outcome, not a fixed delinquency measure observed at a
uniform number of months after origination. Restricting the sample to 36-month final-status
loans makes the contractual term consistent, but it does not remove all differences in when or
how final status was recorded.

## Temporal Partitions

The 1,020,768 eligible final-status rows are assigned by issue date. Exactly 603,589 rows fall
inside the modeling windows, 417,179 are outside the 2011-2015 horizon, and 0 fall into gaps
between configured windows.

| Partition | Configured issue-date window | Rows |
| --- | --- | ---: |
| Train | 2011-01-01 to 2013-12-31 | 157,993 |
| Validation | 2014-01-01 to 2014-06-30 | 71,955 |
| Calibration | 2014-07-01 to 2014-12-31 | 90,615 |
| Test | 2015-01-01 to 2015-12-31 | 283,026 |
| Assigned total | 2011-01-01 to 2015-12-31 | 603,589 |
| Outside horizon | Before 2011 or after 2015 | 417,179 |

In the reported workflow, train fits preprocessing and models; validation supports model,
feature-set, imbalance, and hyperparameter selection; calibration supports calibration-method
evaluation/refit and policy threshold selection; and test is reserved for final evaluation and
post-selection diagnostics.

## Exclusions

The modeling data excludes:

- 33 rows with invalid issue dates;
- 650,914 rows removed by the 36-month term filter after valid-date filtering;
- 588,986 rows without one of the three explicit final outcome labels after term filtering;
- 417,179 otherwise eligible rows outside the configured modeling horizon;
- all post-origination payment, balance, recovery, and collection fields from model features.

No rows were removed for blank IDs or duplicate IDs in this snapshot. These counts describe
this exact hash only.

## Feature Availability

The release uses the challenger feature set, selected to represent information available at
application or underwriting time.

| Type | Features |
| --- | --- |
| Numeric | `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`, `fico_range_low`, `fico_range_high`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` |
| Categorical | `purpose`, `home_ownership`, `verification_status`, `emp_length`, `addr_state` |

The challenger excludes lender-decision variables `int_rate`, `grade`, and `sub_grade`, even
though they are evaluated in a separate full-underwriting experiment. It also excludes fields
that become known only after origination, including payments, recovered amounts, outstanding
principal, last/next payment data, and later credit pulls.

Application-time availability is based on the source field semantics and must be revalidated
against any future production origination process.

## Missingness

The following counts were measured across the 603,589 assigned train, validation, calibration,
and test rows for the 17 challenger features:

| Feature | Missing rows | Missing rate |
| --- | ---: | ---: |
| `emp_length` | 35,885 | 5.9453% |
| `revol_util` | 307 | 0.05086% |
| `dti` | 2 | 0.00033% |
| Other 14 challenger features | 0 | 0% |

Numeric values are normalized, with percent suffixes accepted where applicable, and median
imputed from training data. Categorical values are most-frequent imputed, one-hot encoded with
unknown categories ignored, and infrequent categories grouped using a minimum frequency of
25. Imputation hides missingness from the estimator unless missingness is separately
monitored, so missing-rate drift remains an operational concern.

## Known Selection Effects

- The file contains accepted loans, not the complete applicant pool. Historical underwriting
  determines who appears in the data, creating selection bias and preventing reject inference.
- Filtering to loans with observed final statuses can introduce outcome-maturity and
  survivorship effects.
- LendingClub's customer mix, product terms, policies, macroeconomic conditions, and reporting
  practices may differ from another lender or a current portfolio.
- The 2011-2015 modeling window is a historical slice and cannot establish contemporary
  performance.
- Income, employment, housing, and geography may reflect structural differences. Their
  presence does not make them causal or legally appropriate decision factors in another
  context.

## Protected Attributes

The modeling snapshot does not include the protected attributes needed for a statutory
fair-lending analysis. Race, ethnicity, sex/gender, age, and other legally protected classes
cannot be evaluated directly, and intersectional performance cannot be measured. Available
income, home-ownership, employment, and regional groupings are operational proxy reliability
diagnostics only; they are not substitutes for protected-attribute analysis.

## Responsible Use

Use this data card to reproduce and audit the portfolio demonstration. Do not use this
historical selected sample as the sole basis for a real lending policy, legal conclusion, or
production model approval.
