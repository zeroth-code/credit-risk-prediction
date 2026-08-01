# LendingClub Data

## Source

Use the Kaggle LendingClub accepted loans dataset, covering 2007 through 2018 Q4. The
download requires a personal Kaggle account and acceptance of the dataset terms.

## Expected File

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`. The raw file must not be committed to
Git.

## Verification

After downloading the file, run `scripts/hash_file.py` and record its SHA-256 in the
published data card. Any hash change requires rerunning the complete data and modeling
workflow. The verification script is added in a later task.

## Modeling Population

The modeling population covers 2011-2015 final-status loans with a 36-month term. Treat
`Fully Paid` as good and `Charged Off` or `Default` as bad. Exclude unresolved statuses and
report their counts.

## Limitations

This is historical data shaped by LendingClub's underwriting selection, not a representative
sample of all borrowers. It lacks legal protected-attribute coverage and cannot establish
current production performance or fair-lending compliance.
