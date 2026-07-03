# 11 — Datasets and Licenses (Read Before You Redistribute Anything)

> **You are here:** [Learning path](index.md#learning-path) → **11 Datasets & Licenses**
> **Previous:** [10 — Next Steps](10-next-steps.md)

This repository is designed to be **publishable** on GitHub:

- It does **not** commit large raw datasets.
- It does **not** commit TabFM model weights.
- It only ships small, code-and-doc artifacts that are safe to redistribute.

However, many examples and case studies **download datasets from the web at
runtime**. This page tells you exactly what those sources are and what
licensing/terms caveats you should keep in mind.

> This is not legal advice. For any commercial or production decision, read the
> original dataset license/terms and consult counsel if needed.

## 1) This repository's license vs. TabFM's weights license

- **This educational repository (code + docs):** Apache-2.0 (see `LICENSE`).
- **TabFM source code (`tabfm` package / upstream repo):** Apache-2.0.
- **TabFM released weights (`google/tabfm-1.0.0-pytorch`):** TabFM Non-Commercial
  License v1.0 (non-commercial, non-production; see the Hugging Face model card).

The weights license is the practical blocker for most commercial deployments.
This repo does not change that.

## 2) Datasets used in `examples/`

These are the datasets used by runnable scripts under [`examples/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/examples).

### Built-in scikit-learn datasets (no network download)

- Breast Cancer Wisconsin (Diagnostic): used in `examples/01_minimal_classification.py`
  via `sklearn.datasets.load_breast_cancer`.
- California Housing: used in `examples/02_minimal_regression.py` via
  `sklearn.datasets.fetch_california_housing` (may download once depending on
  scikit-learn cache state).

### Telco churn CSV (downloaded from public mirrors)

Used in `examples/04_churn_baseline_comparison.py` and `problem1_telecom_churn`.

Mirrors (the notebook code tries multiple):

- IBM mirror:
  - `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
  - `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv`
- Community mirror:
  - `https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv`

This repo downloads the CSV and caches it locally under `data/raw/`.

## 3) Datasets used in `problems/` case studies

Each `problems/problem*/` notebook contains its own dataset download cell and
an environment variable override for the URL or dataset ID.

### `problem3_credit_card_transaction_fraud`

- Source:
  - `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`
- Override env var:
  - `FRAUD_DATA_URL=...`

### `problem4_insurance_claim_fraud`

- Source:
  - `https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv`
- Override env var:
  - `INS_DATA_URL=...`

### `problem6_ride_fare_estimation`

- Source:
  - `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv`
- Override env var:
  - `RIDE_DATA_URL=...`

### OpenML datasets (`problem5_house_price_prediction`, `problem7_loan_default_prediction`)

These notebooks use `sklearn.datasets.fetch_openml(...)` (a web download) and
cache a CSV locally after the first run.

- House Prices (Ames): `data_id=42165` (see the notebook header)
- GiveMeSomeCredit: `data_id=46929` (configurable via `OPENML_DATA_ID`)

### Kaggle dataset (`problem2_saas_subscription_churn` / KKBox)

This case study is intentionally advanced and heavy. It requires:

- Kaggle account + acceptance of competition/dataset terms.
- Kaggle API credentials configured in your environment.

This repo does not redistribute KKBox raw logs.

## 4) How this repo keeps itself publishable

- Disallowed tracked paths are enforced in CI by
  [`scripts/validate_repo_hygiene.sh`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/scripts/validate_repo_hygiene.sh).
- `.gitignore` blocks committing:
  - `data/raw/`, `data/models/`
  - `notebooks/data/raw/`, `notebooks/data/models/`
  - `problems/**/data/raw/`, `problems/**/data/models/`

If you add a new dataset or case study, follow the same pattern: download at
runtime, cache locally, and never commit raw data.
