# Google TabFM Implementation

Production-oriented repository for benchmarking and applied workflows with Google TabFM across multiple real tabular problems (classification and regression), with reproducible notebooks, policy outputs, and artifact persistence.

## Project Overview
This repository combines two tracks:

1. **Reusable benchmark package** (`src/tabfm_benchmark`) for multi-dataset TabFM benchmarking.
2. **Applied problem notebooks** (`problems/problem1...problem8`) that implement end-to-end ML workflows, compare TabFM vs XGBoost baselines, and export decision-oriented business artifacts.

The project uses `uv` for dependency/environment management and emphasizes reproducibility via executed notebooks and persisted metrics.

## Problem Statement
Tabular ML teams need a practical way to validate foundation models such as TabFM against strong baselines under realistic constraints:

- mixed task types (churn, fraud, pricing, attrition, loan risk),
- limited GPU memory,
- reproducible train/validation/test separation,
- decision policies beyond raw model scores.

## Objectives
- Provide a reproducible TabFM benchmark scaffold in Python.
- Execute production-style workflows across 8 business-like tabular problems.
- Compare TabFM variants (`default`, `ensemble`, `advanced`) with XGBoost baselines.
- Persist metrics, predictions, threshold curves, and policy/action artifacts.
- Document known operational constraints (memory, checkpoint availability, environment/runtime).

## Architecture / Approach
### Core Components
- **Benchmark package**: [`src/tabfm_benchmark`](src/tabfm_benchmark)
  - dataset specs and loaders,
  - metric computation,
  - unified result schema,
  - benchmark CLI entry point.
- **Strict E2E runner**: [`scripts/run_strict_e2e.py`](scripts/run_strict_e2e.py)
  - notebook execution orchestration,
  - context backoff and OOM-profile retries,
  - artifact validation and report generation.
- **Applied notebooks**: [`problems/`](problems)
  - problem-specific feature engineering,
  - TabFM/XGBoost training,
  - business policy optimization outputs.

### Runtime Pattern
- Seed control + leakage-safe splitting.
- TabFM checkpoint resolution (local override / HF cache / one-time download path in notebook logic).
- Device routing (`auto|cpu|cuda`) with fallback for low-memory GPUs.
- Persisted artifacts per problem under `problems/<problem>/artifacts/`.

## Implementation Process
1. Build benchmark module and tests.
2. Implement problem notebooks with reproducible setup and exported artifacts.
3. Add strict E2E automation for notebook execution and retry behavior.
4. Execute notebooks and persist outputs (`*.executed.ipynb`, metrics CSVs, runtime metadata JSONs, policy summaries).
5. Validate package/tests and document operational behavior.

## Setup and Installation
### Prerequisites
- Linux
- Python 3.11+ (project requires `>=3.11`)
- `uv`

### Install
```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

### Environment Notes
In restricted environments, set:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

This avoids read-only cache path issues for `uv` operations.

## Usage
### 1. Run Tests
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

### 2. Benchmark CLI
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_benchmark.py \
  --device cuda \
  --seed 42 \
  --n-estimators 32 \
  --output artifacts/tabfm_benchmark_results.parquet \
  --summary-output artifacts/tabfm_benchmark_summary.csv
```

### 3. Strict E2E Notebook Runner
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_strict_e2e.py --only problem1_telecom_churn
```

### 4. Notebook Execution (single notebook)
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run jupyter nbconvert \
  --to notebook \
  --execute problems/problem1_telecom_churn/problem1_telecom_churn_tabfm.ipynb \
  --output problem1_telecom_churn_tabfm.executed.ipynb
```

## CI and Release Automation
### CI gates
- GitHub Actions workflow: `.github/workflows/ci.yml`
- Triggers:
  - pull requests targeting `main`
  - pushes to `main`
- Enforced checks:
  - repository hygiene (`./scripts/validate_repo_hygiene.sh`)
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`
  - benchmark CLI contract (`run_benchmark.py --help`)
  - strict E2E CLI contract (`run_strict_e2e.py --help`)

### Release flow (tag-driven)
- GitHub Actions workflow: `.github/workflows/release.yml`
- Trigger: pushing semantic tags matching `v*.*.*`
- Release behavior:
  - reruns deterministic validation checks
  - creates/updates GitHub release for the tag
  - uses `RELEASE_NOTES.md` as release body
  - attaches `HANDBOOK.pdf` to release assets

### Maintainer release steps
```bash
git tag -a v1.0.1 -m "v1.0.1"
git push origin v1.0.1
```

## Experiments / Workflow
- **Problems covered**: 8 (`problem1` to `problem8`), including churn, fraud, regression/pricing, loan default, and attrition.
- **Model families**: TabFM variants + XGBoost baseline.
- **Artifacts produced**:
  - model metrics (`*_model_metrics.csv`),
  - predictions (`*_predictions_test.parquet`),
  - runtime metadata (`*runtime_meta.json`),
  - threshold/policy outputs (`*_threshold_curve.csv`, `*_policy_summary.csv`, etc.).

## Outputs / Results
Below are test-split champion metrics from persisted artifacts in `problems/*/artifacts/`:

| Problem | Champion | Test Metrics (from CSV/JSON artifacts) |
|---|---|---|
| `problem1_telecom_churn` | `tabfm_ensemble_preset` | ROC-AUC `0.8468`, PR-AUC `0.6577`, Accuracy `0.8020`, F1 `0.5854` |
| `problem2_saas_subscription_churn` | `tabfm_default` | ROC-AUC `0.9802`, PR-AUC `0.8115`, Accuracy `0.9545`, F1 `0.7191` |
| `problem3_credit_card_transaction_fraud` | `xgboost_baseline` | ROC-AUC `0.7039`, PR-AUC `0.4053`, Accuracy `0.9970`, F1 `0.5714` |
| `problem4_insurance_claim_fraud` | `tabfm_ensemble_preset` | ROC-AUC `0.7917`, PR-AUC `0.5796`, Accuracy `0.8000`, F1 `0.6341` |
| `problem5_house_price_prediction` | `tabfm_default` | RMSE `24008.32`, MAE `15397.57`, R² `0.8793`, MAPE `9.52%` |
| `problem6_ride_fare_estimation` | `tabfm_ensemble_preset` | RMSE `4.1677`, MAE `2.2068`, R² `0.8343`, MAPE `18.53%` |
| `problem7_loan_default_prediction` | `tabfm_ensemble_preset` | ROC-AUC `0.8318`, PR-AUC `0.3887`, Accuracy `0.9450`, F1 `0.3125` |
| `problem8_employee_attrition_prediction` | `tabfm_ensemble_preset` | ROC-AUC `0.8449`, PR-AUC `0.6399`, Accuracy `0.8824`, F1 `0.5667` |

Additional policy artifacts are available per problem (threshold summaries, top-k campaigns, action policies).

## Limitations
- **License constraint**: TabFM released model weights are under non-commercial terms (see references below).
- **Large local assets**: raw datasets and checkpoints are very large; they are local-only and excluded from git tracking for publishability.
- **Environment sensitivity**:
  - sandboxed notebook execution can fail with socket permission errors,
  - missing checkpoint paths can break notebook execution,
  - low GPU memory can trigger CPU fallback and context-capped training.
- **Runtime variability**: strict E2E duration varies significantly by notebook/problem and environment.

## Future Improvements
- Extend CI with optional notebook smoke checks under strict memory/time caps.
- Add dataset fetch scripts with checksums and resumable downloads.
- Add automated model card/report generation from artifact bundles.
- Introduce optional Git LFS profile for teams that need data+weights versioning.
- Add benchmark trend tracking (MLflow/W&B) across runs.

## References / Sources
Only official or directly used resources from repository context are listed.

### TabFM and Model Resources
- Google Research TabFM repository: https://github.com/google-research/tabfm
- TabFM model weights (Hugging Face): https://huggingface.co/google/tabfm-1.0.0-pytorch
- Google Research announcement: https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/

### Dataset Sources Used in Notebooks
- TensorFlow-hosted credit-card dataset CSV: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- IBM attrition dataset CSV: https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv
- Insurance claims dataset CSV: https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv
- Seaborn taxis dataset CSV: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv
- Telco churn CSV mirrors used by notebook logic:
  - https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv
  - https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv
  - https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

### Framework / Library Documentation
- PyTorch docs: https://pytorch.org/docs/stable/
- scikit-learn docs: https://scikit-learn.org/stable/
- XGBoost docs: https://xgboost.readthedocs.io/
- Polars docs: https://docs.pola.rs/
- pandas docs: https://pandas.pydata.org/docs/
- Jupyter docs: https://jupyter.org/documentation
- Kaggle API docs: https://www.kaggle.com/docs/api
- OpenML docs: https://docs.openml.org/

## Important License Note
This project code is independent, but TabFM released checkpoints are non-commercial. Review upstream license terms before any commercial deployment.
