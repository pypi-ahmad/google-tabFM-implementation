# Google TabFM — Zero to Mastery

A complete, beginner-friendly path to learning **[Google Research's TabFM](https://github.com/google-research/tabfm)**
(Tabular Foundation Model) — from "what even is this?" to running eight
production-style workflows on real business datasets.

> **TL;DR:** TabFM is a pre-trained model that predicts on spreadsheet-like
> data (classification and regression) **without you training anything** —
> it reads your training rows as context at prediction time instead of
> fitting to them. This repo teaches you what that means, how to use it
> correctly, and when it's the right tool versus a conventional model like
> XGBoost.

## Who this is for

- **Complete beginners** to machine learning who can read basic Python and
  want a real, working introduction to foundation models for tabular data.
- **ML practitioners** who know scikit-learn/XGBoost and want a fast,
  accurate on-ramp to TabFM specifically, with the jargon defined and the
  gotchas already found for you.
- **Anyone evaluating TabFM** for a real project, who wants an honest
  picture of its capabilities, limits, and license terms before committing.

You do **not** need prior deep learning experience — [docs/01-prerequisites.md](docs/01-prerequisites.md)
tells you exactly what you need and links out for anything you're missing.

## What you'll be able to do when you're done

By the end of the [learning path](#learning-path) below, you will be able
to: explain what TabFM is and how it differs from a conventionally trained
model; install it and make your first prediction; choose between its
default and ensemble presets with an actual understanding of the tradeoff;
prepare a real dataset correctly; evaluate results honestly against a
baseline; debug the failures you're most likely to hit; and know exactly
what TabFM's license does and doesn't allow.

## Why TabFM matters

Fitting a model to a new tabular dataset traditionally means cleaning data,
picking a model family, and tuning hyperparameters — often the most
time-consuming part of a project, repeated for every new dataset. TabFM
applies the same **in-context learning** idea that makes large language
models useful with a few prompt examples, to tables: it treats your
training rows as context and predicts on new rows in one forward pass, no
training loop required. It won't always beat a well-tuned conventional
model (this repo shows you exactly when it does and doesn't, using real
runs — see [docs/07-evaluation.md](docs/07-evaluation.md)), but it changes
how fast you can get a credible first result. Full explanation, with
citations: [docs/00-overview.md](docs/00-overview.md).

## Quickstart

```bash
git clone https://github.com/pypi-ahmad/google-tabFM-implementation.git
cd google-tabFM-implementation

uv venv --python 3.11
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/uv-cache
uv sync --extra dev --frozen

# One-time fix for a known upstream mismatch in `tabfm==1.0.0` (PyTorch):
# the loader expects `pytorch_model.bin` but the HF repo ships `model.safetensors`.
# This downloads and converts the weights into the expected format.
uv run python scripts/fetch_tabfm_weights.py --task classification
export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch

uv run python examples/01_minimal_classification.py
```

**Expected output** (verified in this repo's own reference environment — a
laptop GPU with under 12 GiB VRAM, so it auto-selects CPU; see
[docs/08-troubleshooting.md](docs/08-troubleshooting.md#cuda-out-of-memory)):
```
Detected GPU with 7.6 GiB VRAM (<12 GiB) — using CPU for a stable run. See docs/08-troubleshooting.md#cuda-out-of-memory.
Loading TabFM classification weights (device=cpu) ...
Test accuracy: 0.9737
Test set size: 114 rows | Train context size: 455 rows
```

If anything about this didn't work, or you want to understand *why* it
worked, don't guess — follow the [learning path](#learning-path) from the
top. It exists precisely so you don't have to reverse-engineer this repo
from a quickstart snippet.

> **Before your first real run:** TabFM's pretrained weights are a
> multi-gigabyte download from Hugging Face, fetched automatically the
> first time you load the model — see
> [docs/02-installation.md §3](docs/02-installation.md#3-disk-and-network-requirements-read-this-before-your-first-run)
> for exact sizes and how to plan around it.

## Learning path

Work through these in order the first time. Each page is short, defines
its own jargon, and links forward/backward.

| # | Doc | You'll learn |
|---|---|---|
| 00 | [Overview](docs/00-overview.md) | What TabFM is, the problem it solves, what it is *not* |
| 01 | [Prerequisites](docs/01-prerequisites.md) | Knowledge and environment you need before starting |
| 02 | [Installation](docs/02-installation.md) | Install this repo and TabFM; disk/network requirements |
| 03 | [First Run](docs/03-first-run.md) | Your first prediction, line by line, with verified expected output |
| 04 | [Core Concepts](docs/04-core-concepts.md) | In-context learning, zero-shot, architecture, ensemble preset, full glossary |
| 05 | [Working with Data](docs/05-working-with-data.md) | Input contract, documented limits, dataset-prep checklist |
| 06 | [Usage Workflows](docs/06-training-or-usage-workflows.md) | Classification & regression workflows; this repo's conventions vs. the library API |
| 07 | [Evaluation](docs/07-evaluation.md) | Metrics defined, baseline comparison, this repo's real results, official benchmark context |
| 08 | [Troubleshooting](docs/08-troubleshooting.md) | Every failure mode this repo has actually reproduced, with verified fixes |
| 09 | [FAQ](docs/09-faq.md) | Licensing, fine-tuning, paper status, and the other questions everyone asks |
| 10 | [Next Steps](docs/10-next-steps.md) | Where to go once you finish this path |
| 11 | [Datasets & Licenses](docs/11-datasets-and-licenses.md) | Dataset sources/terms + what this repo does and does not redistribute |

## Repository structure

```text
docs/          — the numbered learning path above (00-overview.md … 11-datasets-and-licenses.md)
examples/      — runnable scripts, minimal → practical (start here for code)
notebooks/     — interactive walkthroughs (00_beginner_walkthrough.ipynb + a benchmark notebook)
problems/      — 8 advanced, production-style case studies (churn, fraud, pricing, attrition, loan risk)
reports/       — small, tracked CSV summaries backing README/docs tables
src/tabfm_benchmark/ — a small, reusable Python package for multi-dataset TabFM benchmarking
scripts/       — CLI entry points + CI/ops tooling (benchmark runner, weight-format fixer, E2E runner)
tests/         — unit/integration tests for src/tabfm_benchmark
HANDBOOK.md    — the operational/maintainer manual (CI, releases, runtime internals)
```

**New here?** Go to [`docs/00-overview.md`](docs/00-overview.md), not
`src/` or `problems/` — those are advanced/production-shaped code, not the
teaching material.

## First working example

The full walkthrough is [docs/03-first-run.md](docs/03-first-run.md); the
short version:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabfm import TabFMClassifier
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tabfm_v1_0_0.load(model_type="classification", device="cpu")
clf = TabFMClassifier(model=model, random_state=42)
clf.fit(X_train, y_train.to_numpy())          # no training loop — see docs/04
preds = clf.predict(X_test)
```

## This repo's own results (real runs, not illustrative)

Reference test-split champion metrics (TabFM vs. an XGBoost baseline) across
eight real business problems. The table below is backed by a small, tracked
CSV at [`reports/case_studies_summary.csv`](reports/case_studies_summary.csv).
The full `problems/` notebooks generate their own local artifacts when you run
them.

| Problem | Champion | Test Metrics |
|---|---|---|
| `problem1_telecom_churn` | `tabfm_ensemble_preset` | ROC-AUC `0.8468`, PR-AUC `0.6577`, Accuracy `0.8020`, F1 `0.5854` |
| `problem2_saas_subscription_churn` | `tabfm_default` | ROC-AUC `0.9802`, PR-AUC `0.8115`, Accuracy `0.9545`, F1 `0.7191` |
| `problem3_credit_card_transaction_fraud` | `xgboost_baseline` | ROC-AUC `0.7039`, PR-AUC `0.4053`, Accuracy `0.9970`, F1 `0.5714` |
| `problem4_insurance_claim_fraud` | `tabfm_ensemble_preset` | ROC-AUC `0.7917`, PR-AUC `0.5796`, Accuracy `0.8000`, F1 `0.6341` |
| `problem5_house_price_prediction` | `tabfm_default` | RMSE `24008.32`, MAE `15397.57`, R² `0.8793`, MAPE `9.52%` |
| `problem6_ride_fare_estimation` | `tabfm_ensemble_preset` | RMSE `4.1677`, MAE `2.2068`, R² `0.8343`, MAPE `18.53%` |
| `problem7_loan_default_prediction` | `tabfm_ensemble_preset` | ROC-AUC `0.8318`, PR-AUC `0.3887`, Accuracy `0.9450`, F1 `0.3125` |
| `problem8_employee_attrition_prediction` | `tabfm_ensemble_preset` | ROC-AUC `0.8449`, PR-AUC `0.6399`, Accuracy `0.8824`, F1 `0.5667` |

Note `problem3`: the **XGBoost baseline won**. TabFM is strong, not
universally superior — see [docs/07-evaluation.md](docs/07-evaluation.md)
for how to read this table and why every workflow here runs a baseline.

## Troubleshooting

Hit an error? Go straight to [docs/08-troubleshooting.md](docs/08-troubleshooting.md) —
it documents every failure this repo's own authors actually reproduced
while building it, including a real upstream packaging issue in `tabfm==1.0.0`'s
default weight-download path (with a one-command fix).

## Scope and limitations of this repository

- This repo exercises TabFM's **PyTorch backend only** (not JAX).
- TabFM itself has no published peer-reviewed paper as of this writing —
  see [docs/09-faq.md](docs/09-faq.md) for exactly what documentation does
  exist and how we verified it.
- Large raw datasets and model checkpoints are intentionally **not**
  committed to this repository (see `.gitignore`) — they're fetched or
  generated locally per the docs.
- This is an independent educational project. It is not affiliated with or
  endorsed by Google.

## License note — read before any commercial use

This repository's own code has no bearing on TabFM's licensing. **TabFM's
source code is Apache-2.0; its released model weights are under the
separate "TabFM Non-Commercial License v1.0"** — research/evaluation only.
Full explanation: [docs/09-faq.md#licensing](docs/09-faq.md#licensing).

## References

Primary and official sources this repository's documentation is grounded
in — see individual `docs/*.md` pages for inline citations tied to specific
claims.

**TabFM itself**
- [Google Research blog — "Introducing TabFM"](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) — the official announcement: motivation, architecture summary, benchmark methodology.
- [`google-research/tabfm` on GitHub](https://github.com/google-research/tabfm) — source code, official quick-start, license.
- [`tabfm` on PyPI](https://pypi.org/project/tabfm/) — the package this repo depends on.
- [TabFM model card (PyTorch weights)](https://huggingface.co/google/tabfm-1.0.0-pytorch) — intended use, limitations, exact license text.

**Architecture lineage**
- [Hollmann et al., 2022 — TabPFN](https://arxiv.org/abs/2207.01848)
- [Qu et al., 2025 — TabICL](https://arxiv.org/abs/2502.05564)
- [Brown et al., 2020 — in-context learning origin (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Erickson et al., 2025 — TabArena benchmark](https://arxiv.org/abs/2506.16791)

**Datasets used in this repo's examples/case studies**
- [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (with mirrors) — churn examples
- [TensorFlow-hosted credit card fraud dataset](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
- [IBM employee attrition dataset](https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv)
- [Insurance claims dataset](https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv)
- [Seaborn taxis dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv)

**Framework/library documentation**
- [PyTorch](https://pytorch.org/docs/stable/) · [scikit-learn](https://scikit-learn.org/stable/) · [XGBoost](https://xgboost.readthedocs.io/) · [Polars](https://docs.pola.rs/) · [pandas](https://pandas.pydata.org/docs/) · [Jupyter](https://jupyter.org/documentation) · [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) · [Kaggle API](https://www.kaggle.com/docs/api) · [OpenML](https://docs.openml.org/)

---

For CI/release engineering, runtime internals, and the full command
reference for maintaining this repository, see [`HANDBOOK.md`](HANDBOOK.md).
