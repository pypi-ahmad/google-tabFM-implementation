# 01 — Prerequisites

> **You are here:** [Learning path](../README.md#learning-path) → **01 Prerequisites**
> **Previous:** [00 — Overview](00-overview.md) · **Next:** [02 — Installation](02-installation.md)

This page tells you exactly what you need to know and have installed before
continuing. If you already know the concepts in section 1, skip to section 2.

## 1. Knowledge you need (with quick definitions)

You do **not** need to know deep learning, PyTorch internals, or how
Transformers work under the hood — this repo explains the TabFM-specific
concepts as you go in [04-core-concepts.md](04-core-concepts.md). You **do**
need to be comfortable with the following basics:

| Concept | One-line definition | If you're unfamiliar |
|---|---|---|
| **Python basics** | Writing/running functions, importing packages, using `pip`/`uv` | [Python's official tutorial](https://docs.python.org/3/tutorial/) |
| **pandas `DataFrame`** | A table object in Python — rows and named columns, like a spreadsheet | [pandas "10 minutes to pandas"](https://pandas.pydata.org/docs/user_guide/10min.html) |
| **Feature vs. target** | A *feature* is an input column (e.g. `age`, `income`); the *target* is the column you're predicting (e.g. `will_churn`) | — |
| **Classification vs. regression** | *Classification* predicts a category (e.g. churn: yes/no); *regression* predicts a number (e.g. house price) | — |
| **Train/test split** | Splitting your data so you evaluate on rows the model never used for fitting, to get an honest read on performance | [scikit-learn: `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| **Accuracy / RMSE** | *Accuracy*: fraction of classification predictions that were correct. *RMSE*: average prediction error size, in the target's own units, for regression | Defined in full in [07-evaluation.md](07-evaluation.md) |

### Self-check

You're ready for [02-installation.md](02-installation.md) if you can answer:
"if I have a table of houses with a `price` column, is predicting `price` a
classification or a regression problem, and why would I hold out a test set
before checking how good my prediction is?" *(Answer: regression, because
`price` is a continuous number; you hold out a test set so your performance
number reflects generalization instead of the model just memorizing rows it
already saw.)*

## 2. Environment you need

Everything in this repo was built and verified on the following stack.
Deviating from it (e.g., using conda instead of `uv`, or Windows instead of
Linux) is possible but unsupported by this repo's tooling and troubleshooting
docs.

| Requirement | Minimum | Notes |
|---|---|---|
| **OS** | Linux | macOS likely works but is untested here. Windows: use WSL2. |
| **Python** | 3.11+ | Pinned in [`pyproject.toml`](../pyproject.toml). |
| **Package manager** | [`uv`](https://docs.astral.sh/uv/) | This repo's dependency lockfile (`uv.lock`) assumes `uv`; do not mix with `pip install` inside the same environment. |
| **Disk space** | ~20 GB free | TabFM's pretrained weights are several gigabytes **per task** (classification and regression are downloaded separately) — see the callout in [02-installation.md](02-installation.md#3-disk-and-network-requirements--read-this-before-your-first-run). Budget generously. |
| **Network** | Required for first run only | The first time you load a TabFM model, weights are downloaded once from Hugging Face and cached locally. After that, everything runs offline. |
| **GPU** | Optional but recommended | An NVIDIA GPU with ≥12 GiB VRAM runs comfortably; this repo's own reference environment used a laptop GPU with 7.6 GiB VRAM, which works but triggers automatic CPU-safe fallbacks in the heavier examples (explained in [08-troubleshooting.md](08-troubleshooting.md)). No GPU at all still works — inference is just slower. |

### Why `uv` and not plain `pip`/`conda`?

`uv` resolves and locks dependencies deterministically (`uv.lock`), so the
exact versions of `torch`, `tabfm`, `xgboost`, etc. that this repo was
verified against are reproducible on your machine. If you already have
strong opinions about environment management, you can adapt the
`pyproject.toml` dependencies to your tool of choice — but the commands in
this repo's docs assume `uv`.

## 3. What you do *not* need up front

- A GPU (nice to have, not required for the beginner path in
  [03-first-run.md](03-first-run.md)).
- Any prior exposure to foundation models, Transformers, or in-context
  learning — [04-core-concepts.md](04-core-concepts.md) builds this from
  scratch.
- Kaggle/OpenML accounts — only one advanced case study
  ([`problems/problem2_saas_subscription_churn`](../problems/problem2_saas_subscription_churn))
  needs a Kaggle dataset; every other example uses freely downloadable or
  built-in data.

---
**Next:** [02 — Installation →](02-installation.md)
