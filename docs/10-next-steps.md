# 10 — Next Steps

> **You are here:** [Learning path](index.md#learning-path) → **10 Next Steps**
> **Previous:** [09 — FAQ](09-faq.md) · **Next (optional):** [11 — Datasets & Licenses](11-datasets-and-licenses.md)

If you've worked through 00–09 in order, you can now: explain what TabFM is
and isn't, install it, run classification and regression workloads, choose
between the default and ensemble presets, prepare a dataset correctly,
evaluate honestly against a baseline, and debug the failures you're most
likely to hit. That's the "zero to competent practitioner" arc this repo set
out to cover.

## Where to go inside this repo

| If you want... | Go to... |
|---|---|
| An interactive, narrated walkthrough | [`notebooks/00_beginner_walkthrough.ipynb`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/notebooks/00_beginner_walkthrough.ipynb) |
| A reusable multi-dataset benchmark you can extend | [`src/tabfm_benchmark/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/src/tabfm_benchmark) — add a new `DatasetSpec` in `datasets.py` and rerun `scripts/run_benchmark.py` |
| Full, business-realistic, production-style workflows (baseline comparison, model selection, decision policies) | [`problems/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/problems) — start with `problem1_telecom_churn` (classification) and `problem5_house_price_prediction` (regression), the two most self-contained |
| The operational/maintainer manual (CI, releases, runtime internals) | [`HANDBOOK.md`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/HANDBOOK.md) |

A sensible reading order through `problems/` if you want to see the full
range: `problem1_telecom_churn` → `problem5_house_price_prediction` →
`problem7_loan_default_prediction` → the remaining fraud/attrition/pricing
studies. Leave `problem2_saas_subscription_churn` (KKBox) for last — it
requires a Kaggle account and is the heaviest dataset in the repo.

## Where to go outside this repo

- **[Official `google-research/tabfm` repo](https://github.com/google-research/tabfm)** — the source of truth for TabFM itself. Its own `examples/` folder includes `tabarena_classification_example.py` / `tabarena_regression_example.py`, showing how Google ran its own published benchmark.
- **[TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard)** — see how TabFM ranks against XGBoost, LightGBM, CatBoost, and AutoML systems on a living, independently maintained benchmark.
- **Related tabular foundation models worth knowing:**
  - **[TabPFN](https://arxiv.org/abs/2207.01848)** (Hollmann et al.) — one of the two architectures TabFM's column-attention design draws from.
  - **[TabICL](https://arxiv.org/abs/2502.05564)** (Qu et al.) — the other, contributing the efficient in-context Transformer stage.
- **[TimesFM](https://github.com/google-research/timesfm)** — Google's related foundation model for time series, referenced as this release's spiritual predecessor.
- **Contributing to the upstream project**: `google-research/tabfm` has its own `CONTRIBUTING.md` (Google's standard CLA-based OSS process) if you want to report issues or contribute directly to TabFM itself, separate from this teaching repository.

## Keep your knowledge current

TabFM is a fast-moving research release (v1.0.0, first published 2026-06-30,
with no formal paper yet at time of writing). Watch the
[official GitHub repo](https://github.com/google-research/tabfm) and
[Google Research blog](https://research.google/blog/) for updates —
architecture caveats, license terms, or the promised BigQuery `AI.PREDICT`
integration mentioned in the announcement post could change after this
repo was last verified. If you find something here that's gone stale,
that's a sign to re-check the primary sources linked throughout these docs
rather than assume this documentation still holds.

## Give feedback on this repo

This repository aims to be the clearest possible on-ramp to TabFM for
newcomers. If a step didn't work, an explanation didn't land, or you hit an
error not covered in [08-troubleshooting.md](08-troubleshooting.md), open an
issue on this repository so the learning path can improve for the next
person.

---
**Optional:** [11 — Datasets & Licenses →](11-datasets-and-licenses.md)

Or: [Back to the docs index](index.md)
