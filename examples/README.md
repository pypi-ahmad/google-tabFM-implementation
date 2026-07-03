# Examples

Runnable scripts, ordered from minimal to practical. Each one is
self-contained — read top to bottom, no hidden helper modules. Full
explanations live in [`docs/`](../docs/); this file is just a map.

| Script | Tier | What it shows | Docs walkthrough |
|---|---|---|---|
| [`01_minimal_classification.py`](01_minimal_classification.py) | Minimal | Smallest possible TabFM classification call | [docs/03-first-run.md](../docs/03-first-run.md) |
| [`02_minimal_regression.py`](02_minimal_regression.py) | Minimal | Same, for a numeric target | [docs/03-first-run.md](../docs/03-first-run.md) |
| [`03_default_vs_ensemble.py`](03_default_vs_ensemble.py) | Beginner | Default preset vs. ensemble preset, same dataset, accuracy + timing | [docs/04-core-concepts.md §5](../docs/04-core-concepts.md#5-the-ensemble-preset-what-it-actually-turns-on) |
| [`04_churn_baseline_comparison.py`](04_churn_baseline_comparison.py) | Practical (real dataset) | TabFM vs. XGBoost baseline on real, downloaded churn data | [docs/07-evaluation.md](../docs/07-evaluation.md) |

## Running any example

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/<script>.py
```

All scripts:
- Auto-detect a GPU and fall back to CPU automatically if VRAM is under
  12 GiB (see [docs/08-troubleshooting.md](../docs/08-troubleshooting.md#cuda-out-of-memory)).
- Accept `TABFM_DEVICE`, `TABFM_CHECKPOINT_PATH`, `TABFM_CONTEXT_MAX_ROWS`,
  `TABFM_EVAL_MAX_ROWS` as environment-variable overrides — this repo's own
  convention, explained in
  [docs/06-training-or-usage-workflows.md §3](../docs/06-training-or-usage-workflows.md#3-this-repos-own-conventions-vs-the-tabfm-library-api).
- Cap dataset/context size deliberately, for a fast, predictable first run —
  raise the caps if your hardware allows it.

## Looking for the advanced tier?

This repo's "advanced" examples live in [`problems/`](../problems/): eight
full production-style case studies with multiple TabFM variants, business
decision policies, and persisted artifacts. They assume everything these
`examples/` scripts teach — see
[docs/10-next-steps.md](../docs/10-next-steps.md) for a guided entry point.
