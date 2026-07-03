# 06 — Usage Workflows

> **You are here:** [Learning path](index.md#learning-path) → **06 Usage Workflows**
> **Previous:** [05 — Working with Data](05-working-with-data.md) · **Next:** [07 — Evaluation](07-evaluation.md)

This page walks through the two supported workflows (classification,
regression), then explains a distinction that trips people up: **this
repo's own conventions** (environment variables, CLI scripts) layered on top
of **the plain `tabfm` library API**. Confusing the two is the #1 source of
"where is this documented?" confusion when reading this repo's notebooks.

## 1. Classification workflow

```python
from tabfm import TabFMClassifier
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

model = tabfm_v1_0_0.load(model_type="classification", device="cpu")

# Default preset — one straightforward ensemble average.
clf = TabFMClassifier(model=model, random_state=42)
clf.fit(X_train, y_train.to_numpy())
preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)

# Ensemble preset — feature crosses + SVD features + NNLS blending +
# calibration. See docs/04-core-concepts.md, "The ensemble preset," for what
# this changes and why it costs more.
clf_ensemble = TabFMClassifier.ensemble(model=model, random_state=42)
clf_ensemble.fit(X_train, y_train.to_numpy())
```

Both classes share the same `model` object — loading weights once and
reusing them across classifier instances (default vs. ensemble, or across
multiple datasets in a benchmark loop) avoids repeated disk/network I/O.
This is exactly what
[`src/tabfm_benchmark/benchmark.py`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/src/tabfm_benchmark/benchmark.py)
does via `@lru_cache` on `_load_model_for_task`.

## 2. Regression workflow

```python
from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

model = tabfm_v1_0_0.load(model_type="regression", device="cpu")

reg = TabFMRegressor(model=model, random_state=42)
reg.fit(X_train, y_train.to_numpy())
preds = reg.predict(X_test)
```

Internally, `TabFMRegressor` standardizes your target (`StandardScaler`)
before it reaches the model and inverse-transforms predictions back to your
original units automatically — you always work in your target's real units
(dollars, days, whatever `y` originally was), never a normalized scale.

## 3. This repo's own conventions vs. the `tabfm` library API

The `tabfm` package itself only exposes **Python function/constructor
arguments** — `device=`, `checkpoint_path=`, `model_type=`,
`n_estimators=`, etc. It defines **no environment variables** of its own.

Everything below is a convention this repository's own code
(`src/tabfm_benchmark/`, `scripts/`, `problems/*.ipynb`) layers on top, for
reproducibility and CI automation:

| Env var | Read by | Purpose |
|---|---|---|
| `TABFM_DEVICE` | Notebooks in `problems/` | Requested device (`auto`/`cpu`/`cuda`), before this repo's own low-VRAM fallback logic runs |
| `TABFM_CHECKPOINT_PATH` | Notebooks, `scripts/run_strict_e2e.py`, `scripts/run_benchmark.py`, `examples/` | Overrides automatic Hugging Face download with a local checkpoint directory — see [08-troubleshooting.md](08-troubleshooting.md) |
| `TABFM_CONTEXT_MAX_ROWS` | Notebooks, `examples/02_minimal_regression.py` | Caps training-context rows to control memory |
| `TABFM_EVAL_MAX_ROWS` | Notebooks, `examples/02_minimal_regression.py` | Caps rows scored per prediction call |
| `TABFM_FAST_MODE` | `problems/*.ipynb`, `scripts/run_strict_e2e.py` | Switches to a smaller-estimator "fast" profile for quick CI runs |
| `STRICT_E2E` | `problems/*.ipynb`, `scripts/run_strict_e2e.py` | Set by the CI/E2E runner so notebooks can branch into a hardened execution profile |

**Do not expect these to work with the official `tabfm` package outside this
repo** — they are read by *this project's own code*, not by `import tabfm`
itself. If you copy a snippet from `problems/` into a fresh script without
this repo's surrounding code, these variables silently do nothing.

## 4. Using the benchmark CLI

For quickly comparing TabFM against a portfolio of datasets without writing
a notebook, use this repo's benchmark script:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_benchmark.py \
  --device cpu \
  --seed 42 \
  --n-estimators 32 \
  --checkpoint-path "${TABFM_CHECKPOINT_PATH:-}" \
  --output artifacts/tabfm_benchmark_results.parquet \
  --summary-output artifacts/tabfm_benchmark_summary.csv
```

This runs TabFM's ensemble preset across the seven built-in scikit-learn
datasets defined in
[`src/tabfm_benchmark/datasets.py`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/src/tabfm_benchmark/datasets.py)
(iris, wine, breast cancer, digits, covtype, diabetes, california housing)
and writes per-dataset and summary results. Read
[`src/tabfm_benchmark/benchmark.py`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/src/tabfm_benchmark/benchmark.py)
end to end once you're comfortable with the basics — it's under 400 lines
and is this repo's clearest example of a production-shaped (not notebook)
TabFM integration.

## 5. Going further: the advanced case studies

Once you're comfortable with the workflows above, [`problems/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/problems)
contains eight full business-style case studies (churn, fraud, pricing,
attrition, loan risk) that combine TabFM default + ensemble presets, an
XGBoost baseline, model selection ("champion" picking), and
decision-policy layers (threshold tuning, top-k targeting) on real datasets.
They assume everything in this doc and [05](05-working-with-data.md) — see
[10-next-steps.md](10-next-steps.md) for a guided entry point into them.

## References

- [`tabfm/src/classifier_and_regressor.py`](https://github.com/google-research/tabfm/blob/main/tabfm/src/classifier_and_regressor.py) — authoritative source for every method/parameter shown above
- [`src/tabfm_benchmark/benchmark.py`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/src/tabfm_benchmark/benchmark.py) — this repo's reusable benchmark implementation

---
**Next:** [07 — Evaluation →](07-evaluation.md)
