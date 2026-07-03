# 03 — First Run

> **You are here:** [Learning path](index.md#learning-path) → **03 First Run**
> **Previous:** [02 — Installation](02-installation.md) · **Next:** [04 — Core Concepts](04-core-concepts.md)

Time to make your first TabFM prediction. This page walks through
[`examples/01_minimal_classification.py`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/examples/01_minimal_classification.py)
line by line, tells you exactly what to expect, and lists the mistakes
beginners actually make with this code.

## 1. Run it

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/01_minimal_classification.py
```

If this fails with a `FileNotFoundError` mentioning `pytorch_model.bin`, you hit
the known `tabfm==1.0.0` PyTorch checkpoint filename mismatch. Fix it once by
running the converter described in
[02-installation.md §3.5](02-installation.md#35-one-time-weight-conversion-recommended), then rerun the example.

## 2. What it does, step by step

```python
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
```
Loads a small (569-row), real, built-in dataset: 30 numeric measurements
from breast tumor scans, target = malignant vs. benign. It ships with
scikit-learn, so there's no download or external file involved.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Holds out 20% of the data as a test set the model will never see during
`.fit()`, so the accuracy you get back reflects generalization, not
memorization (see [01-prerequisites.md](01-prerequisites.md) if this
sentence was new to you).

```python
model = tabfm_v1_0_0.load(model_type="classification", device=DEVICE, ...)
```
Loads TabFM's pre-trained classification weights. `DEVICE` is auto-detected:
a GPU is used only if one is available **and** has at least 12 GiB of VRAM
(TabFM's weights alone use several GB; less headroom risks an out-of-memory
error — see [08-troubleshooting.md](08-troubleshooting.md#cuda-out-of-memory)).
Otherwise it runs on CPU — slower, but always correct.

```python
clf = TabFMClassifier(model=model, random_state=42)
clf.fit(X_train, y_train.to_numpy())
```
"Fits" the classifier — really, prepares encoders and stores `X_train` as
context. See [04-core-concepts.md](04-core-concepts.md) for why there's no
training loop here.

```python
preds = clf.predict(X_test).astype(y_train.to_numpy().dtype)
accuracy = accuracy_score(y_test, preds)
```
Predicts on the held-out rows and scores them. The explicit `.astype(...)`
works around a real, verified quirk: `TabFMClassifier.predict()` returns an
object-dtype array, which some strict scikit-learn metric functions
misclassify as an "unknown" target type. See
[08-troubleshooting.md](08-troubleshooting.md#predictions-break-scikit-learn-metrics-with-an-unknown-target-type-error) for
the full explanation — this is not something you did wrong.

## 3. Expected output

Running the script end to end in this repo's own reference environment
(CPU fallback path, since its GPU has under 12 GiB VRAM) produced:

```
Detected GPU with 7.6 GiB VRAM (<12 GiB) — using CPU for a stable run. See docs/08-troubleshooting.md#cuda-out-of-memory.
Loading TabFM classification weights (device=cpu) ...
Test accuracy: 0.9737
Test set size: 114 rows | Train context size: 455 rows
```

**How to verify your own run succeeded:** you should see a `Test accuracy`
line with a value roughly in the 0.93–0.98 range (this depends slightly on
your hardware's floating-point behavior and whether you run on CPU or GPU,
but should not vary wildly — this is a well-separated, easy dataset). If you
instead see an exception, go straight to
[08-troubleshooting.md](08-troubleshooting.md); the top entries there cover
the two failures most people hit on a first run.

If you have a GPU with ≥12 GiB VRAM, force it explicitly to see a faster
run — this repo's own testing on a smaller (7.6 GiB) GPU, forcing CUDA
anyway, produced the same `0.9737` accuracy in a few seconds instead of
CPU's longer runtime:

```bash
TABFM_DEVICE=cuda uv run python examples/01_minimal_classification.py
```

## 4. Now try regression

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/02_minimal_regression.py
```

This mirrors the classification script for a numeric target (California
housing prices). Verified reference output:

```
Loading TabFM regression weights (device=cuda) ...
Test MAE: 0.3694 (target unit: $100,000s)
Test R^2: 0.7775
Test set size: 300 rows | Train context size: 300 rows
```

Note the script caps both training-context rows and evaluation rows to 300
by default — this is a **memory** decision, not a library limitation. See
[05-working-with-data.md](05-working-with-data.md#2-documented-hard-limits-verified-from-googles-own-model-card)
for why context size matters, and raise
`TABFM_CONTEXT_MAX_ROWS`/`TABFM_EVAL_MAX_ROWS` if your hardware allows it.

## 5. Common beginner mistakes

- **Forgetting `stratify=y` on a classification split.** Without it, a rare
  class can end up entirely in the test set (or entirely absent from it),
  producing a misleadingly high or undefined accuracy.
- **Comparing accuracy across runs without a fixed `random_state`.** TabFM's
  ensemble mechanics involve randomness (feature/row permutations); without
  a seed, two runs on the same data can give slightly different numbers —
  by design, not a bug.
  Both example scripts pin `random_state=42` for this reason.
- **Assuming a bigger GPU is always better.** More VRAM lets you use bigger
  context sizes and more ensemble members, but a small, well-separated
  dataset like `breast_cancer` gets a strong result even on CPU. Don't
  reach for a bigger machine before you've confirmed you actually need one.
- **Running the regression example with the full test set on a small GPU.**
  This repo's own reference GPU (7.6 GiB) ran out of memory scoring all
  ~4,100 California-housing test rows at once — capping `EVAL_ROW_CAP`
  fixed it. See [08-troubleshooting.md](08-troubleshooting.md#cuda-out-of-memory).

## 6. Want an interactive version?

[`notebooks/00_beginner_walkthrough.ipynb`](https://github.com/pypi-ahmad/google-tabFM-implementation/blob/main/notebooks/00_beginner_walkthrough.ipynb)
covers the same ground as this page cell-by-cell, with inline explanations,
and goes one step further by comparing the default preset against the
ensemble preset on the same dataset.

---
**Next:** [04 — Core Concepts →](04-core-concepts.md)
