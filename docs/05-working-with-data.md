# 05 — Working with Data

> **You are here:** [Learning path](../README.md#learning-path) → **05 Working with Data**
> **Previous:** [04 — Core Concepts](04-core-concepts.md) · **Next:** [06 — Usage Workflows](06-training-or-usage-workflows.md)

This page covers what TabFM expects as input, its documented hard limits,
and practical prep steps this repo recommends beyond what's officially
required. **We label recommendations that go beyond Google's own
documentation explicitly** — don't confuse repo convention with a library
requirement.

## 1. Input contract

Both `TabFMClassifier.fit(X, y)` and `TabFMRegressor.fit(X, y)` expect:

- **`X`**: a `pandas.DataFrame` with one row per example and one column per
  feature. Columns can be a mix of numeric and categorical dtypes — TabFM
  handles the encoding internally (via `TransformToNumerical` and
  `CategoricalOrdinalEncoder` under the hood; you don't call these
  yourself).
- **`y`**: a 1-D array-like of labels (classification) or numeric targets
  (regression). This repo's own code consistently passes `y.to_numpy()`
  (converting a pandas `Series` to a plain array) — do the same for
  predictable behavior.

## 2. Documented hard limits (verified from Google's own model card)

| Limit | Value | Source |
|---|---|---|
| **Maximum classes** | 10 | Fixed architectural constant (`max_classes` in `ClassificationConfig`); classification datasets with more than 10 classes are not supported. |
| **Feature count** | Optimized for tables up to ~500 features | Stated as guidance in the model card, not a hard failure — behavior may degrade beyond this, not necessarily error. |
| **Row/context scaling** | Memory scales with training-row count (all rows become context) | Model card states this qualitatively; no fixed numeric row cap is published. |
| **Tasks supported** | Binary/multiclass classification, regression | No ranking, survival analysis, multi-output, or time-series-native support. |
| **Fine-tuning** | Not supported / not an intended use | Zero-shot/in-context only — see [04-core-concepts.md](04-core-concepts.md). |

This repo enforces the class-count limit explicitly in
[`src/tabfm_benchmark/benchmark.py`](../src/tabfm_benchmark/benchmark.py)
(`MAX_SUPPORTED_CLASSES = 10`), skipping any dataset that exceeds it rather
than letting it fail deep inside the model — see
[`tests/test_class_guard.py`](../tests/test_class_guard.py) for the test
that pins this behavior.

## 3. What's *not* documented — and our recommendation

Google's primary sources (blog, model card, GitHub README) do **not** state
an explicit policy for:

- **Missing values** — no documented behavior.
- **High-cardinality categorical columns** (e.g., a `user_id`-like column
  with thousands of unique values) — no documented limit.

**Our recommendation (not an official TabFM requirement):** treat TabFM like
any other model that expects clean, complete input:

- Impute or explicitly drop missing values before calling `.fit()` /
  `.predict()` (e.g., `sklearn.impute.SimpleImputer`).
- Cap or bucket extremely high-cardinality categorical columns (an ID column
  is rarely a useful predictive feature anyway).
- Remove obvious leakage columns (anything that encodes the target directly,
  e.g., a `days_until_cancellation` column when predicting churn).

You'll see this pattern in every notebook under [`problems/`](../problems/):
a dedicated cleaning step runs before any TabFM call.

## 4. Leakage-safe splitting

Always split before fitting anything, and stratify classification splits by
the target so class proportions are preserved in both halves:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,       # reproducibility — see docs/01-prerequisites.md
    stratify=y,             # classification only; omit for regression
)
```

This is the exact pattern used in
[`src/tabfm_benchmark/benchmark.py`](../src/tabfm_benchmark/benchmark.py)
(`_split_dataset`) and every example in this repo.

## 5. Practical dataset-prep checklist

Before calling `TabFMClassifier`/`TabFMRegressor` on your own data:

- [ ] Target column separated out from features (`X` vs. `y`).
- [ ] No target-leaking columns left in `X`.
- [ ] Missing values handled (imputed or rows/columns dropped deliberately).
- [ ] Categorical columns are genuine categories, not high-cardinality IDs.
- [ ] Classification: 10 or fewer classes in `y`.
- [ ] Train/test split done with a fixed `random_state`, stratified if
      classification.
- [ ] You know roughly how many rows will end up as context (`len(X_train)`)
      — large context sizes cost more time and memory (see
      [06](06-training-or-usage-workflows.md) and
      [08-troubleshooting.md](08-troubleshooting.md#cuda-out-of-memory)).

## References

- [TabFM model card (limitations section)](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- [`tabfm/src/classifier_and_regressor.py`](https://github.com/google-research/tabfm/blob/main/tabfm/src/classifier_and_regressor.py) — preprocessing classes referenced above
- [scikit-learn: `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---
**Next:** [06 — Usage Workflows →](06-training-or-usage-workflows.md)
