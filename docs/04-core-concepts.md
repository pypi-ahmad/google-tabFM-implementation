# 04 — Core Concepts

> **You are here:** [Learning path](index.md#learning-path) → **04 Core Concepts**
> **Previous:** [03 — First Run](03-first-run.md) · **Next:** [05 — Working with Data](05-working-with-data.md)

You ran your first prediction in [03-first-run.md](03-first-run.md). This
page explains *why* the code worked the way it did, and defines every
TabFM-specific term you'll see in this repo's notebooks and scripts. Read
this once, fully — it's the vocabulary the rest of the repo assumes you
have.

## 1. In-context learning, precisely

**In-context learning (ICL)** means a model conditions its output on
examples given to it *as input at inference time*, instead of examples used
to update its weights during training. Large language models do this when
you put a few examples in a prompt; TabFM does the structural equivalent for
tables:

- Your **training rows** (features + known labels) become the "context."
- Your **new rows** (features only) are the "query."
- TabFM's Transformer attends over the context to produce a prediction for
  each query row, in a single forward pass — no gradient updates happen.

This is why calling `.fit(X_train, y_train)` on a `TabFMClassifier` /
`TabFMRegressor` does **not** train a model in the traditional sense. Under
the hood, `.fit()`:

1. Fits lightweight preprocessing objects (a categorical encoder, a
   numerical scaler) on `X_train`.
2. Stores `X_train`/`y_train` to be used as context at prediction time.

There is no loss function being minimized against your data, no epochs, and
no learning rate to tune. The actual "learning" happened once, at Google,
when TabFM's weights were trained on hundreds of millions of synthetic
datasets — what you're doing is closer to "showing the model examples" than
"training a model."

## 2. Zero-shot

**Zero-shot** means the model produces a useful prediction with **zero**
gradient-descent training steps on your specific data. It does not mean the
model ignores your data — it still needs your training rows as context (see
above). "Zero-shot" and "in-context learning" describe the same underlying
mechanism from two angles: ICL is *how* it works, zero-shot is *what you get*
(a usable model with no training loop on your side).

## 3. The architecture, one level deeper

Recall from [00-overview.md](00-overview.md) that TabFM combines ideas from
two published tabular foundation models, per Google's own announcement:

```
┌─────────────────────────────────────────────────────────────┐
│  Your table (rows × columns, mixed numeric/categorical)      │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
                 1. COLUMN ATTENTION
        (learns what kind of feature each column is —
         inspired by TabPFN's column-wise attention)
                             ▼
                 2. ROW COMPRESSION
        (each row becomes one dense vector — keeps the
         next stage cheap even for many rows)
                             ▼
              3. IN-CONTEXT TRANSFORMER
        (attends training rows against query rows to
         predict — the efficient approach from TabICL)
                             ▼
                    Predictions for your new rows
```

*Source:
[Google Research blog, "Introducing TabFM"](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/);
architecture lineage papers:
[TabPFN (Hollmann et al., 2022)](https://arxiv.org/abs/2207.01848),
[TabICL (Qu et al., 2025)](https://arxiv.org/abs/2502.05564).*

You can see these stages reflected directly in the model's configuration
(from [`tabfm/src/pytorch/tabfm_v1_0_0.py`](https://github.com/google-research/tabfm/blob/main/tabfm/src/pytorch/tabfm_v1_0_0.py)):

| Config field | What it controls |
|---|---|
| `col_num_blocks`, `col_nhead`, `col_num_inds` | Column-attention stage depth/width |
| `row_num_blocks`, `row_nhead`, `row_num_cls` | Row-compression stage |
| `icl_num_blocks`, `icl_nhead` | In-context Transformer depth/width |
| `embed_dim` | Internal vector size used throughout |
| `max_classes` | Hard limit of **10** — a fixed architectural constant, not a setting you can raise |
| `feature_group_size`, `num_freq` | How numeric features are grouped and Fourier-encoded before entering the model |

You will not need to touch these directly — they're baked into the released
checkpoint — but recognizing them removes the mystery when you see
`ClassificationConfig`/`RegressionConfig` in this repo's source.

## 4. Training data: synthetic, not real

TabFM was **trained entirely on synthetic data**: hundreds of millions of
artificial tabular datasets generated from structural causal models (SCMs),
not on scraped real-world tables. Google's stated reason is that large,
diverse, non-proprietary real tabular datasets are scarce. The practical
consequence, stated explicitly in the model's own documentation: **"Performance
on specific real-world domains, minority groups, or edge distributions is
not fully characterised."** Treat every result on your own data as something
to measure, not assume — this is the entire point of
[07-evaluation.md](07-evaluation.md).

## 5. The ensemble preset (what it actually turns on)

Both `TabFMClassifier` and `TabFMRegressor` have a `.ensemble(model=...)`
constructor alongside the plain `TabFMClassifier(model=...)` constructor.
This repo's code and notebooks call the plain form "`default`" and the
`.ensemble(...)` form the "**ensemble preset**." Here is exactly what
changes (from the installed `tabfm` package's own docstrings):

| Setting | Default | Ensemble preset |
|---|---|---|
| `n_estimators` | 32 | 32 |
| `average_logits` | `True` (average logits, then softmax) | `False` (average calibrated probabilities instead) |
| `n_feature_crosses` | `0` (disabled) | `"sqrt"` — adds √(n_features) random pairwise feature products per ensemble member |
| `n_svd_features` | `0` (disabled) | `"sqrt"` — adds √(n_features) features derived from an SVD projection of the data |
| `enable_nnls` | `False` | `True` — blends ensemble members with **non-negative least squares (NNLS)**-fitted weights instead of a plain average |
| `binary_calibration_method` | `None` | `"platt"` — rescales predicted probabilities for binary classification using **Platt scaling**, a standard technique that fits a small logistic curve on top of raw scores to make probabilities better-calibrated |
| `multiclass_calibration_method` | `None` | `"vector"` — the multiclass analogue of Platt scaling |

In plain terms: the **default** preset runs one straightforward ensemble of
`n_estimators` TabFM forward passes (each seeing a different random
feature/row permutation) and averages them. The **ensemble preset** adds
extra engineered features per member, learns *smarter, non-uniform* weights
for combining members (NNLS) instead of a flat average, and calibrates the
output probabilities. It is typically more accurate and always more
expensive to run. You choose which to use — see
[06-training-or-usage-workflows.md](06-training-or-usage-workflows.md) for
when each makes sense.

### What is "NNLS blending," in one sentence?

Instead of averaging all ensemble members equally, NNLS solves a small
optimization problem to find *non-negative* weights for each member that
best match known outcomes — members that predict better get a bigger say,
and no member gets a negative (canceling-out) weight.

### What is "Platt scaling," in one sentence?

A simple technique (Platt, 1999) that fits a 1-D logistic regression on top
of a model's raw scores so that a predicted "0.8" actually corresponds to
roughly an 80% real-world frequency of the positive class — it fixes
*overconfident or underconfident* probabilities without changing which class
is predicted.

## 6. Glossary

Every term used in this repo's notebooks and docs, one place:

| Term | Meaning |
|---|---|
| **In-context learning (ICL)** | Conditioning a prediction on examples given as input, not on gradient updates ([§1](#1-in-context-learning-precisely)). |
| **Zero-shot** | Producing a usable prediction with no training steps on your specific data ([§2](#2-zero-shot)). |
| **Context** | The training rows shown to TabFM at prediction time. |
| **Context cap** | This repo's own convention (`TABFM_CONTEXT_MAX_ROWS`) for limiting how many context rows are used, to control memory — not a `tabfm` library setting. |
| **Ensemble preset** | The `.ensemble(model=...)` constructor preset described in [§5](#5-the-ensemble-preset-what-it-actually-turns-on). |
| **Feature cross** | An engineered feature made by combining two existing features (e.g., multiplying them), used by the ensemble preset. |
| **SVD feature** | An engineered feature derived from a low-rank (SVD) projection of the input data, used by the ensemble preset. |
| **NNLS (non-negative least squares) blending** | Learning non-negative combination weights for ensemble members instead of averaging them equally. |
| **Calibration / Platt scaling** | Adjusting predicted probabilities so they better reflect real-world frequencies, without changing the predicted class. |
| **Champion model** | This repo's term (used in `problems/`) for whichever model variant scored best on the validation set for a given problem. |
| **PR-AUC** | Area under the precision-recall curve — a classification metric that is more informative than accuracy when the positive class is rare (e.g., fraud, churn). Defined fully in [07-evaluation.md](07-evaluation.md). |
| **Threshold policy / top-k campaign** | Decision rules built on top of a model's predicted probabilities (e.g., "flag the top 10% highest-risk customers") — covered in [07-evaluation.md](07-evaluation.md) and the [`problems/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/problems) case studies. |

## 7. TabFM vs. a conventional tabular model (XGBoost)

| | Conventional (e.g., XGBoost) | TabFM |
|---|---|---|
| Training required per dataset | Yes — fit trees on your data | No — frozen weights, your data is context |
| Hyperparameter tuning | Usually needed for best results | Not required to get a first result |
| Inference cost | Very fast, small model | Slower, larger model; scales with context size |
| Interpretability tools | Mature (feature importance, SHAP) | Less mature/standard |
| License for commercial use | Typically permissive (XGBoost is Apache-2.0) | **Not permitted** for TabFM's released weights (see [09-faq.md](09-faq.md#licensing)) |
| Best accuracy on a given dataset | Can win with enough tuning | Competitive out-of-the-box; not guaranteed to win — see [07-evaluation.md](07-evaluation.md) |

This repo runs both side by side in every case study under
[`problems/`](https://github.com/pypi-ahmad/google-tabFM-implementation/tree/main/problems) so you can see the tradeoff on real data rather
than take either side's marketing claim.

## References

- [Google Research blog — "Introducing TabFM"](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
- [Hollmann et al., 2022 — TabPFN](https://arxiv.org/abs/2207.01848)
- [Qu et al., 2025 — TabICL](https://arxiv.org/abs/2502.05564)
- [Brown et al., 2020 — in-context learning origin](https://arxiv.org/abs/2005.14165)
- [`tabfm` source — `classifier_and_regressor.py`](https://github.com/google-research/tabfm/blob/main/tabfm/src/classifier_and_regressor.py) — authoritative source for every constructor parameter described above.

---
**Next:** [05 — Working with Data →](05-working-with-data.md)
