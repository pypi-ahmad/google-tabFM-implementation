# 07 — Evaluation

> **You are here:** [Learning path](../README.md#learning-path) → **07 Evaluation**
> **Previous:** [06 — Usage Workflows](06-training-or-usage-workflows.md) · **Next:** [08 — Troubleshooting](08-troubleshooting.md)

A prediction without an honest evaluation is a guess with extra steps. This
page defines the metrics this repo uses, shows how to compare TabFM against
a baseline, and reports what Google itself has (and hasn't) published about
TabFM's benchmark performance — with the exact caveats that matter.

## 1. Metrics, defined plainly

| Metric | Task | Plain-English definition | Range / reading it |
|---|---|---|---|
| **Accuracy** | Classification | Fraction of predictions that were exactly correct | 0–1, higher is better; misleading when classes are imbalanced |
| **F1 (macro)** | Classification | Balances precision (of predicted positives, how many were right) and recall (of actual positives, how many were found), averaged equally across classes | 0–1, higher is better |
| **ROC-AUC** | Classification | Probability the model ranks a random positive example above a random negative one | 0.5 = random guessing, 1.0 = perfect ranking |
| **PR-AUC** | Classification (imbalanced) | Like ROC-AUC, but focused on the positive class — far more informative than accuracy or ROC-AUC when the positive class is rare (fraud, churn) | Higher is better; baseline = the positive class's prevalence rate |
| **RMSE** | Regression | Root mean squared error — average prediction error size, in the target's own units, penalizing large errors more | 0 = perfect, lower is better |
| **MAE** | Regression | Mean absolute error — average prediction error size, in the target's own units, treating all errors equally | 0 = perfect, lower is better |
| **R²** | Regression | Fraction of the target's variance explained by the model | 1.0 = perfect, 0.0 = no better than predicting the mean, can go negative |
| **MAPE** | Regression | Mean absolute percentage error — average error as a percentage of the true value | 0% = perfect, lower is better; unstable near zero-valued targets |

## 2. Always compare against a baseline

A TabFM accuracy of 0.94 means nothing on its own — you need to know what a
simple, well-understood model achieves on the *same* split. Every workflow
in this repo trains an **XGBoost baseline** alongside TabFM for exactly this
reason (see
[`problems/problem1_telecom_churn`](../problems/problem1_telecom_churn) for
a full worked example). The pattern:

```python
from xgboost import XGBClassifier

baseline = XGBClassifier(random_state=42)
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)
```

Compute the same metric for both models on the identical test split, then
compare. Whichever wins **on your specific dataset** is the one to use in
production — don't assume either model universally wins (see
[04-core-concepts.md §7](04-core-concepts.md#7-tabfm-vs-a-conventional-tabular-model-xgboost)).

## 3. This repo's own results

The table below is reproduced from persisted artifacts in
[`problems/*/artifacts/`](../problems/) — real numbers from real runs in
this repository, not illustrative figures:

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

**Read this table carefully**: TabFM wins the "champion" slot in 7 of 8
problems here, but note `problem3` — the **XGBoost baseline won** on credit
card fraud detection. This is the whole point of always running a baseline:
TabFM is strong, not universally superior.

## 4. What Google has published about TabFM's benchmark performance

Google evaluated TabFM on **[TabArena](https://arxiv.org/abs/2506.16791)**
(Erickson et al., NeurIPS 2025), an independent, third-party "living
benchmark" for tabular ML — 38 classification datasets (700–150,000 rows)
and 13 regression datasets. Two important, verified caveats before citing
any number:

1. **Google's own published results (in the
   [`google-research/tabfm` GitHub repo's `results/` folder](https://github.com/google-research/tabfm/tree/main/results))
   contain only TabFM's own scores — no XGBoost/LightGBM/AutoML baseline
   numbers are included in those files.** The blog's claim of beating
   heavily-tuned baselines rests on TabArena's own external Elo-based
   leaderboard (linked from the blog), not on a self-contained comparison
   table Google publishes directly.
2. TabArena reports **ROC-AUC only for binary classification** datasets (30
   of the 38) and **log-loss for multiclass** datasets (the remaining 8) —
   they aren't directly comparable to each other.

With those caveats stated, here is what we independently computed by
downloading and parsing Google's own published per-fold result files
(`results/*.parquet` in the official repo — verified 30 binary datasets,
matching the documented dataset count):

| Model | Macro-averaged ROC-AUC across 30 TabArena binary-classification datasets |
|---|---|
| TabFM (default) | **0.859** |
| TabFM (ensemble preset) | **0.860** |

For the full picture — including head-to-head Elo rankings against
XGBoost, LightGBM, CatBoost, and AutoML systems — see the live
[TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard)
directly rather than relying on any single number reproduced here; it is
an actively maintained external resource, not a snapshot we control.

## 5. Beyond raw metrics: decision policies

Raw accuracy/ROC-AUC rarely matches a business decision directly — "should
we call this customer" depends on the cost of a false alarm vs. a missed
churn. Every case study in [`problems/`](../problems/) builds a **threshold
policy** or **top-k campaign** layer on top of raw model scores (e.g., "flag
the riskiest 10% of customers"), using cost assumptions the notebooks label
explicitly as illustrative. Treat those specific dollar figures as
placeholders for your own — the *pattern* (translate a score into a
capacity-constrained action) is the reusable part, and is covered fully in
those notebooks rather than duplicated here.

## References

- [TabArena paper (Erickson et al., 2025)](https://arxiv.org/abs/2506.16791)
- [TabArena live leaderboard](https://huggingface.co/spaces/TabArena/leaderboard)
- [`google-research/tabfm` results data](https://github.com/google-research/tabfm/tree/main/results) — source of the macro-ROC-AUC figures above, independently recomputed for this doc
- [Google Research blog — "Introducing TabFM"](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

---
**Next:** [08 — Troubleshooting →](08-troubleshooting.md)
