# 09 — FAQ

> **You are here:** [Learning path](../README.md#learning-path) → **09 FAQ**
> **Previous:** [08 — Troubleshooting](08-troubleshooting.md) · **Next:** [10 — Next Steps](10-next-steps.md)

## Is TabFM an official Google product?

No. Both the PyPI package and the GitHub repository state explicitly:
*"This is not an officially supported Google product."* It is a Google
Research release — real, public, and maintained on GitHub, but without the
support guarantees of a Google Cloud product.

## Licensing

**Can I use TabFM commercially?** The code and the weights have different
licenses — this matters.

- **Source code** (the `tabfm` package, the GitHub repo): **Apache License
  2.0** — permissive, commercial use allowed.
- **Released model weights** (what you actually download and run
  predictions with): **TabFM Non-Commercial License v1.0**, a bespoke
  license published alongside the Hugging Face model card. It restricts use
  to non-commercial purposes (testing, evaluation, research not tied to
  commercial gain or production use) and additionally **prohibits
  redistributing** the weights or derivatives.

Practically: you can read, modify, and redistribute the *code* freely, but
running the pre-trained *weights* in a commercial product is not permitted
under the terms Google published. Read the full license text at the model
card before any production/commercial decision — this page is not legal
advice.

## Can I fine-tune TabFM on my own data?

No. The model card explicitly lists "tasks requiring task-specific
fine-tuning" as **not** an intended use. TabFM is in-context/zero-shot
only — see [04-core-concepts.md](04-core-concepts.md). If you need a model
tuned specifically to your data distribution, a conventional trained model
(e.g., XGBoost) remains the right tool.

## How does TabFM compare to XGBoost?

Neither universally wins. TabFM skips the training/tuning step and is
competitive out-of-the-box; a well-tuned XGBoost model can still beat it on
a specific dataset (this repo's own case studies show XGBoost winning on
`problem3_credit_card_transaction_fraud` — see
[07-evaluation.md](07-evaluation.md)). The practical takeaway: run both and
compare on your data, don't assume either wins by default.

## Why does TabFM need my whole training set at prediction time?

Because it makes predictions via **in-context learning** — your training
rows are the "context" the model reads before predicting, not material it
trains on ahead of time. This is architecturally different from a model
like XGBoost, which bakes what it learned from your training data into
fixed tree structures it can reuse without seeing that data again. See
[04-core-concepts.md](04-core-concepts.md) for the full explanation.

## Is there a research paper for TabFM?

**No arXiv paper or technical report exists for TabFM specifically**, as of
this writing (verified via arXiv's own search API — no results for
"TabFM"). Documentation consists of the
[official blog post](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/),
the Hugging Face model card, and the source code itself. The blog does cite
formal papers for the two architectures TabFM combines —
[TabPFN](https://arxiv.org/abs/2207.01848) and
[TabICL](https://arxiv.org/abs/2502.05564) — and for the benchmark
methodology used, [TabArena](https://arxiv.org/abs/2506.16791).

## Does TabFM support time series, text, or images?

No. It supports binary/multiclass classification (≤10 classes) and
regression on standard tabular (rows × columns) data only. For time series
specifically, Google has a separate research release,
**[TimesFM](https://github.com/google-research/timesfm)**, referenced in
TabFM's own announcement blog as the precedent this release follows.

## What's the difference between the JAX and PyTorch backends?

Both implement the same published architecture and are released as
separate Hugging Face checkpoints
([`google/tabfm-1.0.0-jax`](https://huggingface.co/google/tabfm-1.0.0-jax)
and
[`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
respectively). This repository exercises the **PyTorch backend exclusively**
for consistency across all examples, notebooks, and case studies — if you
want to experiment with the JAX backend, see the official repo's own
`examples/` folder for JAX-specific usage, which this project does not
otherwise cover.

## How big is the download, and can I shrink it?

The classification checkpoint measured about 6 GB on disk in this project's
own environment; regression is comparable. Both together is realistically
12+ GB. You can fetch only the task you need — see
[08-troubleshooting.md#reduce-download-size](08-troubleshooting.md#reduce-download-size).

## Does any of my data get sent to Google when I run TabFM?

Based on what's verifiable from the open-source code: after the one-time
weight download, `TabFMClassifier`/`TabFMRegressor` run entirely locally —
your feature/label data never appears in any network call in the source we
inspected. We flag this as **based on source-code inspection, not an
official privacy statement from Google** — if this matters for a compliance
decision, verify independently rather than relying on this FAQ.

## Is TabFM production-ready?

That depends entirely on your use case, your data, and your license
posture. Technically: yes, it runs reliably and produces real predictions,
as this entire repository demonstrates across eight case studies. Legally:
the non-commercial weights license (see [above](#licensing)) blocks most
production commercial deployments outright — check that first, before
technical readiness.

---
**Next:** [10 — Next Steps →](10-next-steps.md)
