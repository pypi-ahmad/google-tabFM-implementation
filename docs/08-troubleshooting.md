# 08 — Troubleshooting

> **You are here:** [Learning path](../README.md#learning-path) → **08 Troubleshooting**
> **Previous:** [07 — Evaluation](07-evaluation.md) · **Next:** [09 — FAQ](09-faq.md)

Symptom → cause → fix, ordered by how likely you are to hit each one.
Everything here was actually reproduced in this repo's own environment
while building it — nothing is theoretical.

## Missing checkpoint file on a fresh install

**Symptom:**
```
FileNotFoundError: Weights not found at: .../classification/pytorch_model.bin
```
right after `tabfm_v1_0_0_pytorch.load()` finishes downloading from Hugging
Face.

**Cause (verified, upstream — not your setup):** `tabfm==1.0.0`'s automatic
download path looks for a file named `pytorch_model.bin`. The files
actually published at
[`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
are named `model.safetensors`. We confirmed this directly: listing the live
Hugging Face repo's files shows only `model.safetensors` under both
`classification/` and `regression/`, while
[`tabfm/src/pytorch/tabfm_v1_0_0.py`](https://github.com/google-research/tabfm/blob/main/tabfm/src/pytorch/tabfm_v1_0_0.py)'s
`load()` function hardcodes the `.bin` filename. This is a packaging
mismatch in the released `tabfm` PyPI package, not a mistake you made.

**Fix:** run this repo's converter once, which downloads the safetensors
weights and re-saves them in the format `load()` expects (tensor names are
identical between the two formats — this is a straight container
conversion, verified with `strict=True` state-dict loading, not a model
port):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --task both
export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch
```

Then pass `checkpoint_path=os.environ["TABFM_CHECKPOINT_PATH"]` to
`tabfm_v1_0_0.load(...)` (all example scripts in this repo already do this
when the env var is set).

## CUDA out of memory

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ... MiB.
```

**Cause:** TabFM's weights alone occupy several GB of GPU memory once
loaded, and memory use during `.fit()`/`.predict()` scales further with
**both** the number of training-context rows and the number of rows scored
per `.predict()` call. On a GPU with less than ~12 GiB of VRAM, there's very
little headroom left for that scaling before you hit a wall — this repo's
own reference GPU (7.6 GiB) reproduces this reliably.

**Fixes, in order of preference:**
1. Run on CPU instead (slower, always correct): pass `device="cpu"`, or set
   `TABFM_DEVICE=cpu`.
2. Reduce the training-context size (`TABFM_CONTEXT_MAX_ROWS` in this
   repo's examples, or just pass a smaller `X_train` yourself).
3. Reduce the number of rows scored per `.predict()` call
   (`TABFM_EVAL_MAX_ROWS` in `examples/02_minimal_regression.py`) — scoring
   a large test set in one call is a common, easy-to-miss OOM cause even
   when the *training* context is already small.
4. Reduce `n_estimators` on `TabFMClassifier`/`TabFMRegressor`.
5. If you have ≥12 GiB VRAM and still hit this, reduce `batch_size` (default
   `1`) is already minimal — check for other processes holding GPU memory
   with `nvidia-smi`.

This repo's own `problems/` notebooks encode a `<12 GiB → fall back to
CPU-safe profile` rule for exactly this reason — see
[06-training-or-usage-workflows.md](06-training-or-usage-workflows.md) for
where that convention lives in code.

## Predictions break scikit-learn metrics with an unknown target type error

**Symptom:**
```
ValueError: Classification metrics can't handle a mix of binary and unknown targets
```
when calling `sklearn.metrics.accuracy_score(y_test, preds)` right after
`TabFMClassifier.predict(X_test)`.

**Cause (verified):** `TabFMClassifier.predict()` returns an
object-dtype NumPy array (even when the underlying values are plain
integers). Some scikit-learn metric functions call
`sklearn.utils.multiclass.type_of_target()` internally, which classifies a
plain `object`-dtype array of Python ints as `"unknown"` rather than
`"binary"`/`"multiclass"` — even though `y_test` itself (a normal `int64`
array or `pandas.Series`) is correctly classified. The mismatch trips
`accuracy_score`'s internal consistency check.

**Fix:** cast predictions to your label dtype before scoring:

```python
preds = clf.predict(X_test).astype(y_train.to_numpy().dtype)
```

This is already done in
[`examples/01_minimal_classification.py`](../examples/01_minimal_classification.py).

## uv cache is read-only

**Symptom:**
```
Could not create temporary file .../.cache/uv/... Read-only file system
```

**Cause:** the default `uv` cache path isn't writable in your environment
(common in sandboxed/CI environments).

**Fix:**
```bash
export UV_CACHE_DIR=/tmp/uv-cache
```
Every command in this repo's docs assumes this may be necessary — add it
whenever a `uv run`/`uv sync` command fails this way.

## Notebook kernel fails with a socket permission error

**Symptom:**
```
PermissionError: [Errno 1] Operation not permitted
```
during Jupyter kernel startup inside a sandboxed notebook execution.

**Cause:** the sandbox restricts socket/networking operations that a Jupyter
kernel process needs at startup.

**Fix:** run notebook execution (`jupyter nbconvert --execute`,
`scripts/run_strict_e2e.py`) in a regular, non-sandboxed shell.

## Reduce download size

If you only need one task (classification *or* regression), don't download
both:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --task classification
```

Or, without this repo's converter script, using `huggingface_hub` directly:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="google/tabfm-1.0.0-pytorch",
    allow_patterns=["classification/*"],   # skip the other task entirely
)
```

## Interrupted download

A partial Hugging Face download resumes automatically — re-run the same
command rather than deleting anything. If you suspect a corrupted partial
file, clear just that repo's cache:

```bash
python -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir())"
```
and use the reported command to selectively delete that revision.

## Network fetch fails for a demo dataset CSV

Several notebooks under [`problems/`](../problems/) fetch small CSVs from
public GitHub mirrors at run time (e.g., the Telco churn dataset). If your
network blocks these, the affected cell will raise after exhausting its
listed mirrors. There is currently no bundled offline fallback for these
demo datasets — if you're in a restricted network environment, download the
CSV manually from one of the documented mirrors in the notebook's markdown
and place it at the path the cell expects.

## GitHub CLI authentication (maintainers only)

```bash
gh auth login -h github.com
gh auth status   # verify
```
Needed only if you're cutting a release — see
[10-next-steps.md](10-next-steps.md) and [`RELEASE_NOTES.md`](../RELEASE_NOTES.md).

---
**Next:** [09 — FAQ →](09-faq.md)
