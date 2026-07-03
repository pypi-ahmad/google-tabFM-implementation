# 02 — Installation

> **You are here:** [Learning path](../README.md#learning-path) → **02 Installation**
> **Previous:** [01 — Prerequisites](01-prerequisites.md) · **Next:** [03 — First Run](03-first-run.md)

## 1. Clone and install

```bash
git clone https://github.com/pypi-ahmad/google-tabFM-implementation.git
cd google-tabFM-implementation

uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

What each command does:

- `uv venv --python 3.11` — creates an isolated Python 3.11 virtual
  environment in `.venv/`, so this project's dependencies never collide with
  anything else on your machine.
- `source .venv/bin/activate` — makes that environment the active `python`
  for your current shell.
- `uv sync --extra dev` — installs every dependency pinned in
  [`uv.lock`](../uv.lock), including the `dev` extra (`pytest`, `nbconvert`,
  `ipykernel`) needed to run tests and notebooks. This is also the step that
  installs `tabfm[pytorch]` itself — TabFM is a regular PyPI package, not a
  separate SDK you configure elsewhere.

### Restricted / sandboxed environments

If `uv` fails with a read-only cache error, point its cache somewhere
writable first:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
```

Every command example in this repo's docs assumes this is set if you hit
that error. See [08-troubleshooting.md](08-troubleshooting.md#uv-cache-is-read-only)
for the full symptom/fix writeup.

## 2. Verify the install

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

**Expected output:** all tests pass (this repo's package test suite covers
dataset loaders, metric computation, the result schema, and the class-count
guard — see [`tests/`](../tests/)). If you see failures here, stop and work
through [08-troubleshooting.md](08-troubleshooting.md) before moving on —
everything downstream assumes a clean install.

## 3. Disk and network requirements — read this before your first run

TabFM ships **code** via PyPI, but the **pretrained weights** are a separate
download from Hugging Face, fetched automatically the first time you call
`tabfm_v1_0_0_pytorch.load(...)` in your code. This is the single biggest
practical surprise for newcomers, so it gets its own callout:

> **The classification checkpoint alone is about 6 GB on disk** (measured
> directly in this project's own reference environment). The regression
> checkpoint is downloaded separately and is comparable in size. If your
> workflow uses both classification and regression (as several examples in
> this repo do), budget **at least 15–20 GB free disk space** and a stable
> internet connection for the first run. After the first download, weights
> are cached locally (by default under `~/.cache/huggingface/`) and reused —
> every subsequent run is offline and fast to load.

Practical implications:

- **Run [03-first-run.md](03-first-run.md) somewhere with a decent
  connection** — don't attempt the first download over a slow tethered
  connection.
- **The download is resumable.** If it's interrupted, re-running the same
  script resumes rather than restarting (this is a built-in property of the
  `huggingface_hub` download client).
- **You can shrink the first download** if you only need one task
  (classification *or* regression). See
  [08-troubleshooting.md](08-troubleshooting.md#reduce-download-size) for the
  exact snippet.
- **CI and automated environments** should pre-populate the cache once and
  reuse it, rather than downloading fresh on every run — see
  [`scripts/run_strict_e2e.py`](../scripts/run_strict_e2e.py) for how this
  repo's own automation handles a `TABFM_CHECKPOINT_PATH` override (a
  convention specific to this repo, not part of the `tabfm` package itself —
  more on that distinction in
  [06-training-or-usage-workflows.md](06-training-or-usage-workflows.md)).

## 4. GPU vs. CPU

TabFM runs on CPU or GPU. Nothing in the install step is GPU-specific — the
`tabfm[pytorch]` extra installs a CPU-capable PyTorch by default. If you have
an NVIDIA GPU and want to use it, make sure `torch.cuda.is_available()`
returns `True` in your environment (this depends on your system's CUDA
driver, which is outside this repo's scope to install). Every example script
in [`examples/`](../examples/) auto-detects the device and falls back to CPU
if no GPU is present — you do not need to configure anything to get started.

## 5. What "installed correctly" looks like

Run this one-liner to confirm the `tabfm` package itself is importable and
reports its version:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import tabfm; print(tabfm.__version__)"
```

**Expected output:** `1.0.0`

If this fails, you have an installation problem, not a TabFM usage problem —
go back to step 1. If it succeeds, continue to
[03-first-run.md](03-first-run.md), where you'll load the model and make
your first prediction.

---
**Next:** [03 — First Run →](03-first-run.md)
