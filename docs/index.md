# Google TabFM — Zero to Mastery

This is the documentation site version of the learning path. If you're reading
on GitHub, the main entry point is still the repository `README.md`.

## Quickstart

```bash
git clone https://github.com/pypi-ahmad/google-tabFM-implementation.git
cd google-tabFM-implementation

uv venv --python 3.11
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/uv-cache
uv sync --extra dev --frozen

# One-time fix for a known upstream mismatch in `tabfm==1.0.0` (PyTorch):
uv run python scripts/fetch_tabfm_weights.py --task classification
export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch

uv run python examples/01_minimal_classification.py
```

If you hit an error, go to [08 — Troubleshooting](08-troubleshooting.md).

## Learning path

Work through these in order the first time:

| # | Doc | You'll learn |
|---|---|---|
| 00 | [Overview](00-overview.md) | What TabFM is, the problem it solves, what it is *not* |
| 01 | [Prerequisites](01-prerequisites.md) | Knowledge and environment you need before starting |
| 02 | [Installation](02-installation.md) | Install this repo and TabFM; disk/network requirements |
| 03 | [First Run](03-first-run.md) | Your first prediction, line by line, with expected output |
| 04 | [Core Concepts](04-core-concepts.md) | In-context learning, zero-shot, architecture, ensemble preset, glossary |
| 05 | [Working with Data](05-working-with-data.md) | Input contract, documented limits, dataset-prep checklist |
| 06 | [Usage Workflows](06-training-or-usage-workflows.md) | Classification & regression workflows; repo conventions vs. library API |
| 07 | [Evaluation](07-evaluation.md) | Metrics defined, baseline comparison, interpreting results |
| 08 | [Troubleshooting](08-troubleshooting.md) | Failure modes reproduced in this repo, with fixes |
| 09 | [FAQ](09-faq.md) | Licensing, fine-tuning, paper status, and common questions |
| 10 | [Next Steps](10-next-steps.md) | Where to go once you finish the core path |
| 11 | [Datasets & Licenses](11-datasets-and-licenses.md) | Dataset sources/terms + what this repo does and does not redistribute |

