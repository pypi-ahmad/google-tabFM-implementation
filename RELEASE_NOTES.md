# Release Notes

## v2.0.0 - Zero-to-Mastery Learning Path

This is a ground-up educational overhaul, not an incremental patch. The
repository's original benchmark/production code is preserved and improved,
but the project is now a genuine "zero to mastery" TabFM curriculum with a
numbered learning path, runnable minimal-to-practical examples, an
interactive beginner notebook, and a real upstream bug found, verified, and
fixed along the way.

### Added
- **`docs/00-overview.md` → `docs/10-next-steps.md`**: an 11-page, numbered
  learning path — what TabFM is, prerequisites, installation, first run,
  core concepts (in-context learning, zero-shot, architecture, the ensemble
  preset, a full glossary), working with data, usage workflows, evaluation
  methodology, troubleshooting, FAQ, and next steps. Every page defines its
  jargon on first use and cites primary sources for every factual claim.
- **`examples/`**: four runnable, minimal-to-practical scripts
  (`01_minimal_classification.py`, `02_minimal_regression.py`,
  `03_default_vs_ensemble.py`, `04_churn_baseline_comparison.py`), each
  executed end to end in this repo's own reference environment to capture
  real, verified expected output — not illustrative numbers.
- **`notebooks/00_beginner_walkthrough.ipynb`**: an interactive, heavily
  narrated companion notebook covering the same ground as the first-run and
  core-concepts docs, including a live default-vs-ensemble-preset
  comparison with an honest "the fancier option isn't always better"
  result.
- **`scripts/fetch_tabfm_weights.py`**: fixes a verified upstream packaging
  bug in `tabfm==1.0.0` — its PyTorch auto-download path looks for
  `pytorch_model.bin`, but the files published on Hugging Face
  (`google/tabfm-1.0.0-pytorch`) are `model.safetensors`. This script
  downloads the safetensors weights and re-saves them in the expected
  format (verified with `strict=True` state-dict loading — a pure container
  conversion, not a model port). Documented in full in
  `docs/08-troubleshooting.md` and `HANDBOOK.md §14.3`.
- New `safetensors` dependency (required by the fix above).

### Changed
- **`README.md`**: fully rewritten as a beginner-first entry point — clear
  audience, learning path, quickstart with verified expected output,
  repository structure, real results table, and a properly contextualized
  references section (each source has a stated reason for inclusion).
- **`HANDBOOK.md`**: repositioned as the operational/maintainer manual,
  cross-linked to the new `docs/` learning path; corrected the exact weights
  license name (**"TabFM Non-Commercial License v1.0"**, not just
  "non-commercial"); clarified that `TABFM_*` environment variables are this
  repository's own convention, not part of the `tabfm` package API.
- **`notebooks/tabfm_quickstart_benchmark.ipynb`**: no longer hard-fails
  when CUDA is unavailable — now uses this repo's `<12 GiB VRAM → CPU
  fallback` convention like every other notebook.
- **`problems/problem1_telecom_churn`**: added a header cell clarifying its
  relationship to `notebooks/tabfm_telco_churn_production.ipynb` (same
  workflow; this copy adds CI/E2E automation hooks) so learners know which
  one to read first.

### Fixed
- Removed 9 committed `*.executed.ipynb` files that baked in the previous
  maintainer's absolute local filesystem path and stale timestamps;
  `artifacts/*.csv`/`*.json` remain as the path-free evidence of each run.
  `*.executed.ipynb` is now gitignored.
- Fixed a real, reproducible `ValueError` when scoring `TabFMClassifier`
  predictions with strict scikit-learn metrics (object-dtype array
  misclassified as an "unknown" target type) — documented and worked around
  in every example.

### Verification
Commands executed:

```bash
./scripts/validate_repo_hygiene.sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_benchmark.py --help
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_strict_e2e.py --help
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --help
```

Observed status:
- Hygiene checks passed.
- `7 passed, 1 warning` for test suite.
- All four `examples/` scripts executed successfully end to end with real
  captured output (see `docs/03-first-run.md`, `docs/07-evaluation.md`, and
  each script's own docstring).
- All internal documentation cross-links (including heading anchors)
  verified to resolve correctly.

## v1.0.3 - Release Publisher Hardening

### Highlights
- Replaced `softprops/action-gh-release@v2` in `.github/workflows/release.yml` with direct `gh` CLI publishing to eliminate Node 20 deprecation exposure.
- Added idempotent release behavior:
  - if the tag release already exists, edit notes/title and re-upload `HANDBOOK.pdf` with `--clobber`;
  - if it does not exist, create the release and attach `HANDBOOK.pdf`.
- Kept full release gate parity with CI (hygiene checks, tests, CLI contract checks) before publish.

### Verification
Commands executed:

```bash
./scripts/validate_repo_hygiene.sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Observed status:
- Hygiene checks passed.
- `7 passed, 1 warning` for test suite.

## v1.0.2 - Workflow Compatibility Patch

### Highlights
- Fixed GitHub Action resolution by pinning `astral-sh/setup-uv` to `v8.2.0`.
- Upgraded `actions/checkout` to `v6` and `actions/setup-python` to `v6` to remove Node 20 deprecation risk on hosted runners.
- Revalidated CI gates after workflow updates.

### Verification
Commands executed:

```bash
./scripts/validate_repo_hygiene.sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Observed status:
- Hygiene checks passed.
- `7 passed, 1 warning` for test suite.

## v1.0.1 - CI and Release Automation Hardening

### Highlights
- Added CI workflow: `.github/workflows/ci.yml`
  - Runs on pull requests to `main` and pushes to `main`.
  - Enforces repository hygiene plus deterministic validation checks.
- Added tag-driven release workflow: `.github/workflows/release.yml`
  - Runs on semantic tags (`v*.*.*`).
  - Revalidates checks, publishes GitHub release using `RELEASE_NOTES.md`, and uploads `HANDBOOK.pdf`.
- Added repository hygiene script: `scripts/validate_repo_hygiene.sh`
  - Fails if tracked files exceed 90 MiB.
  - Fails if tracked files exist under disallowed raw-data/model paths.
- Updated README and handbook with CI/release operating contract.

### Verification
Commands executed:

```bash
./scripts/validate_repo_hygiene.sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_benchmark.py --help
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_strict_e2e.py --help
```

Observed status:
- Hygiene checks passed.
- `7 passed, 1 warning` for test suite.

### Notes
- Release publication is now tag-driven; maintainers should update this file before pushing the next `vX.Y.Z` tag.

## v1.0.0 - Public Documentation and Release Baseline

### Highlights
- Rewrote `README.md` for public GitHub presentation with full project narrative:
  - overview, problem statement, objectives,
  - architecture and workflow,
  - setup/usage,
  - results, limitations, and future improvements,
  - official references and source links.
- Added comprehensive zero-to-mastery project manual in `HANDBOOK.md`.
- Generated `HANDBOOK.pdf` from handbook markdown.
- Added publish-safe ignore rules for local-only heavyweight assets:
  - `data/raw/`
  - `problems/**/data/raw/`
  - `problems/**/data/models/`

### Verification
Commands executed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_benchmark.py --help
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_strict_e2e.py --help
```

Observed test status:
- `7 passed, 1 warning`

### Notes
- Existing executed notebooks and problem artifacts were used as the primary source of real outputs.
- GitHub CLI authentication must be valid (`gh auth login -h github.com`) before push/release creation.
- TabFM released weights remain subject to non-commercial licensing terms.
