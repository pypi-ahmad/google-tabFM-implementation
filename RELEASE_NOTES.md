# Release Notes

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
