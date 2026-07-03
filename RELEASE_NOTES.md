# Release Notes

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
