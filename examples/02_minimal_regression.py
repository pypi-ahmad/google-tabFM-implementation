"""Minimal TabFM regression example.

Same idea as 01_minimal_classification.py, applied to a numeric target
instead of a category. TabFM's regressor standardizes the target internally
and inverse-transforms predictions back to the original scale for you.

Walkthrough: docs/03-first-run.md
Concepts (in-context learning, zero-shot): docs/04-core-concepts.md

Usage:
    uv run python examples/02_minimal_regression.py
"""

from __future__ import annotations

import os

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

# "auto" picks a GPU if one is available AND has enough VRAM (TabFM's ~6 GB
# checkpoint leaves little headroom below 12 GiB), else falls back to CPU.
DEVICE = os.environ.get("TABFM_DEVICE", "auto")
if DEVICE == "auto":
    import torch

    DEVICE = "cpu"
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        DEVICE = "cuda" if gpu_mem_gb >= 12.0 else "cpu"
        if DEVICE == "cpu":
            print(
                f"Detected GPU with {gpu_mem_gb:.1f} GiB VRAM (<12 GiB) — "
                "using CPU for a stable run. See "
                "docs/08-troubleshooting.md#cuda-out-of-memory."
            )

CHECKPOINT_PATH = os.environ.get("TABFM_CHECKPOINT_PATH") or None

# California housing has 20,640 rows. TabFM's memory use scales with BOTH
# the training context size and the number of rows scored per prediction
# call, so this example caps both to stay comfortable on an 8 GB laptop GPU
# (or CPU). Raise these if you have more VRAM — see docs/05-working-with-data.md.
ROW_CAP = int(os.environ.get("TABFM_CONTEXT_MAX_ROWS", "300"))
EVAL_ROW_CAP = int(os.environ.get("TABFM_EVAL_MAX_ROWS", "300"))


def main() -> None:
    # 1. Load a real, built-in regression dataset: predict median house
    #    value (in $100,000s) for California districts from 8 numeric
    #    features (income, house age, rooms, location, ...).
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_train) > ROW_CAP:
        X_train = X_train.sample(n=ROW_CAP, random_state=42)
        y_train = y_train.loc[X_train.index]
    if len(X_test) > EVAL_ROW_CAP:
        X_test = X_test.sample(n=EVAL_ROW_CAP, random_state=42)
        y_test = y_test.loc[X_test.index]

    print(f"Loading TabFM regression weights (device={DEVICE}) ...")
    try:
        model = tabfm_v1_0_0.load(
            model_type="regression",
            device=DEVICE,
            checkpoint_path=CHECKPOINT_PATH,
        )
    except FileNotFoundError as exc:
        print("\nERROR: TabFM weights were downloaded but not found in the format the loader expects.")
        print(f"Underlying error: {exc}\n")
        print("Fix (one-time):")
        print("  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --task regression")
        print("  export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch")
        print("\nThen rerun this example. Full writeup: docs/08-troubleshooting.md#missing-checkpoint-file-on-a-fresh-install")
        raise

    # n_estimators=4 keeps this "minimal" example light on memory; the
    # default of 32 is more accurate but can exceed 8 GB of GPU memory at
    # this context size (see docs/08-troubleshooting.md#cuda-out-of-memory).
    reg = TabFMRegressor(model=model, n_estimators=4, random_state=42)
    reg.fit(X_train, y_train.to_numpy())

    preds = reg.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.4f} (target unit: $100,000s)")
    print(f"Test R^2: {r2:.4f}")
    print(f"Test set size: {len(X_test)} rows | Train context size: {len(X_train)} rows")


if __name__ == "__main__":
    main()
