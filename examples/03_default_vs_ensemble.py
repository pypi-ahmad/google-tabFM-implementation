"""Compare TabFM's default preset against its ensemble preset.

Builds on 01_minimal_classification.py: same dataset, same split, but this
time fits both `TabFMClassifier(model=model)` (default) and
`TabFMClassifier.ensemble(model=model)` (ensemble preset) so you can see the
accuracy/runtime tradeoff directly. Read docs/04-core-concepts.md#5 for what
the ensemble preset changes internally (feature crosses, SVD features, NNLS
blending, calibration) before running this.

Usage:
    uv run python examples/03_default_vs_ensemble.py
"""

from __future__ import annotations

import os
import time

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabfm import TabFMClassifier
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

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

# The ensemble preset does more work per member (feature crosses, SVD
# features, NNLS blending) — a smaller n_estimators keeps this comparison
# fast without changing what it demonstrates. See
# docs/04-core-concepts.md, "The ensemble preset," for the full breakdown.
N_ESTIMATORS = int(os.environ.get("TABFM_N_ESTIMATORS", "16"))


def run_variant(name: str, clf, X_train, y_train, X_test, y_test) -> None:
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test).astype(y_train.dtype)
    proba = clf.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    print(f"{name:>18} | accuracy={accuracy:.4f} | roc_auc={roc_auc:.4f} | {elapsed:.1f}s")


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    print(f"Loading TabFM classification weights (device={DEVICE}) ...")
    model = tabfm_v1_0_0.load(
        model_type="classification", device=DEVICE, checkpoint_path=CHECKPOINT_PATH
    )

    print(f"{'variant':>18} | {'metric':<28} | time")
    run_variant(
        "default",
        TabFMClassifier(model=model, n_estimators=N_ESTIMATORS, random_state=42),
        X_train, y_train_arr, X_test, y_test_arr,
    )
    run_variant(
        "ensemble preset",
        TabFMClassifier.ensemble(model=model, n_estimators=N_ESTIMATORS, random_state=42),
        X_train, y_train_arr, X_test, y_test_arr,
    )


if __name__ == "__main__":
    main()
