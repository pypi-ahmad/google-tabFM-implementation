"""Minimal TabFM classification example.

The smallest possible working use of TabFM: load the pre-trained
classification model, fit it on a small built-in dataset, and predict on a
held-out test split. No training loop, no hyperparameter search — TabFM
uses your training rows as in-context examples at prediction time.

Walkthrough: docs/03-first-run.md
Concepts (in-context learning, zero-shot): docs/04-core-concepts.md

Usage:
    uv run python examples/01_minimal_classification.py

First run downloads TabFM's classification weights from Hugging Face
(several GB, one-time, cached under ~/.cache/huggingface). See
docs/02-installation.md before running this if you haven't already.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabfm import TabFMClassifier
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

# "auto" picks a GPU if one is available AND has enough VRAM (TabFM's ~6 GB
# checkpoint leaves little headroom below 12 GiB), else falls back to CPU.
# Override with: TABFM_DEVICE=cpu uv run python examples/01_minimal_classification.py
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

# Optional: point at a local checkpoint directory instead of downloading.
# See docs/08-troubleshooting.md#reduce-download-size.
CHECKPOINT_PATH = os.environ.get("TABFM_CHECKPOINT_PATH") or None


def main() -> None:
    # 1. Load a small, real, built-in tabular dataset (569 rows, 30 numeric
    #    features, binary target: malignant vs. benign tumor).
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # 2. Hold out a test split so accuracy reflects generalization, not
    #    memorization (see docs/01-prerequisites.md).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Load TabFM's pre-trained classification weights. This is a frozen
    #    model — nothing here is trained on your data.
    print(f"Loading TabFM classification weights (device={DEVICE}) ...")
    try:
        model = tabfm_v1_0_0.load(
            model_type="classification",
            device=DEVICE,
            checkpoint_path=CHECKPOINT_PATH,
        )
    except FileNotFoundError as exc:
        # Common first-run failure: tabfm==1.0.0 downloads safetensors from HF but
        # its PyTorch loader looks for `pytorch_model.bin`. This repo ships a
        # one-command converter; see docs/08-troubleshooting.md.
        print("\nERROR: TabFM weights were downloaded but not found in the format the loader expects.")
        print(f"Underlying error: {exc}\n")
        print("Fix (one-time):")
        print("  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --task classification")
        print("  export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch")
        print("\nThen rerun this example. Full writeup: docs/08-troubleshooting.md#missing-checkpoint-file-on-a-fresh-install")
        raise

    # 4. "Fit" here only prepares label/feature encoders and stores your
    #    training rows as context — there is no gradient descent step.
    clf = TabFMClassifier(model=model, random_state=42)
    clf.fit(X_train, y_train.to_numpy())

    # 5. Predict on the held-out rows. TabFM returns labels as an
    #    object-dtype array mirroring your original label dtype — cast to a
    #    plain numeric array before handing it to strict scikit-learn metric
    #    functions (see docs/08-troubleshooting.md, "Predictions break
    #    scikit-learn metrics with an unknown target type error").
    preds = clf.predict(X_test).astype(y_train.to_numpy().dtype)

    accuracy = accuracy_score(y_test, preds)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test set size: {len(X_test)} rows | Train context size: {len(X_train)} rows")


if __name__ == "__main__":
    main()
