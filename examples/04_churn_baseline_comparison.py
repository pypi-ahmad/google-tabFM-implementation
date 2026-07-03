"""Practical example: TabFM vs. XGBoost on a real churn dataset.

Downloads the public IBM Telco customer-churn dataset, does minimal
leakage-safe cleaning, and compares TabFM's default preset against an
XGBoost baseline on the same test split — the evaluation pattern described
in docs/07-evaluation.md. This is a trimmed, readable version of the full
production workflow in notebooks/tabfm_telco_churn_production.ipynb and
problems/problem1_telecom_churn/ (which add cost-sensitive threshold
policies and multiple TabFM variants on top of this same core comparison).

Usage:
    uv run python examples/04_churn_baseline_comparison.py
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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

# The full dataset has ~5,600 training rows after the split below. TabFM's
# memory/runtime scale with context size (docs/05-working-with-data.md), so
# this example caps both training context and evaluation rows to stay fast
# and safe on modest hardware. Raise these if your hardware allows it.
CONTEXT_MAX_ROWS = int(os.environ.get("TABFM_CONTEXT_MAX_ROWS", "1200"))
EVAL_MAX_ROWS = int(os.environ.get("TABFM_EVAL_MAX_ROWS", "800"))

# Mirrors of the same public dataset — see docs/08-troubleshooting.md if all
# three are unreachable from your network.
TELCO_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def download_dataset(output_path: Path, urls: list[str]) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(
                request, timeout=45, context=ssl.create_default_context()
            ) as response:
                output_path.write_bytes(response.read())
            return output_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"All dataset mirrors failed: {urls}") from last_error


def load_and_clean() -> tuple[pd.DataFrame, pd.Series]:
    path = download_dataset(DATA_PATH, TELCO_URLS)
    df = pd.read_csv(path)

    # TotalCharges is stored as text with some blank values for brand-new
    # customers; coerce to numeric and impute with the median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])  # customerID is an identifier, not a feature
    return X, y


def main() -> None:
    X, y = load_and_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cap both models to the SAME subset — a fair comparison needs identical
    # training/evaluation data. This is purely a speed/memory concession for
    # a "practical example" script; problems/problem1_telecom_churn runs the
    # equivalent comparison on the full dataset.
    if len(X_train) > CONTEXT_MAX_ROWS:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=CONTEXT_MAX_ROWS, random_state=42, stratify=y_train
        )
    if len(X_test) > EVAL_MAX_ROWS:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=EVAL_MAX_ROWS, random_state=42, stratify=y_test
        )

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Churn rate: {y.mean():.3f}")

    # --- XGBoost baseline: handles categorical text columns natively when
    # enable_categorical=True and dtype is set to "category". ---
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    for col in X_train_xgb.select_dtypes(include=["object", "str"]).columns:
        X_train_xgb[col] = X_train_xgb[col].astype("category")
        X_test_xgb[col] = X_test_xgb[col].astype(
            pd.CategoricalDtype(categories=X_train_xgb[col].cat.categories)
        )

    baseline = XGBClassifier(random_state=42, enable_categorical=True)
    baseline.fit(X_train_xgb, y_train)
    baseline_proba = baseline.predict_proba(X_test_xgb)[:, 1]

    # --- TabFM default preset ---
    print(f"Loading TabFM classification weights (device={DEVICE}) ...")
    try:
        model = tabfm_v1_0_0.load(
            model_type="classification", device=DEVICE, checkpoint_path=CHECKPOINT_PATH
        )
    except FileNotFoundError as exc:
        print("\nERROR: TabFM weights were downloaded but not found in the format the loader expects.")
        print(f"Underlying error: {exc}\n")
        print("Fix (one-time):")
        print("  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_tabfm_weights.py --task classification")
        print("  export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch")
        print("\nThen rerun this example. Full writeup: docs/08-troubleshooting.md#missing-checkpoint-file-on-a-fresh-install")
        raise
    clf = TabFMClassifier(model=model, random_state=42)
    clf.fit(X_train, y_train.to_numpy())
    tabfm_proba = clf.predict_proba(X_test)[:, 1]

    print()
    print(f"{'model':>18} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    for name, proba in [("xgboost_baseline", baseline_proba), ("tabfm_default", tabfm_proba)]:
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        print(f"{name:>18} | {roc_auc:>8.4f} | {pr_auc:>8.4f}")

    print(
        "\nSee docs/07-evaluation.md for what these metrics mean and why "
        "PR-AUC matters more than accuracy on an imbalanced target like churn."
    )


if __name__ == "__main__":
    main()
