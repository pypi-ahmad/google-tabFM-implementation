from __future__ import annotations

import numpy as np

from tabfm_benchmark.benchmark import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_compute_classification_metrics() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_compute_regression_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0

