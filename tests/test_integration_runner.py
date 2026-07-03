from __future__ import annotations

import numpy as np
import pandas as pd

from tabfm_benchmark.benchmark import _run_with_specs, save_results
from tabfm_benchmark.types import DatasetSpec


class _StubClassifier:
    def fit(self, X, y):
        y_series = pd.Series(y)
        self._majority = y_series.mode().iloc[0]
        return self

    def predict(self, X):
        return np.repeat(self._majority, len(X))


class _StubRegressor:
    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def test_run_with_specs_and_save_results(monkeypatch, tmp_path) -> None:
    def cls_loader() -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 1, 0, 0, 1, 0]})
        y = pd.Series([0, 1, 0, 1, 1, 0], name="target")
        return X, y

    def reg_loader() -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [3, 2, 1, 0, 1, 2]})
        y = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5, 2.0], name="target")
        return X, y

    specs = [
        DatasetSpec(name="cls_small", task_type="classification", loader=cls_loader),
        DatasetSpec(name="reg_small", task_type="regression", loader=reg_loader),
    ]

    def _fake_load_model(task_type, device, checkpoint_path):
        return object()

    def _fake_build_estimator(task_type, model, n_estimators, use_ensemble_preset):
        if task_type == "classification":
            return _StubClassifier()
        return _StubRegressor()

    monkeypatch.setattr(
        "tabfm_benchmark.benchmark._load_model_for_task", _fake_load_model
    )
    monkeypatch.setattr(
        "tabfm_benchmark.benchmark._build_estimator", _fake_build_estimator
    )

    df = _run_with_specs(
        dataset_specs=specs,
        device="cpu",
        seed=42,
        checkpoint_path="",
        n_estimators=4,
        use_ensemble_preset=True,
    )
    assert df.height == 2
    assert set(df["status"].to_list()) == {"ok"}

    parquet_path = tmp_path / "results.parquet"
    csv_path = tmp_path / "summary.csv"
    save_results(df, parquet_path)
    save_results(df, csv_path)
    assert parquet_path.exists()
    assert csv_path.exists()
