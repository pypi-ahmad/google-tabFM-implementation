from __future__ import annotations

import numpy as np
import pandas as pd

from tabfm_benchmark.benchmark import _run_with_specs
from tabfm_benchmark.types import DatasetSpec


def test_class_cardinality_guard_skips_dataset() -> None:
    def loader() -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"x": np.arange(220)})
        y = pd.Series(np.arange(220) % 11, name="target")
        return X, y

    specs = [DatasetSpec(name="too_many_classes", task_type="classification", loader=loader)]
    df = _run_with_specs(
        dataset_specs=specs,
        device="cpu",
        seed=7,
        n_estimators=32,
        use_ensemble_preset=True,
    )
    row = df.row(0, named=True)
    assert row["status"] == "skipped"
    assert "exceeding max supported 10 classes" in (row["skip_reason"] or "")

