from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    """Benchmark dataset descriptor."""

    name: str
    task_type: TaskType
    loader: Callable[[], tuple[pd.DataFrame, pd.Series]]
    row_cap: int | None = None

