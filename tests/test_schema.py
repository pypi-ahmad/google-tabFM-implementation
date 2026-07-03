from __future__ import annotations

from tabfm_benchmark.benchmark import RESULT_SCHEMA, _to_result_frame


def test_result_schema_is_stable_for_empty_frame() -> None:
    df = _to_result_frame([])
    assert df.columns == list(RESULT_SCHEMA.keys())

