from __future__ import annotations

from tabfm_benchmark.datasets import (
    default_dataset_specs,
    load_breast_cancer_dataset,
    load_diabetes_dataset,
    load_digits_dataset,
    load_iris_dataset,
    load_wine_dataset,
)


def test_local_dataset_loaders_return_non_empty_frames() -> None:
    loaders = [
        load_iris_dataset,
        load_wine_dataset,
        load_breast_cancer_dataset,
        load_digits_dataset,
        load_diabetes_dataset,
    ]
    for loader in loaders:
        X, y = loader()
        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)
        assert X.shape[1] > 0


def test_default_portfolio_contains_expected_names() -> None:
    specs = default_dataset_specs()
    names = {spec.name for spec in specs}
    assert "covtype" in names
    assert "california_housing" in names
    assert len(specs) >= 7

