from __future__ import annotations

import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)

from tabfm_benchmark.types import DatasetSpec


def load_iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_iris(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_wine_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_wine(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_breast_cancer_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_digits_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_digits(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_covtype_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = fetch_covtype(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_diabetes_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def load_california_housing_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return X, y.rename("target")


def default_dataset_specs() -> list[DatasetSpec]:
    """Returns the default multi-dataset benchmark portfolio."""
    return [
        DatasetSpec(
            name="iris",
            task_type="classification",
            loader=load_iris_dataset,
        ),
        DatasetSpec(
            name="wine",
            task_type="classification",
            loader=load_wine_dataset,
        ),
        DatasetSpec(
            name="breast_cancer",
            task_type="classification",
            loader=load_breast_cancer_dataset,
        ),
        DatasetSpec(
            name="digits",
            task_type="classification",
            loader=load_digits_dataset,
        ),
        DatasetSpec(
            name="covtype",
            task_type="classification",
            loader=load_covtype_dataset,
            row_cap=120_000,
        ),
        DatasetSpec(
            name="diabetes",
            task_type="regression",
            loader=load_diabetes_dataset,
        ),
        DatasetSpec(
            name="california_housing",
            task_type="regression",
            loader=load_california_housing_dataset,
        ),
    ]

