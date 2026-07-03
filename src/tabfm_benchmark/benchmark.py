from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
import torch

from tabfm import TabFMClassifier, TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

from tabfm_benchmark.datasets import default_dataset_specs
from tabfm_benchmark.types import DatasetSpec, TaskType

MAX_SUPPORTED_CLASSES = 10

RESULT_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.Utf8,
    "task_type": pl.Utf8,
    "status": pl.Utf8,
    "skip_reason": pl.Utf8,
    "seed": pl.Int64,
    "device": pl.Utf8,
    "n_estimators": pl.Int64,
    "use_ensemble_preset": pl.Boolean,
    "n_rows": pl.Int64,
    "n_features": pl.Int64,
    "n_train": pl.Int64,
    "n_test": pl.Int64,
    "fit_seconds": pl.Float64,
    "predict_seconds": pl.Float64,
    "accuracy": pl.Float64,
    "f1_macro": pl.Float64,
    "rmse": pl.Float64,
    "mae": pl.Float64,
    "r2": pl.Float64,
}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Computes classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Computes regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False."
        )


def _cap_rows(
    X: pd.DataFrame, y: pd.Series, task_type: TaskType, row_cap: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= row_cap:
        return X, y
    if task_type == "classification":
        X_capped, _, y_capped, _ = train_test_split(
            X,
            y,
            train_size=row_cap,
            stratify=y,
            random_state=seed,
        )
        return X_capped, y_capped
    sampled_idx = y.sample(n=row_cap, random_state=seed).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def _split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if task_type == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )


@lru_cache(maxsize=4)
def _load_model_for_task(task_type: TaskType, device: str):
    model_type = "classification" if task_type == "classification" else "regression"
    return tabfm_v1_0_0.load(model_type=model_type, device=device)


def _build_estimator(
    task_type: TaskType,
    model,
    n_estimators: int,
    use_ensemble_preset: bool,
):
    if task_type == "classification":
        if use_ensemble_preset:
            return TabFMClassifier.ensemble(model=model, n_estimators=n_estimators)
        return TabFMClassifier(model=model, n_estimators=n_estimators)
    if use_ensemble_preset:
        return TabFMRegressor.ensemble(model=model, n_estimators=n_estimators)
    return TabFMRegressor(model=model, n_estimators=n_estimators)


def _empty_record(
    dataset: str,
    task_type: TaskType,
    seed: int,
    device: str,
    n_estimators: int,
    use_ensemble_preset: bool,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "task_type": task_type,
        "status": "error",
        "skip_reason": None,
        "seed": seed,
        "device": device,
        "n_estimators": n_estimators,
        "use_ensemble_preset": use_ensemble_preset,
        "n_rows": None,
        "n_features": None,
        "n_train": None,
        "n_test": None,
        "fit_seconds": None,
        "predict_seconds": None,
        "accuracy": None,
        "f1_macro": None,
        "rmse": None,
        "mae": None,
        "r2": None,
    }


def _to_result_frame(records: list[dict[str, object]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=RESULT_SCHEMA)

    df = pl.DataFrame(records)
    for col, dtype in RESULT_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    df = df.select(
        [pl.col(col).cast(dtype, strict=False).alias(col) for col, dtype in RESULT_SCHEMA.items()]
    )
    return df


def _run_single_dataset(
    spec: DatasetSpec,
    seed: int,
    device: str,
    n_estimators: int,
    use_ensemble_preset: bool,
) -> dict[str, object]:
    record = _empty_record(
        dataset=spec.name,
        task_type=spec.task_type,
        seed=seed,
        device=device,
        n_estimators=n_estimators,
        use_ensemble_preset=use_ensemble_preset,
    )

    try:
        X, y = spec.loader()
        if spec.row_cap is not None:
            X, y = _cap_rows(
                X=X, y=y, task_type=spec.task_type, row_cap=spec.row_cap, seed=seed
            )

        X_train, X_test, y_train, y_test = _split_dataset(
            X=X, y=y, task_type=spec.task_type, seed=seed
        )
        record["n_rows"] = int(len(X))
        record["n_features"] = int(X.shape[1])
        record["n_train"] = int(len(X_train))
        record["n_test"] = int(len(X_test))

        if spec.task_type == "classification":
            n_classes = int(pd.Series(y_train).nunique())
            if n_classes > MAX_SUPPORTED_CLASSES:
                record["status"] = "skipped"
                record["skip_reason"] = (
                    f"Dataset has {n_classes} classes, exceeding max supported "
                    f"{MAX_SUPPORTED_CLASSES} classes."
                )
                return record

        model = _load_model_for_task(spec.task_type, device)
        estimator = _build_estimator(
            task_type=spec.task_type,
            model=model,
            n_estimators=n_estimators,
            use_ensemble_preset=use_ensemble_preset,
        )

        fit_start = time.perf_counter()
        estimator.fit(X_train, y_train.to_numpy())
        fit_end = time.perf_counter()

        pred_start = time.perf_counter()
        preds = estimator.predict(X_test)
        pred_end = time.perf_counter()

        record["fit_seconds"] = fit_end - fit_start
        record["predict_seconds"] = pred_end - pred_start

        if spec.task_type == "classification":
            metrics = compute_classification_metrics(
                y_true=y_test.to_numpy(), y_pred=np.asarray(preds)
            )
            record.update(metrics)
        else:
            metrics = compute_regression_metrics(
                y_true=y_test.to_numpy(), y_pred=np.asarray(preds, dtype=np.float64)
            )
            record.update(metrics)

        record["status"] = "ok"
        return record
    except Exception as exc:  # pragma: no cover - exercised in integration paths
        record["status"] = "error"
        record["skip_reason"] = str(exc)
        return record


def _run_with_specs(
    dataset_specs: list[DatasetSpec],
    device: str,
    seed: int,
    n_estimators: int,
    use_ensemble_preset: bool,
) -> pl.DataFrame:
    _validate_device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    records: list[dict[str, object]] = []
    for spec in dataset_specs:
        logger.info("Running dataset: {} ({})", spec.name, spec.task_type)
        records.append(
            _run_single_dataset(
                spec=spec,
                seed=seed,
                device=device,
                n_estimators=n_estimators,
                use_ensemble_preset=use_ensemble_preset,
            )
        )
    return _to_result_frame(records)


def run_benchmark(
    device: str = "cuda",
    seed: int = 42,
    n_estimators: int = 32,
    use_ensemble_preset: bool = True,
) -> pl.DataFrame:
    """Runs the default multi-dataset TabFM benchmark and returns results."""
    return _run_with_specs(
        dataset_specs=default_dataset_specs(),
        device=device,
        seed=seed,
        n_estimators=n_estimators,
        use_ensemble_preset=use_ensemble_preset,
    )


def save_results(df: pl.DataFrame, path: str | Path) -> None:
    """Saves result DataFrame to parquet or csv."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df.write_parquet(output_path)
        return
    if output_path.suffix == ".csv":
        df.write_csv(output_path)
        return
    raise ValueError(f"Unsupported output extension: {output_path.suffix}")


def render_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Renders per-task aggregated benchmark summary."""
    if df.is_empty():
        return df

    return (
        df.filter(pl.col("status") == "ok")
        .group_by("task_type")
        .agg(
            pl.len().alias("dataset_count"),
            pl.col("fit_seconds").mean().alias("mean_fit_seconds"),
            pl.col("predict_seconds").mean().alias("mean_predict_seconds"),
            pl.col("accuracy").mean().alias("mean_accuracy"),
            pl.col("f1_macro").mean().alias("mean_f1_macro"),
            pl.col("rmse").mean().alias("mean_rmse"),
            pl.col("mae").mean().alias("mean_mae"),
            pl.col("r2").mean().alias("mean_r2"),
        )
        .sort("task_type")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabFM multi-dataset benchmark.")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=32,
        help="Number of ensemble estimators",
    )
    parser.add_argument(
        "--no-ensemble-preset",
        action="store_true",
        help="Disable TabFM ensemble preset and use plain estimator config",
    )
    parser.add_argument(
        "--output",
        default="artifacts/tabfm_benchmark_results.parquet",
        help="Output path (.parquet or .csv)",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/tabfm_benchmark_summary.csv",
        help="Summary output path (.parquet or .csv)",
    )
    args = parser.parse_args()

    df = run_benchmark(
        device=args.device,
        seed=args.seed,
        n_estimators=args.n_estimators,
        use_ensemble_preset=not args.no_ensemble_preset,
    )
    summary_df = render_summary(df)

    save_results(df, args.output)
    save_results(summary_df, args.summary_output)
    logger.info("Wrote benchmark results to {}", args.output)
    logger.info("Wrote benchmark summary to {}", args.summary_output)


if __name__ == "__main__":
    main()

