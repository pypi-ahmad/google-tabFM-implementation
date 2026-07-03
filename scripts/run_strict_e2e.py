#!/usr/bin/env python3
"""Internal CI/E2E regression runner — not a learning example.

Executes every notebook under problems/ via `jupyter nbconvert --execute`,
retrying with progressively smaller context/row-sampling profiles on
out-of-memory signatures, then validates that each problem's expected
artifacts were produced and writes a run report
(artifacts/strict_e2e_run_report.{csv,md}).

If you're learning TabFM, this is not where to start — see
docs/00-overview.md and the notebooks under notebooks/ and problems/
themselves. This script exists to keep those notebooks honest in CI, not to
teach TabFM usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OOM_PATTERNS = (
    "cuda out of memory",
    "out of memory",
    "memoryerror",
    "std::bad_alloc",
    "resourceexhausted",
    "oom-kill",
    "killed process",
    "process exited with code 137",
)


@dataclass
class OOMProfile:
    name: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    context_candidates: list[int] | None = None


@dataclass
class ProjectConfig:
    name: str
    notebook: Path
    output_name: str
    artifacts_dir: Path
    runtime_meta_path: Path
    env: dict[str, str] = field(default_factory=dict)
    required_artifacts: list[Path] = field(default_factory=list)
    use_context_backoff: bool = True
    context_candidates: list[int] | None = None
    oom_profiles: list[OOMProfile] | None = None


@dataclass
class RunResult:
    name: str
    status: str
    attempts: int
    duration_sec: float
    context_requested: int | None
    context_effective: int | None
    backoff_triggered: bool
    oom_level: int | None
    retry_reason: str
    env_overrides_json: str
    output_notebook: str
    log_file: str
    artifact_check: str
    error: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict end-to-end notebooks.")
    parser.add_argument(
        "--start-from",
        default="",
        help="Start from this project name (inclusive).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated project names to run.",
    )
    return parser.parse_args()


def classify_failure(proc: subprocess.CompletedProcess[str]) -> tuple[str, bool]:
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    if proc.returncode in (137, -9):
        return "oom_retriable", True
    if any(pattern in combined for pattern in OOM_PATTERNS):
        return "oom_retriable", True
    return "deterministic_fail", False


def validate_artifacts(paths: list[Path]) -> tuple[bool, str]:
    missing: list[str] = []
    empty: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        if path.is_file() and path.stat().st_size == 0:
            empty.append(str(path))
    if missing or empty:
        bits: list[str] = []
        if missing:
            bits.append(f"missing={missing}")
        if empty:
            bits.append(f"empty={empty}")
        return False, "; ".join(bits)
    return True, "ok"


def run_nbconvert(
    python_bin: Path,
    notebook: Path,
    output_name: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(python_bin),
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        str(notebook),
        "--output",
        output_name,
        "--output-dir",
        str(notebook.parent),
    ]
    return subprocess.run(cmd, env=env, text=True, capture_output=True)


def stamp_runtime_meta(
    meta_path: Path,
    strict_timestamp: str,
    context_requested: int | None,
    context_effective: int | None,
    attempts: int,
    success: bool,
    oom_level: int | None,
    retry_reason: str,
    env_overrides: dict[str, str],
) -> None:
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    meta.update(
        {
            "strict_e2e": True,
            "strict_e2e_run_timestamp_utc": strict_timestamp,
            "strict_e2e_context_requested": context_requested,
            "strict_e2e_context_effective": context_effective,
            "strict_e2e_backoff_triggered": bool(
                (context_requested is not None and context_effective is not None and context_effective != context_requested)
                or (oom_level is not None and oom_level > 0)
            ),
            "strict_e2e_attempts": attempts,
            "strict_e2e_success": bool(success),
            "strict_e2e_oom_level": oom_level,
            "strict_e2e_retry_reason": retry_reason,
            "strict_e2e_env_overrides": env_overrides,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))


def make_problem2_oom_profiles() -> list[OOMProfile]:
    return [
        OOMProfile(
            name="full_real",
            env_overrides={
                "KKBOX_SAMPLE_TRAIN_ROWS": "0",
                "KKBOX_SAMPLE_EVAL_ROWS": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
            },
            context_candidates=[50_000, 20_000, 10_000, 5_000, 2_500, 1_500],
        ),
        OOMProfile(
            name="backoff_level_1",
            env_overrides={
                "KKBOX_SAMPLE_TRAIN_ROWS": "300000",
                "KKBOX_SAMPLE_EVAL_ROWS": "300000",
                "TABFM_EVAL_MAX_ROWS": "50000",
            },
            context_candidates=[10_000, 5_000, 2_500, 1_500],
        ),
        OOMProfile(
            name="backoff_level_2",
            env_overrides={
                "KKBOX_SAMPLE_TRAIN_ROWS": "180000",
                "KKBOX_SAMPLE_EVAL_ROWS": "180000",
                "TABFM_EVAL_MAX_ROWS": "20000",
            },
            context_candidates=[5_000, 2_500, 1_500],
        ),
        OOMProfile(
            name="backoff_level_3",
            env_overrides={
                "KKBOX_SAMPLE_TRAIN_ROWS": "120000",
                "KKBOX_SAMPLE_EVAL_ROWS": "120000",
                "TABFM_EVAL_MAX_ROWS": "10000",
            },
            context_candidates=[2_500, 1_500, 1_200, 1_000],
        ),
    ]


def build_projects(root: Path, checkpoint: Path) -> list[ProjectConfig]:
    p1_dir = root / "problems" / "problem1_telecom_churn"
    p2_dir = root / "problems" / "problem2_saas_subscription_churn"
    p3_dir = root / "problems" / "problem3_credit_card_transaction_fraud"
    p4_dir = root / "problems" / "problem4_insurance_claim_fraud"
    p5_dir = root / "problems" / "problem5_house_price_prediction"
    p6_dir = root / "problems" / "problem6_ride_fare_estimation"
    p7_dir = root / "problems" / "problem7_loan_default_prediction"
    p8_dir = root / "problems" / "problem8_employee_attrition_prediction"

    return [
        ProjectConfig(
            name="problem1_telecom_churn",
            notebook=p1_dir / "problem1_telecom_churn_tabfm.ipynb",
            output_name="problem1_telecom_churn_tabfm.executed.ipynb",
            artifacts_dir=p1_dir / "artifacts",
            runtime_meta_path=p1_dir / "artifacts" / "telco_churn_runtime_meta.json",
            env={
                "STRICT_E2E": "1",
                "TABFM_FAST_MODE": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p1_dir / "problem1_telecom_churn_tabfm.executed.ipynb",
                p1_dir / "artifacts" / "telco_churn_runtime_meta.json",
                p1_dir / "artifacts" / "telco_churn_model_metrics.csv",
                p1_dir / "artifacts" / "telco_churn_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[1200, 1000, 800, 600],
        ),
        ProjectConfig(
            name="problem2_saas_subscription_churn",
            notebook=p2_dir / "problem2_saas_subscription_churn_kkbox.ipynb",
            output_name="problem2_saas_subscription_churn_kkbox.executed.ipynb",
            artifacts_dir=p2_dir / "artifacts",
            runtime_meta_path=p2_dir / "artifacts" / "problem2_kkbox_runtime_meta.json",
            env={
                "TABFM_FAST_MODE": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
                "KKBOX_MIN_SAMPLE_ROWS": "2000",
            },
            required_artifacts=[
                p2_dir / "problem2_saas_subscription_churn_kkbox.executed.ipynb",
                p2_dir / "artifacts" / "problem2_kkbox_runtime_meta.json",
                p2_dir / "artifacts" / "problem2_kkbox_model_metrics.csv",
                p2_dir / "artifacts" / "problem2_kkbox_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[50_000, 20_000, 10_000, 5_000, 2_500, 1_500],
            oom_profiles=make_problem2_oom_profiles(),
        ),
        ProjectConfig(
            name="problem3_credit_card_transaction_fraud",
            notebook=p3_dir / "problem3_credit_card_transaction_fraud_tabfm.ipynb",
            output_name="problem3_credit_card_transaction_fraud_tabfm.executed.ipynb",
            artifacts_dir=p3_dir / "artifacts",
            runtime_meta_path=p3_dir / "artifacts" / "problem3_fraud_runtime_meta.json",
            env={
                "FRAUD_SAMPLE_TRAIN_ROWS": "0",
                "FRAUD_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p3_dir / "problem3_credit_card_transaction_fraud_tabfm.executed.ipynb",
                p3_dir / "artifacts" / "problem3_fraud_runtime_meta.json",
                p3_dir / "artifacts" / "problem3_fraud_model_metrics.csv",
                p3_dir / "artifacts" / "problem3_fraud_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[50_000, 20_000, 10_000, 5_000, 2_500, 1_500],
        ),
        ProjectConfig(
            name="problem4_insurance_claim_fraud",
            notebook=p4_dir / "problem4_insurance_claim_fraud_tabfm.ipynb",
            output_name="problem4_insurance_claim_fraud_tabfm.executed.ipynb",
            artifacts_dir=p4_dir / "artifacts",
            runtime_meta_path=p4_dir / "artifacts" / "problem4_insurance_runtime_meta.json",
            env={
                "INS_SAMPLE_TRAIN_ROWS": "0",
                "INS_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p4_dir / "problem4_insurance_claim_fraud_tabfm.executed.ipynb",
                p4_dir / "artifacts" / "problem4_insurance_runtime_meta.json",
                p4_dir / "artifacts" / "problem4_insurance_model_metrics.csv",
                p4_dir / "artifacts" / "problem4_insurance_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[5_000, 2_500, 1_500],
        ),
        ProjectConfig(
            name="problem5_house_price_prediction",
            notebook=p5_dir / "problem5_house_price_prediction_tabfm.ipynb",
            output_name="problem5_house_price_prediction_tabfm.executed.ipynb",
            artifacts_dir=p5_dir / "artifacts",
            runtime_meta_path=p5_dir / "artifacts" / "problem5_houseprice_runtime_meta.json",
            env={
                "HP_SAMPLE_TRAIN_ROWS": "0",
                "HP_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p5_dir / "problem5_house_price_prediction_tabfm.executed.ipynb",
                p5_dir / "artifacts" / "problem5_houseprice_runtime_meta.json",
                p5_dir / "artifacts" / "problem5_houseprice_model_metrics.csv",
                p5_dir / "artifacts" / "problem5_houseprice_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[5_000, 2_500, 1_500],
        ),
        ProjectConfig(
            name="problem6_ride_fare_estimation",
            notebook=p6_dir / "problem6_ride_fare_estimation_tabfm.ipynb",
            output_name="problem6_ride_fare_estimation_tabfm.executed.ipynb",
            artifacts_dir=p6_dir / "artifacts",
            runtime_meta_path=p6_dir / "artifacts" / "problem6_ridefare_runtime_meta.json",
            env={
                "RIDE_SAMPLE_TRAIN_ROWS": "0",
                "RIDE_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p6_dir / "problem6_ride_fare_estimation_tabfm.executed.ipynb",
                p6_dir / "artifacts" / "problem6_ridefare_runtime_meta.json",
                p6_dir / "artifacts" / "problem6_ridefare_model_metrics.csv",
                p6_dir / "artifacts" / "problem6_ridefare_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[10_000, 5_000, 2_500, 1_500],
        ),
        ProjectConfig(
            name="problem7_loan_default_prediction",
            notebook=p7_dir / "problem7_loan_default_prediction_tabfm.ipynb",
            output_name="problem7_loan_default_prediction_tabfm.executed.ipynb",
            artifacts_dir=p7_dir / "artifacts",
            runtime_meta_path=p7_dir / "artifacts" / "problem7_loan_runtime_meta.json",
            env={
                "LOAN_SAMPLE_TRAIN_ROWS": "0",
                "LOAN_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p7_dir / "problem7_loan_default_prediction_tabfm.executed.ipynb",
                p7_dir / "artifacts" / "problem7_loan_runtime_meta.json",
                p7_dir / "artifacts" / "problem7_loan_model_metrics.csv",
                p7_dir / "artifacts" / "problem7_loan_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[50_000, 20_000, 10_000, 5_000, 2_500, 1_500],
        ),
        ProjectConfig(
            name="problem8_employee_attrition_prediction",
            notebook=p8_dir / "problem8_employee_attrition_prediction_tabfm.ipynb",
            output_name="problem8_employee_attrition_prediction_tabfm.executed.ipynb",
            artifacts_dir=p8_dir / "artifacts",
            runtime_meta_path=p8_dir / "artifacts" / "problem8_attrition_runtime_meta.json",
            env={
                "ATTRITION_SAMPLE_TRAIN_ROWS": "0",
                "ATTRITION_SAMPLE_EVAL_ROWS": "0",
                "TABFM_FAST_MODE": "0",
                "TABFM_EVAL_MAX_ROWS": "0",
                "TABFM_DEVICE": "auto",
                "TABFM_CHECKPOINT_PATH": str(checkpoint),
            },
            required_artifacts=[
                p8_dir / "problem8_employee_attrition_prediction_tabfm.executed.ipynb",
                p8_dir / "artifacts" / "problem8_attrition_runtime_meta.json",
                p8_dir / "artifacts" / "problem8_attrition_model_metrics.csv",
                p8_dir / "artifacts" / "problem8_attrition_predictions_test.parquet",
            ],
            use_context_backoff=True,
            context_candidates=[5_000, 2_500, 1_500],
        ),
    ]


def filter_projects(
    projects: list[ProjectConfig],
    start_from: str,
    only: str,
) -> list[ProjectConfig]:
    name_to_idx = {cfg.name: idx for idx, cfg in enumerate(projects)}
    selected = projects

    if start_from:
        if start_from not in name_to_idx:
            valid = ", ".join(name_to_idx.keys())
            raise ValueError(f"Unknown --start-from={start_from}. Valid names: {valid}")
        selected = selected[name_to_idx[start_from] :]

    if only:
        requested = [item.strip() for item in only.split(",") if item.strip()]
        unknown = [name for name in requested if name not in name_to_idx]
        if unknown:
            valid = ", ".join(name_to_idx.keys())
            raise ValueError(f"Unknown --only names={unknown}. Valid names: {valid}")
        requested_set = set(requested)
        selected = [cfg for cfg in selected if cfg.name in requested_set]

    if not selected:
        raise ValueError("No projects selected after applying --start-from/--only filters.")
    return selected


def main() -> int:
    args = parse_args()
    root = repo_root()
    python_bin = root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        print(f"Missing python environment: {python_bin}", file=sys.stderr)
        return 2

    checkpoint = (
        root
        / "problems"
        / "problem2_saas_subscription_churn"
        / "data"
        / "models"
        / "google-tabfm-1.0.0-pytorch"
        / "classification"
        / "pytorch_model.bin"
    )
    if not checkpoint.exists():
        print(f"Missing checkpoint: {checkpoint}", file=sys.stderr)
        return 2

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_context_candidates = [
        200_000_000,
        50_000_000,
        20_000_000,
        5_000_000,
        1_000_000,
        500_000,
        200_000,
        100_000,
        50_000,
        20_000,
        10_000,
        5_000,
        2_500,
        1_500,
    ]

    all_projects = build_projects(root=root, checkpoint=checkpoint)
    try:
        projects = filter_projects(
            projects=all_projects,
            start_from=args.start_from.strip(),
            only=args.only.strip(),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results: list[RunResult] = []
    strict_timestamp = now_utc_iso()

    base_env = os.environ.copy()
    base_env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/matplotlib-cache",
            "UV_CACHE_DIR": "/tmp/uv-cache",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for cfg in projects:
        ensure_dir(cfg.artifacts_dir)

        success = False
        start_project = time.time()
        error_msg = ""
        last_log_file = ""
        artifact_check = "not_checked"
        context_effective: int | None = None
        attempts = 0
        retry_reason = "not_started"
        effective_oom_level: int | None = None
        effective_profile_env: dict[str, str] = {}
        deterministic_fail = False

        oom_profiles = cfg.oom_profiles if cfg.oom_profiles else [OOMProfile(name="default")]

        first_profile = oom_profiles[0]
        if cfg.use_context_backoff:
            first_context_candidates = (
                first_profile.context_candidates
                if first_profile.context_candidates
                else (cfg.context_candidates if cfg.context_candidates else default_context_candidates)
            )
            context_requested = first_context_candidates[0]
        else:
            context_requested = None

        for oom_idx, oom_profile in enumerate(oom_profiles):
            if cfg.use_context_backoff:
                context_list = (
                    oom_profile.context_candidates
                    if oom_profile.context_candidates
                    else (cfg.context_candidates if cfg.context_candidates else default_context_candidates)
                )
            else:
                context_list = [None]

            for context_value in context_list:
                attempts += 1
                env = base_env.copy()
                env.update(cfg.env)
                env.update(oom_profile.env_overrides)
                if context_value is not None:
                    env["TABFM_CONTEXT_MAX_ROWS"] = str(context_value)

                run_log = cfg.artifacts_dir / f"strict_e2e_{cfg.name}_{run_timestamp}_attempt{attempts}.log"
                last_log_file = str(run_log)

                print(
                    f"[{cfg.name}] starting attempt={attempts} oom_level={oom_idx} profile={oom_profile.name} context={context_value} notebook={cfg.notebook.name}",
                    flush=True,
                )

                attempt_start = time.time()
                proc = run_nbconvert(
                    python_bin=python_bin,
                    notebook=cfg.notebook,
                    output_name=cfg.output_name,
                    env=env,
                )
                attempt_elapsed = time.time() - attempt_start

                retry_reason, is_oom = classify_failure(proc)
                error_msg = (proc.stderr or proc.stdout or "Notebook execution failed").strip().splitlines()[-1][:800]

                log_payload = [
                    f"project={cfg.name}",
                    f"attempt={attempts}",
                    f"oom_level={oom_idx}",
                    f"oom_profile={oom_profile.name}",
                    f"context={context_value}",
                    f"returncode={proc.returncode}",
                    f"elapsed_sec={attempt_elapsed:.2f}",
                    f"retry_reason={retry_reason}",
                    f"env_overrides={json.dumps(oom_profile.env_overrides, sort_keys=True)}",
                    "\n[STDOUT]\n",
                    proc.stdout,
                    "\n[STDERR]\n",
                    proc.stderr,
                ]
                run_log.write_text("\n".join(log_payload))

                if proc.returncode == 0:
                    output_notebook = cfg.notebook.parent / cfg.output_name
                    required_artifacts = list(dict.fromkeys([output_notebook, *cfg.required_artifacts]))
                    artifacts_ok, artifact_detail = validate_artifacts(required_artifacts)
                    artifact_check = artifact_detail
                    if artifacts_ok:
                        print(
                            f"[{cfg.name}] attempt={attempts} succeeded in {attempt_elapsed:.1f}s (artifact_check=ok)",
                            flush=True,
                        )
                        success = True
                        context_effective = context_value
                        effective_oom_level = oom_idx
                        effective_profile_env = dict(oom_profile.env_overrides)
                        break
                    retry_reason = "deterministic_fail"
                    error_msg = f"Artifact validation failed: {artifact_detail}"
                    deterministic_fail = True
                    print(
                        f"[{cfg.name}] attempt={attempts} failed artifact_check={artifact_detail}",
                        flush=True,
                    )
                    break

                print(
                    f"[{cfg.name}] attempt={attempts} failed rc={proc.returncode} oom_level={oom_idx} context={context_value} retry_reason={retry_reason} error={error_msg}",
                    flush=True,
                )

                effective_oom_level = oom_idx
                effective_profile_env = dict(oom_profile.env_overrides)

                if not is_oom:
                    deterministic_fail = True
                    break

            if success or deterministic_fail:
                break

        duration = time.time() - start_project
        stamp_runtime_meta(
            meta_path=cfg.runtime_meta_path,
            strict_timestamp=strict_timestamp,
            context_requested=context_requested,
            context_effective=context_effective,
            attempts=attempts,
            success=success,
            oom_level=effective_oom_level,
            retry_reason=retry_reason,
            env_overrides=effective_profile_env,
        )

        output_notebook = str(cfg.notebook.parent / cfg.output_name)
        result = RunResult(
            name=cfg.name,
            status="success" if success else "failed",
            attempts=attempts,
            duration_sec=duration,
            context_requested=context_requested,
            context_effective=context_effective,
            backoff_triggered=bool(
                (context_requested is not None and context_effective is not None and context_effective != context_requested)
                or (effective_oom_level is not None and effective_oom_level > 0)
            ),
            oom_level=effective_oom_level,
            retry_reason=retry_reason,
            env_overrides_json=json.dumps(effective_profile_env, sort_keys=True),
            output_notebook=output_notebook,
            log_file=last_log_file,
            artifact_check=artifact_check,
            error=error_msg,
        )
        results.append(result)

        print(
            f"[{cfg.name}] status={result.status} attempts={result.attempts} duration={result.duration_sec:.1f}s context={result.context_effective} oom_level={result.oom_level} retry_reason={result.retry_reason}",
            flush=True,
        )

        if not success:
            break

    report_dir = root / "artifacts"
    ensure_dir(report_dir)
    report_csv = report_dir / "strict_e2e_run_report.csv"
    report_md = report_dir / "strict_e2e_run_report.md"

    with report_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "project",
                "status",
                "attempts",
                "duration_sec",
                "context_requested",
                "context_effective",
                "backoff_triggered",
                "oom_level",
                "retry_reason",
                "env_overrides",
                "artifact_check",
                "output_notebook",
                "log_file",
                "error",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.name,
                    row.status,
                    row.attempts,
                    f"{row.duration_sec:.2f}",
                    row.context_requested,
                    row.context_effective,
                    row.backoff_triggered,
                    row.oom_level,
                    row.retry_reason,
                    row.env_overrides_json,
                    row.artifact_check,
                    row.output_notebook,
                    row.log_file,
                    row.error,
                ]
            )

    lines = [
        "# Strict E2E Run Report",
        "",
        f"- Timestamp (UTC): {strict_timestamp}",
        f"- Projects attempted: {len(results)}",
        "",
        "| Project | Status | Attempts | Duration (s) | Context Req | Context Eff | OOM Level | Retry Reason | Backoff |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.status} | {row.attempts} | {row.duration_sec:.1f} | {row.context_requested} | {row.context_effective} | {row.oom_level} | {row.retry_reason} | {row.backoff_triggered} |"
        )
    lines.append("")
    lines.append("## Logs")
    for row in results:
        lines.append(f"- `{row.name}`: `{row.log_file}`")
    lines.append("")
    lines.append("## Artifact Validation")
    for row in results:
        lines.append(f"- `{row.name}`: {row.artifact_check}")
    lines.append("")
    failures = [row for row in results if row.status != "success"]
    if failures:
        lines.append("## Failure")
        for row in failures:
            lines.append(f"- `{row.name}`: {row.error}")
    else:
        lines.append("All attempted projects completed successfully.")

    report_md.write_text("\n".join(lines))

    if any(row.status != "success" for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
