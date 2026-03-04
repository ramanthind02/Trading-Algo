from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


def resolve_objective_metric_name(perm_cfg: object, wf_config: object | None) -> str:
    perm_objective = getattr(perm_cfg, "objective_metric", None)
    builtin = getattr(perm_objective, "builtin", None) if perm_objective else None
    if builtin is not None:
        return str(builtin)
    return str(getattr(wf_config, "objective_metric_name", "sharpe")) if wf_config else "sharpe"


def apply_objective_metric(config: object, objective_metric_name: str) -> object:
    wf_config = getattr(config, "walkforward", None)
    if wf_config is None:
        return config
    return replace(config, walkforward=replace(wf_config, objective_metric_name=objective_metric_name))


def compute_significance(
    original_metric: float,
    null_metrics: np.ndarray,
    nreps: int,
    alpha: float,
) -> tuple[float, float, bool]:
    n_ge = int((null_metrics >= original_metric).sum()) if null_metrics.size else 0
    p_value = float(1 + n_ge) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0)) if null_metrics.size else 0.0
    passed = bool(original_metric > critical_value)
    return p_value, critical_value, passed


def build_report_payload(
    *,
    effective_mode: str,
    nreps: int,
    random_seed: int | None,
    alpha: float,
    original_metric: float,
    p_value: float,
    critical_value: float,
    passed: bool,
    per_ticker_reports: dict[str, dict[str, float]] | None = None,
    skipped_tickers: list[str] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": effective_mode,
        "nreps": nreps,
        "random_seed": random_seed,
        "alpha": alpha,
        "original_metric": original_metric,
        "p_value": p_value,
        "critical_value": critical_value,
        "passed": passed,
    }
    if per_ticker_reports:
        report["aggregation"] = "mean"
        report["per_ticker"] = per_ticker_reports
        report["skipped_tickers"] = skipped_tickers or []
    return report


def write_permutation_outputs(
    *,
    out_dir: Path,
    report_filename: str,
    report_payload: dict[str, Any],
    null_metrics: np.ndarray,
    per_ticker_nulls: dict[str, np.ndarray] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / report_filename
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    np.save(out_dir / "null_distribution.npy", null_metrics)
    if per_ticker_nulls:
        for label, values in per_ticker_nulls.items():
            safe_label = label.replace("/", "_").replace(" ", "_")
            np.save(out_dir / f"null_distribution_{safe_label}.npy", values)
    return report_path
