from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import pandas as pd

from utils.evaluation.walkforward.runner import WalkforwardRunReport


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def resolve_walkforward_output_dir(
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    output_subdir: str = "walkforward",
) -> Path:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    return Path(root_dir) / validated_feature_type / validated_module_name / output_subdir


def _to_scalar(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_param_label(param_label: str) -> dict[str, str]:
    parts = [part for part in str(param_label).split("|") if part]
    kv_pairs = [part.split("=", 1) for part in parts if "=" in part]
    return {f"param_{key.strip()}": value.strip() for key, value in kv_pairs if key.strip()}


@dataclass(frozen=True)
class WalkforwardArtifactPaths:
    output_dir: Path
    tearsheets_dir: Path
    report_json: Path


def _build_selected_params_detailed(
    report: WalkforwardRunReport,
    research_context: dict[str, object],
) -> pd.DataFrame:
    selected_cols = [
        "fold_id",
        "param_label",
        "raw_objective",
        "oos_objective",
        "smoothed_objective",
        "rank",
        "trade_frequency",
        "selected_in_top_k",
        "selected_long_bin",
    ]
    available_cols = [col for col in selected_cols if col in report.fold_scores_df.columns]
    # Include rows selected as top-k (selected_in_top_k) or as single rank-1 (selected_feature)
    sel_top_k = (
        report.fold_scores_df["selected_in_top_k"].astype(bool)
        if "selected_in_top_k" in report.fold_scores_df.columns
        else False
    )
    sel_feature = report.fold_scores_df["selected_feature"].astype(bool)
    selection_mask = sel_top_k | sel_feature if isinstance(sel_top_k, pd.Series) else sel_feature
    selected_rows = report.fold_scores_df.loc[selection_mask, available_cols].copy()
    if selected_rows.empty:
        return pd.DataFrame()

    fold_window = report.folds_df[["fold_id", "train_start", "train_end", "test_start", "test_end"]]
    detailed = selected_rows.merge(fold_window, on="fold_id", how="left")
    objective_metric_name = getattr(report, "objective_metric_name", "")
    if objective_metric_name:
        detailed = detailed.assign(objective_metric_name=objective_metric_name)

    parsed_params = pd.DataFrame([_parse_param_label(label) for label in detailed["param_label"]])
    context_cols = pd.DataFrame(
        [{f"context_{key}": _to_scalar(value) for key, value in sorted(research_context.items())}]
    )
    context_frame = pd.concat([context_cols] * len(detailed), ignore_index=True)
    return pd.concat([detailed.reset_index(drop=True), parsed_params, context_frame], axis=1)


def write_walkforward_artifacts(
    report: WalkforwardRunReport,
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    research_context: dict[str, object] | None = None,
    output_subdir: str = "walkforward",
) -> WalkforwardArtifactPaths:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    output_dir = resolve_walkforward_output_dir(
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        root_dir=root_dir,
        output_subdir=output_subdir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tearsheets_dir = output_dir / "tearsheets"

    context = dict(research_context or {})
    if not report.folds_df.empty and "test_end" in report.folds_df.columns:
        last_test_end = report.folds_df["test_end"].max()
        context["last_fold_test_end"] = str(pd.Timestamp(last_test_end).date())
    agg_ret = getattr(report, "aggregate_oos_returns", None)
    if agg_ret is not None and hasattr(agg_ret, "index") and len(agg_ret.index) > 0:
        context["aggregate_returns_last_date"] = str(pd.Timestamp(agg_ret.index.max()).date())

    tearsheet_files: list[str] = []
    if tearsheets_dir.exists():
        tearsheet_files = sorted(
            str(p.relative_to(tearsheets_dir)) for p in tearsheets_dir.rglob("*.html")
        )

    report_payload = {
        "feature_type": validated_feature_type,
        "module_name": validated_module_name,
        "output_dir": str(output_dir),
        "tearsheets_dir": str(tearsheets_dir),
        "tearsheet_files": tearsheet_files,
        "objective_metric_name": getattr(report, "objective_metric_name", ""),
        "timeframe": report.timeframe.name,
        "research_context": {key: _to_scalar(value) for key, value in sorted(context.items())},
    }

    report_json_path = output_dir / "report.json"
    report_json_path.write_text(
        json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    return WalkforwardArtifactPaths(
        output_dir=output_dir,
        tearsheets_dir=tearsheets_dir,
        report_json=report_json_path,
    )
