from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd

from feature_research.walkforward.runner import WalkforwardRunReport


_FOLD_SCORE_BASE_COLS = [
    "fold_id",
    "param_label",
    "raw_objective",
    "smoothed_objective",
    "rank",
    "selected_feature",
]
_FOLD_SCORE_ENHANCED_COLS = [
    "trade_frequency",
    "selected_in_top_k",
]


@dataclass(frozen=True)
class WalkforwardArtifactPaths:
    output_dir: Path
    folds_csv: Path
    fold_scores_csv: Path
    selection_summary_csv: Path
    selected_params_detailed_csv: Path
    oos_metrics_csv: Path
    report_json: Path
    walkforward_stability_png: Path
    fold_timeline_png: Path
    summary_md: Path
    summary_html: Path


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def resolve_walkforward_output_dir(
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
) -> Path:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    return Path(root_dir) / validated_feature_type / validated_module_name / "walkforward"


def _parse_param_label(param_label: str) -> dict[str, str]:
    parts = [part for part in str(param_label).split("|") if part]
    kv_pairs = [part.split("=", 1) for part in parts if "=" in part]
    return {f"param_{key.strip()}": value.strip() for key, value in kv_pairs if key.strip()}


def _to_scalar(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_selected_params_detailed(
    report: WalkforwardRunReport,
    research_context: dict[str, object],
) -> pd.DataFrame:
    selected_cols = [
        "fold_id",
        "param_label",
        "raw_objective",
        "smoothed_objective",
        "rank",
        "trade_frequency",
        "selected_in_top_k",
    ]
    available_cols = [col for col in selected_cols if col in report.fold_scores_df.columns]
    selection_mask = (
        report.fold_scores_df["selected_in_top_k"].astype(bool)
        if "selected_in_top_k" in report.fold_scores_df.columns
        else report.fold_scores_df["selected_feature"].astype(bool)
    )
    selected_rows = report.fold_scores_df.loc[selection_mask, available_cols].copy()
    if selected_rows.empty:
        return pd.DataFrame()

    fold_window = report.folds_df[["fold_id", "train_start", "train_end", "test_start", "test_end"]]
    detailed = selected_rows.merge(fold_window, on="fold_id", how="left")

    parsed_params = pd.DataFrame([_parse_param_label(label) for label in detailed["param_label"]])
    context_cols = pd.DataFrame(
        [{f"context_{key}": _to_scalar(value) for key, value in sorted(research_context.items())}]
    )
    context_frame = pd.concat([context_cols] * len(detailed), ignore_index=True)
    return pd.concat([detailed.reset_index(drop=True), parsed_params, context_frame], axis=1)


def _build_oos_metrics(report: WalkforwardRunReport) -> pd.DataFrame:
    selected = report.fold_scores_df.loc[report.fold_scores_df["selected_feature"].astype(bool)].copy()
    total_folds = int(len(report.folds_df))
    if selected.empty:
        return pd.DataFrame(
            {
                "metric": ["num_folds", "num_selected_rows"],
                "value": [total_folds, 0],
            }
        )

    raw = selected["raw_objective"].astype(float)
    smooth = selected["smoothed_objective"].astype(float)
    chosen_counts = report.selection_summary_df["selected_feature"].value_counts(dropna=False)
    most_selected_feature = str(chosen_counts.index[0]) if not chosen_counts.empty else ""
    most_selected_count = int(chosen_counts.iloc[0]) if not chosen_counts.empty else 0

    metrics: list[tuple[str, object]] = [
        ("num_folds", total_folds),
        ("num_selected_rows", int(len(selected))),
        ("mean_selected_raw_objective", float(raw.mean())),
        ("median_selected_raw_objective", float(raw.median())),
        ("std_selected_raw_objective", float(raw.std(ddof=0))),
        ("min_selected_raw_objective", float(raw.min())),
        ("max_selected_raw_objective", float(raw.max())),
        ("positive_raw_fold_rate", float((raw > 0.0).mean())),
        ("mean_selected_smoothed_objective", float(smooth.mean())),
        ("unique_selected_features", int(report.selection_summary_df["selected_feature"].nunique())),
        ("most_selected_feature", most_selected_feature),
        ("most_selected_feature_count", most_selected_count),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def _frame_to_markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows_"
    subset = frame.head(max_rows).copy()
    headers = [str(col) for col in subset.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in subset.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_markdown(
    path: Path,
    feature_type: str,
    module_name: str,
    report: WalkforwardRunReport,
    oos_metrics_df: pd.DataFrame,
    selected_params_df: pd.DataFrame,
) -> None:
    lines = [
        f"# Walkforward Summary: {feature_type}/{module_name}",
        "",
        "## Overview",
        f"- Folds: {len(report.folds_df)}",
        f"- Scored rows: {len(report.fold_scores_df)}",
        f"- Selection rows: {len(report.selection_summary_df)}",
        "",
        "## Aggregated OOS Metrics",
        _frame_to_markdown_table(oos_metrics_df, max_rows=200),
        "",
        "## Selected Parameters By Fold",
        _frame_to_markdown_table(selected_params_df, max_rows=50),
        "",
        "## Fold Timeline (Tabular)",
        _frame_to_markdown_table(report.folds_df, max_rows=50),
        "",
        "## Notes",
        "- Use `oos_metrics.csv` for aggregate metrics.",
        "- Use `selected_params_detailed.csv` for parameter-level analysis and context fields.",
        "- Use `fold_scores.csv` for full ranking data per fold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_html(
    path: Path,
    feature_type: str,
    module_name: str,
    report: WalkforwardRunReport,
    oos_metrics_df: pd.DataFrame,
    selected_params_df: pd.DataFrame,
) -> None:
    html = "".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>",
            escape(f"Walkforward Summary: {feature_type}/{module_name}"),
            "</title><style>",
            "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}",
            "table{border-collapse:collapse;margin:12px 0;width:100%}",
            "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}",
            "th{background:#f5f5f5}",
            "h1,h2{margin-top:24px}",
            "</style></head><body>",
            f"<h1>{escape(f'Walkforward Summary: {feature_type}/{module_name}')}</h1>",
            "<h2>Overview</h2>",
            "<ul>",
            f"<li>Folds: {len(report.folds_df)}</li>",
            f"<li>Scored rows: {len(report.fold_scores_df)}</li>",
            f"<li>Selection rows: {len(report.selection_summary_df)}</li>",
            "</ul>",
            "<h2>Aggregated OOS Metrics</h2>",
            oos_metrics_df.to_html(index=False, escape=True),
            "<h2>Selected Parameters By Fold</h2>",
            selected_params_df.to_html(index=False, escape=True),
            "<h2>Fold Timeline (Tabular)</h2>",
            report.folds_df.to_html(index=False, escape=True),
            "</body></html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def write_walkforward_artifacts(
    report: WalkforwardRunReport,
    walkforward_stability_figure: Figure,
    fold_timeline_figure: Figure,
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    research_context: dict[str, object] | None = None,
) -> WalkforwardArtifactPaths:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    output_dir = resolve_walkforward_output_dir(
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        root_dir=root_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = WalkforwardArtifactPaths(
        output_dir=output_dir,
        folds_csv=output_dir / "folds.csv",
        fold_scores_csv=output_dir / "fold_scores.csv",
        selection_summary_csv=output_dir / "selection_summary.csv",
        selected_params_detailed_csv=output_dir / "selected_params_detailed.csv",
        oos_metrics_csv=output_dir / "oos_metrics.csv",
        report_json=output_dir / "report.json",
        walkforward_stability_png=output_dir / "walkforward_stability.png",
        fold_timeline_png=output_dir / "fold_timeline.png",
        summary_md=output_dir / "summary.md",
        summary_html=output_dir / "summary.html",
    )

    report.folds_df[
        [
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_samples",
            "test_samples",
        ]
    ].to_csv(paths.folds_csv, index=False, lineterminator="\n")
    fold_score_cols = _FOLD_SCORE_BASE_COLS + [
        col for col in _FOLD_SCORE_ENHANCED_COLS if col in report.fold_scores_df.columns
    ]
    report.fold_scores_df[fold_score_cols].to_csv(
        paths.fold_scores_csv, index=False, lineterminator="\n"
    )
    report.selection_summary_df[
        [
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "top_k_features",
        ]
    ].to_csv(paths.selection_summary_csv, index=False, lineterminator="\n")

    context = research_context or {}
    selected_params_df = _build_selected_params_detailed(report=report, research_context=context)
    selected_params_df.to_csv(paths.selected_params_detailed_csv, index=False, lineterminator="\n")

    oos_metrics_df = _build_oos_metrics(report)
    oos_metrics_df.to_csv(paths.oos_metrics_csv, index=False, lineterminator="\n")

    walkforward_stability_figure.savefig(paths.walkforward_stability_png)
    fold_timeline_figure.savefig(paths.fold_timeline_png)

    _write_summary_markdown(
        path=paths.summary_md,
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        report=report,
        oos_metrics_df=oos_metrics_df,
        selected_params_df=selected_params_df,
    )
    _write_summary_html(
        path=paths.summary_html,
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        report=report,
        oos_metrics_df=oos_metrics_df,
        selected_params_df=selected_params_df,
    )

    report_payload = {
        "feature_type": validated_feature_type,
        "module_name": validated_module_name,
        "output_dir": str(output_dir),
        "artifact_files": {
            "folds_csv": str(paths.folds_csv),
            "fold_scores_csv": str(paths.fold_scores_csv),
            "selection_summary_csv": str(paths.selection_summary_csv),
            "selected_params_detailed_csv": str(paths.selected_params_detailed_csv),
            "oos_metrics_csv": str(paths.oos_metrics_csv),
            "report_json": str(paths.report_json),
            "walkforward_stability_png": str(paths.walkforward_stability_png),
            "fold_timeline_png": str(paths.fold_timeline_png),
            "summary_md": str(paths.summary_md),
            "summary_html": str(paths.summary_html),
        },
        "research_context": {key: _to_scalar(value) for key, value in sorted(context.items())},
        "row_counts": {
            "folds": int(len(report.folds_df)),
            "fold_scores": int(len(report.fold_scores_df)),
            "selection_summary": int(len(report.selection_summary_df)),
            "selected_params_detailed": int(len(selected_params_df)),
            "oos_metrics": int(len(oos_metrics_df)),
        },
    }
    paths.report_json.write_text(
        json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    return paths
