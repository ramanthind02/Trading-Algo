"""Data-only report generation for permutation validation."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from feature_selection.validation.reports import (
    FunnelStatistics,
    PermutationTestSuite,
    ReportBundle,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_placeholder_png(path: Path, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MINIMAL_PNG + json.dumps(metadata, sort_keys=True).encode("utf-8"))
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def plot_null_distribution(report: VectorShuffleReport, output_path: Path) -> Path:
    return _write_placeholder_png(
        output_path,
        {
            "kind": "null_distribution",
            "param_combo": report.param_combo,
            "p_value": report.p_value,
            "critical_value": report.critical_value,
        },
    )


def plot_walkforward_stability(
    report: WalkforwardStabilityReport,
    output_path: Path,
) -> Path:
    return _write_placeholder_png(
        output_path,
        {
            "kind": "walkforward_stability",
            "feature_name": report.feature_name,
            "is_stable": report.is_stable,
            "folds": len(report.fold_results),
        },
    )


def plot_funnel_diagram(stats: FunnelStatistics, output_path: Path) -> Path:
    return _write_placeholder_png(
        output_path,
        {
            "kind": "funnel",
            "total_params": stats.total_params,
            "stage1_pass": stats.stage1_pass,
            "stage2_pass": stats.stage2_pass,
            "stable_params": stats.stable_params,
            "ensemble_candidates": stats.ensemble_candidates,
        },
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows([{key: _jsonable(value) for key, value in row.items()} for row in rows])
    return path


def _build_markdown(suite: PermutationTestSuite) -> str:
    borderline = [
        report.param_combo
        for report in suite.stage2_reports.values()
        if 0.05 <= report.p_value <= 0.15
    ]
    red_flags = []
    if borderline:
        red_flags.append(f"WARNING: Borderline Stage 2 p-values for {', '.join(borderline)}.")
    if not suite.stage3_report.is_stable:
        red_flags.append(
            f"WARNING: Walkforward instability detected ({suite.stage3_report.stability_verdict})."
        )
    if not red_flags:
        red_flags.append("None identified from the current permutation summaries.")

    ensemble_lines = [f"- `{combo}`" for combo in suite.ensemble_candidates] or ["- None"]

    return "\n".join(
        [
            "# Permutation Validation Summary",
            "",
            "## Feature Overview",
            f"- Feature: `{suite.feature_name}`",
            f"- Feature type: `{suite.feature_type}`",
            f"- Summary: {suite.summary}",
            "",
            "## Funnel Statistics",
            f"- Total params: {suite.funnel_stats.total_params}",
            f"- Stage 1 pass: {suite.funnel_stats.stage1_pass}",
            f"- Stage 2 pass: {suite.funnel_stats.stage2_pass}",
            f"- Stable params: {suite.funnel_stats.stable_params}",
            f"- Ensemble candidates: {suite.funnel_stats.ensemble_candidates}",
            f"- Computational savings: {suite.funnel_stats.computational_savings_pct:.1f}%",
            "",
            "## Ensemble Candidates",
            *ensemble_lines,
            "",
            "## Interpretation Guidance",
            "- Review JSON and CSV artifacts for Power BI or external analysis.",
            "- Treat null-distribution image files as compatibility placeholders only.",
            "",
            "## Red Flags",
            *(f"- {flag}" for flag in red_flags),
            "",
            "## Next Steps",
            "- Use Power BI or downstream tabular tooling for research visualizations.",
            "- Keep QuantStats tearsheets and approved HTML report flows in Python.",
        ]
    )


def generate_permutation_reports(
    suite: PermutationTestSuite,
    output_dir: Path,
) -> ReportBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)

    suite_json = _write_json(output_dir / "suite_summary.json", asdict(suite))
    suite_markdown = (output_dir / "suite_summary.md")
    suite_markdown.write_text(_build_markdown(suite), encoding="utf-8")

    stage1_summary = _write_csv(
        output_dir / "stage1_summary.csv",
        [
            {
                "param_combo": report.param_combo,
                "original_metric": report.original_metric,
                "critical_value": report.critical_value,
                "p_value": report.p_value,
                "passed": report.passed,
            }
            for report in suite.stage1_reports.values()
        ],
    )
    stage2_summary = _write_csv(
        output_dir / "stage2_summary.csv",
        [
            {
                "param_combo": report.param_combo,
                "permutation_mode": report.permutation_mode,
                "original_metric": report.original_metric,
                "critical_value": report.critical_value,
                "p_value": report.p_value,
                "passed": report.passed,
                "no_trade_permutations": report.no_trade_permutations,
            }
            for report in suite.stage2_reports.values()
        ],
    )
    stage3_summary = _write_csv(
        output_dir / "stage3_summary.csv",
        [
            {
                "fold_id": fold.fold_id,
                "fold_start": fold.fold_period[0],
                "fold_end": fold.fold_period[1],
                "top_k_params": json.dumps(fold.top_k_params),
                "passed_permutation_overlay": json.dumps(fold.passed_permutation_overlay),
            }
            for fold in suite.stage3_report.fold_results
        ],
    )
    funnel_summary = _write_csv(
        output_dir / "funnel_summary.csv",
        [asdict(suite.funnel_stats)],
    )
    oos_summary = _write_csv(
        output_dir / "oos_summary.csv",
        [
            {
                "param_combo": report.param_combo,
                "vector_passed": report.vector_report.passed,
                "vector_p_value": report.vector_report.p_value,
                "candle_passed": None if report.candle_report is None else report.candle_report.passed,
                "candle_p_value": None if report.candle_report is None else report.candle_report.p_value,
                "passed": report.passed,
            }
            for report in suite.phase3_oos_reports.values()
        ],
    )
    combo_decision_table = _write_csv(
        output_dir / "combo_decision_table.csv",
        [asdict(record) for record in suite.combo_decisions.values()],
    )

    stage1_plots = {
        key: plot_null_distribution(report, output_dir / "stage1" / f"null_dist_{key}.png")
        for key, report in suite.stage1_reports.items()
    }
    stage2_plots = {
        key: plot_null_distribution(report, output_dir / "stage2" / f"null_dist_{key}.png")
        for key, report in suite.stage2_reports.items()
    }
    stage3_plot = plot_walkforward_stability(
        suite.stage3_report,
        output_dir / "stage3" / "walkforward_stability.png",
    )
    funnel_plot = plot_funnel_diagram(
        suite.funnel_stats,
        output_dir / "funnel" / "funnel_diagram.png",
    )

    for key, report in suite.phase3_oos_reports.items():
        plot_null_distribution(report.vector_report, output_dir / "oos" / "vector" / f"null_dist_{key}.png")
        if report.candle_report is not None:
            plot_null_distribution(report.candle_report, output_dir / "oos" / "candle" / f"null_dist_{key}.png")

    return ReportBundle(
        suite_json=suite_json,
        suite_markdown=suite_markdown,
        suite_html=None,
        stage1_plots=stage1_plots,
        stage2_plots=stage2_plots,
        stage3_plot=stage3_plot,
        funnel_plot=funnel_plot,
        stage1_summary=stage1_summary,
        stage2_summary=stage2_summary,
        stage3_summary=stage3_summary,
        funnel_summary=funnel_summary,
        oos_summary=oos_summary,
        combo_decision_table=combo_decision_table,
        timestamp=timestamp,
    )
