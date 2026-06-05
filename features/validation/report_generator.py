"""Data-only report generation for permutation validation."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features.validation.reports import (
    FunnelStatistics,
    PermutationTestSuite,
    ReportBundle,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _styled_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfc")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    return fig, ax


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
    fig, ax = _styled_axes(f"Null Distribution: {report.param_combo}")
    null_values = np.asarray(report.null_distribution, dtype=float)
    n_bins = max(10, min(30, int(np.sqrt(max(len(null_values), 1))) * 2))
    ax.hist(
        null_values,
        bins=n_bins,
        color="#93c5fd",
        edgecolor="#1d4ed8",
        alpha=0.85,
    )
    ax.axvline(
        report.critical_value,
        color="#d97706",
        linestyle="--",
        linewidth=2,
        label=f"critical={report.critical_value:.3f}",
    )
    ax.axvline(
        report.original_metric,
        color="#15803d",
        linewidth=2,
        label=f"observed={report.original_metric:.3f}",
    )
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Frequency")
    verdict = "PASS" if report.passed else "FAIL"
    ax.text(
        0.98,
        0.95,
        f"{verdict}\np={report.p_value:.3f}\nalpha={report.alpha:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    ax.legend(loc="upper left")
    return _save_figure(fig, output_path)


def plot_walkforward_stability(
    report: WalkforwardStabilityReport,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(ncols=2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#ffffff")

    param_counts = (
        np.array([param for fold in report.fold_results for param in fold.top_k_params], dtype=object)
        if report.fold_results
        else np.array([], dtype=object)
    )
    if param_counts.size > 0:
        params, counts = np.unique(param_counts, return_counts=True)
        order = np.argsort(counts)
        axes[0].barh(
            params[order],
            counts[order],
            color="#60a5fa",
            edgecolor="#1d4ed8",
        )
        axes[0].set_xlabel("Selections across folds")
    else:
        axes[0].text(0.5, 0.5, "No fold selections", ha="center", va="center")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    axes[0].set_title("Top-K Selection Frequency", fontsize=11, fontweight="bold")
    axes[0].set_facecolor("#fbfbfc")
    axes[0].grid(alpha=0.2, linestyle="--", axis="x")

    fold_labels = [str(fold.fold_id) for fold in report.fold_results]
    pass_share = [
        (
            float(sum(fold.passed_permutation_overlay)) / len(fold.passed_permutation_overlay)
            if fold.passed_permutation_overlay
            else 0.0
        )
        for fold in report.fold_results
    ]
    overlap_rate = float(report.consistency_metrics.get("overlap_rate", 0.0))
    axes[1].bar(
        fold_labels,
        pass_share,
        color="#34d399",
        edgecolor="#047857",
        label="Permutation pass share",
    )
    axes[1].axhline(
        overlap_rate,
        color="#dc2626",
        linestyle="--",
        linewidth=2,
        label=f"overlap_rate={overlap_rate:.2f}",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Share / overlap")
    axes[1].set_title(
        f"Walkforward Verdict: {report.stability_verdict}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_facecolor("#fbfbfc")
    axes[1].grid(alpha=0.2, linestyle="--", axis="y")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_funnel_diagram(stats: FunnelStatistics, output_path: Path) -> Path:
    fig, ax = _styled_axes("Permutation Funnel")
    stages = [
        "Total",
        "Stage 1",
        "Stage 2",
        "Stable",
        "Candidates",
    ]
    values = [
        stats.total_params,
        stats.stage1_pass,
        stats.stage2_pass,
        stats.stable_params,
        stats.ensemble_candidates,
    ]
    colors = ["#94a3b8", "#60a5fa", "#22c55e", "#f59e0b", "#a855f7"]
    bars = ax.bar(stages, values, color=colors, edgecolor="#334155")
    ax.set_ylabel("Count")
    ax.text(
        0.98,
        0.95,
        f"Savings={stats.computational_savings_pct:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    _ = [
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.05,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        for bar, value in zip(bars, values, strict=False)
    ]
    return _save_figure(fig, output_path)


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
            "- Review JSON and CSV artifacts together with the generated Matplotlib PNGs.",
            "- Use CSV exports as the canonical inputs for any custom Matplotlib views.",
            "",
            "## Red Flags",
            *(f"- {flag}" for flag in red_flags),
            "",
            "## Next Steps",
            "- Use the exported CSVs as inputs to the repo's Matplotlib visualization flow.",
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
