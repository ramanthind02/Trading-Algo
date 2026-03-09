"""
T017: Report Generation and Visualization for Permutation Testing.

Generates plots, markdown, and JSON exports from PermutationTestSuite results.
"""
from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from feature_selection.validation.reports import (
    FunnelStatistics,
    OutOfSamplePermutationReport,
    PermutationTestSuite,
    PipelinePermutationReport,
    ReportBundle,
    VectorShuffleReport,
    WalkforwardStabilityReport,
)
from feature_selection.validation.stability_analysis import compute_jaccard_overlap


def plot_null_distribution(
    report: Union[VectorShuffleReport, PipelinePermutationReport],
    output_path: Path,
) -> Path:
    """Plot null distribution histogram with original metric and critical value overlay.

    Args:
        report: VectorShuffleReport or PipelinePermutationReport.
        output_path: Path to save the figure (.png).

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    null_dist = report.null_distribution

    ax.hist(null_dist, bins=30, color='steelblue', alpha=0.7, edgecolor='white',
            label='Null distribution')

    # Determine color based on pass/fail
    orig_color = 'green' if report.passed else 'red'
    ax.axvline(report.original_metric, color=orig_color, linewidth=2,
               linestyle='-', label=f'Original metric ({report.original_metric:.4f})')
    ax.axvline(report.critical_value, color='orange', linewidth=2,
               linestyle='--', label=f'Critical value alpha={report.alpha} ({report.critical_value:.4f})')

    verdict = 'PASS' if report.passed else 'FAIL'
    ax.set_title(
        f'Null Distribution — {report.param_combo}\n'
        f'p-value={report.p_value:.4f}  |  {verdict}',
        fontsize=12,
    )
    ax.set_xlabel('Metric value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_walkforward_stability(
    stability_report: WalkforwardStabilityReport,
    output_path: Path,
) -> Path:
    """Multi-panel walkforward stability plot.

    Panel 1: Per-fold top-K heatmap (params x folds, shaded if in top-K).
    Panel 2: Parameter landscape — smoothed objective per fold (bar chart).
    Panel 3: Stability timeline — overlap_rate evolution across consecutive folds.

    Args:
        stability_report: WalkforwardStabilityReport from Stage 3.
        output_path: Path to save the figure.

    Returns:
        Path to the saved figure.
    """
    fold_results = stability_report.fold_results
    if not fold_results:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No fold results', ha='center', va='center')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return output_path

    # Collect all param names across folds
    all_params = sorted({
        param
        for fr in fold_results
        for param in fr.smoothed_objectives
    })
    fold_ids = [fr.fold_id for fr in fold_results]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(all_params)), 12))

    # --- Panel 1: Top-K heatmap ---
    ax1 = axes[0]
    heatmap_data = np.zeros((len(all_params), len(fold_results)))
    for col, fr in enumerate(fold_results):
        for row, param in enumerate(all_params):
            if param in fr.top_k_params:
                heatmap_data[row, col] = 1.0

    im = ax1.imshow(heatmap_data, aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax1.set_xticks(range(len(fold_ids)))
    ax1.set_xticklabels(fold_ids, rotation=30, fontsize=8)
    ax1.set_yticks(range(len(all_params)))
    ax1.set_yticklabels(all_params, fontsize=8)
    ax1.set_title(f'Top-K Selection Heatmap (top_k={stability_report.top_k})', fontsize=11)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Parameter')
    plt.colorbar(im, ax=ax1, fraction=0.02)

    # --- Panel 2: Smoothed objectives per fold (last fold as reference) ---
    ax2 = axes[1]
    last_fold = fold_results[-1]
    param_objs = [last_fold.smoothed_objectives.get(p, 0.0) for p in all_params]
    bar_colors = ['green' if p in last_fold.top_k_params else 'steelblue' for p in all_params]
    ax2.bar(range(len(all_params)), param_objs, color=bar_colors, edgecolor='white', alpha=0.8)
    ax2.set_xticks(range(len(all_params)))
    ax2.set_xticklabels(all_params, rotation=45, fontsize=8)
    ax2.set_title(f'Smoothed Objectives (last fold: {last_fold.fold_id})\nGreen = top-K', fontsize=11)
    ax2.set_ylabel('Smoothed objective')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: Stability timeline ---
    ax3 = axes[2]
    if len(fold_results) > 1:
        overlap_vals = []
        pair_labels = []
        for i in range(len(fold_results) - 1):
            a = set(fold_results[i].top_k_params)
            b = set(fold_results[i + 1].top_k_params)
            ov = compute_jaccard_overlap(a, b)
            overlap_vals.append(ov)
            pair_labels.append(f'{fold_results[i].fold_id}->{fold_results[i+1].fold_id}')

        colors = ['green' if v >= 0.5 else 'red' for v in overlap_vals]
        ax3.bar(range(len(overlap_vals)), overlap_vals, color=colors, edgecolor='white', alpha=0.8)
        ax3.axhline(0.5, color='orange', linewidth=1.5, linestyle='--', label='Stability threshold (0.5)')
        ax3.set_xticks(range(len(pair_labels)))
        ax3.set_xticklabels(pair_labels, rotation=30, fontsize=8)
        ax3.set_ylim(0, 1.1)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Need >= 2 folds for stability timeline',
                 ha='center', va='center', transform=ax3.transAxes)

    overall_rate = stability_report.consistency_metrics.get('overlap_rate', 0.0)
    verdict_color = 'green' if stability_report.is_stable else 'red'
    ax3.set_title(
        f'Stability Timeline — overlap_rate={overall_rate:.2f}  '
        f'[{"STABLE" if stability_report.is_stable else "UNSTABLE"}]',
        fontsize=11, color=verdict_color,
    )
    ax3.set_ylabel('Top-K overlap rate')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f'Walkforward Stability: {stability_report.feature_name}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_funnel_diagram(
    funnel_stats: FunnelStatistics,
    output_path: Path,
) -> Path:
    """Bar chart showing param counts at each validation stage.

    Args:
        funnel_stats: FunnelStatistics from T016.
        output_path: Path to save the figure.

    Returns:
        Path to the saved figure.
    """
    stages = ['Total', 'Stage 1\nPassers', 'Stage 2\nPassers', 'Stable\nParams', 'Ensemble\nCandidates']
    counts = [
        funnel_stats.total_params,
        funnel_stats.stage1_pass,
        funnel_stats.stage2_pass,
        funnel_stats.stable_params,
        funnel_stats.ensemble_candidates,
    ]
    colors = ['steelblue', '#4CAF50', '#8BC34A', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stages, counts, color=colors, edgecolor='white', alpha=0.85)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    ax.set_ylabel('Number of parameter combinations')
    ax.set_title(
        f'Validation Funnel — Computational savings: {funnel_stats.computational_savings_pct:.1f}%',
        fontsize=12,
    )
    ax.set_ylim(0, funnel_stats.total_params * 1.2 + 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _suite_to_json_dict(suite: PermutationTestSuite) -> dict:
    """Serialise PermutationTestSuite to a JSON-compatible dict."""

    def _report_to_dict(r: Union[VectorShuffleReport, PipelinePermutationReport]) -> dict:
        base = {
            'param_combo': r.param_combo,
            'original_metric': r.original_metric,
            'p_value': r.p_value,
            'critical_value': r.critical_value,
            'passed': r.passed,
            'alpha': r.alpha,
            'nreps': r.nreps,
            'null_distribution_mean': float(r.null_distribution.mean()),
            'null_distribution_std': float(r.null_distribution.std()),
        }
        if isinstance(r, PipelinePermutationReport):
            base['feature_type'] = r.feature_type
            base['permutation_mode'] = r.permutation_mode
            base['no_trade_permutations'] = r.no_trade_permutations
        return base

    def _oos_report_to_dict(report: OutOfSamplePermutationReport) -> dict:
        vector_report = report.vector_report
        candle_report = report.candle_report
        return {
            'param_combo': report.param_combo,
            'passed': report.passed,
            'vector_report': _report_to_dict(vector_report),
            'candle_report': _report_to_dict(candle_report) if candle_report is not None else None,
        }

    stage3 = suite.stage3_report
    return {
        'feature_name': suite.feature_name,
        'feature_type': suite.feature_type,
        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        'stage1_reports': {k: _report_to_dict(v) for k, v in suite.stage1_reports.items()},
        'stage2_reports': {k: _report_to_dict(v) for k, v in suite.stage2_reports.items()},
        'stage3': {
            'feature_name': stage3.feature_name,
            'is_stable': stage3.is_stable,
            'stability_verdict': stage3.stability_verdict,
            'consistency_metrics': stage3.consistency_metrics,
            'n_folds': len(stage3.fold_results),
            'top_k': stage3.top_k,
        },
        'funnel_stats': {
            'total_params': suite.funnel_stats.total_params,
            'stage1_pass': suite.funnel_stats.stage1_pass,
            'stage2_pass': suite.funnel_stats.stage2_pass,
            'stable_params': suite.funnel_stats.stable_params,
            'ensemble_candidates': suite.funnel_stats.ensemble_candidates,
            'computational_savings_pct': suite.funnel_stats.computational_savings_pct,
        },
        'ensemble_candidates': suite.ensemble_candidates,
        'phase3_oos_reports': {
            param_combo: _oos_report_to_dict(report)
            for param_combo, report in suite.phase3_oos_reports.items()
        },
        'combo_decisions': {
            param_combo: {
                'param_combo': decision.param_combo,
                'stage1_passed': decision.stage1_passed,
                'stage2_passed': decision.stage2_passed,
                'walkforward_stable': decision.walkforward_stable,
                'oos_passed': decision.oos_passed,
                'final_status': decision.final_status,
            }
            for param_combo, decision in suite.combo_decisions.items()
        },
    }


def _build_markdown(suite: PermutationTestSuite) -> str:
    """Build researcher-readable markdown summary."""
    lines = [
        f"# Permutation Test Report: {suite.feature_name}",
        "",
        "## Feature Overview",
        f"- Feature: `{suite.feature_name}`",
        f"- Type: `{suite.feature_type}`",
        f"- Parameters tested: {suite.funnel_stats.total_params}",
        "",
        "## Funnel Statistics",
        f"| Stage | Params Tested | Passers |",
        f"|-------|--------------|---------|",
        f"| Stage 1 (Vector Shuffle) | {suite.funnel_stats.total_params} | {suite.funnel_stats.stage1_pass} |",
        f"| Stage 2 (Pipeline Permutation) | {suite.funnel_stats.stage1_pass} | {suite.funnel_stats.stage2_pass} |",
        f"| Stage 3 (Walkforward Stability) | removed | — |",
        f"",
        f"**Computational savings:** {suite.funnel_stats.computational_savings_pct:.1f}%",
        "",
        "## Ensemble Candidates",
    ]

    if suite.ensemble_candidates:
        for cand in suite.ensemble_candidates:
            lines.append(f"- `{cand}`")
    else:
        lines.append("_No ensemble candidates identified._")

    lines += [
        "",
        "## Stability Analysis",
        f"- Stage 3 (walkforward permutation) has been removed. Verdict: **{suite.stage3_report.stability_verdict}**",
        "",
        "## Interpretation Guidance",
        "- **p-value < alpha**: Feature metric is unlikely under the null of no predictive relationship.",
        "",
        "## Red Flags",
    ]

    # Red flags: unstable feature
    if not suite.stage3_report.is_stable:
        lines.append("WARNING: **Feature shows temporal instability** — top-K params vary across folds.")

    # Red flags: borderline p-values in Stage 2
    borderline = [
        k for k, r in suite.stage2_reports.items()
        if 0.05 <= r.p_value <= 0.15
    ]
    if borderline:
        lines.append(
            f"WARNING: **Borderline p-values in Stage 2** for: {', '.join(borderline)} "
            f"(consider increasing nreps or using alpha=0.05)"
        )

    if not borderline and suite.stage3_report.is_stable:
        lines.append("No significant red flags detected.")

    lines += [
        "",
        "## Next Steps",
        "1. Review null distribution plots in `stage1/` and `stage2/` directories.",
        "2. Inspect walkforward stability plot in `stage3/walkforward_stability.png`.",
        "3. Select 2-3 ensemble members from the candidate list above.",
        "4. Proceed to out-of-sample validation after ensemble formation.",
    ]

    return '\n'.join(lines)


def generate_permutation_reports(
    suite: PermutationTestSuite,
    output_dir: Path,
    format: Literal['json', 'markdown', 'html'] = 'markdown',
) -> ReportBundle:
    """Generate all plots and text reports for a PermutationTestSuite.

    Directory structure created:
        output_dir/
        ├── suite_summary.md
        ├── suite_summary.json
        ├── stage1/null_dist_{param}.png  (one per param)
        ├── stage2/null_dist_{param}.png  (one per stage2 param)
        ├── stage3/walkforward_stability.png
        └── funnel/funnel_diagram.png

    Args:
        suite: PermutationTestSuite from T016.
        output_dir: Root directory for output artifacts.
        format: Primary text format ('json', 'markdown', or 'html').

    Returns:
        ReportBundle with paths to all generated artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Text reports ---
    md_path = output_dir / 'suite_summary.md'
    md_path.write_text(_build_markdown(suite))

    json_data = _suite_to_json_dict(suite)
    json_path = output_dir / 'suite_summary.json'
    json_path.write_text(json.dumps(json_data, indent=2))

    # --- Stage 1 null distribution plots ---
    stage1_plots: Dict[str, Path] = {}
    stage1_dir = output_dir / 'stage1'
    stage1_dir.mkdir(exist_ok=True)
    for param_combo, report in suite.stage1_reports.items():
        plot_path = stage1_dir / f'null_dist_{param_combo}.png'
        plot_null_distribution(report, plot_path)
        stage1_plots[param_combo] = plot_path

    # --- Stage 2 null distribution plots ---
    stage2_plots: Dict[str, Path] = {}
    stage2_dir = output_dir / 'stage2'
    stage2_dir.mkdir(exist_ok=True)
    for param_combo, report in suite.stage2_reports.items():
        plot_path = stage2_dir / f'null_dist_{param_combo}.png'
        plot_null_distribution(report, plot_path)
        stage2_plots[param_combo] = plot_path

    # --- Stage 3 stability plot ---
    stage3_dir = output_dir / 'stage3'
    stage3_dir.mkdir(exist_ok=True)
    stage3_plot = stage3_dir / 'walkforward_stability.png'
    plot_walkforward_stability(suite.stage3_report, stage3_plot)

    # --- OOS null distribution plots ---
    oos_vector_dir = output_dir / 'oos' / 'vector'
    oos_vector_dir.mkdir(parents=True, exist_ok=True)
    oos_candle_dir = output_dir / 'oos' / 'candle'
    oos_candle_dir.mkdir(parents=True, exist_ok=True)
    for param_combo, oos_report in suite.phase3_oos_reports.items():
        vector_plot_path = oos_vector_dir / f'null_dist_{param_combo}.png'
        plot_null_distribution(oos_report.vector_report, vector_plot_path)

        if oos_report.candle_report is not None:
            candle_plot_path = oos_candle_dir / f'null_dist_{param_combo}.png'
            plot_null_distribution(oos_report.candle_report, candle_plot_path)

    # --- Combo decision table CSV ---
    combo_csv_path = output_dir / 'combo_decision_table.csv'
    with combo_csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'param_combo',
                'stage1_passed',
                'stage2_passed',
                'walkforward_stable',
                'oos_passed',
                'final_status',
            ],
        )
        writer.writeheader()
        for param_combo, decision in sorted(suite.combo_decisions.items()):
            writer.writerow(
                {
                    'param_combo': decision.param_combo or param_combo,
                    'stage1_passed': decision.stage1_passed,
                    'stage2_passed': decision.stage2_passed,
                    'walkforward_stable': decision.walkforward_stable,
                    'oos_passed': decision.oos_passed,
                    'final_status': decision.final_status,
                }
            )

    # --- Funnel diagram ---
    funnel_dir = output_dir / 'funnel'
    funnel_dir.mkdir(exist_ok=True)
    funnel_plot = funnel_dir / 'funnel_diagram.png'
    plot_funnel_diagram(suite.funnel_stats, funnel_plot)

    return ReportBundle(
        suite_json=json_path,
        suite_markdown=md_path,
        suite_html=None,
        stage1_plots=stage1_plots,
        stage2_plots=stage2_plots,
        stage3_plot=stage3_plot,
        funnel_plot=funnel_plot,
        timestamp=datetime.now(tz=timezone.utc),
    )
