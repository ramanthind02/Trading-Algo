"""Matplotlib reports for validation-zone robustness tests."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantfoundry_core.robustness.validation import ValidationRobustnessReport

ROLLING_Z_MEAN_WINDOW = 60


@dataclass(frozen=True)
class HoldoutRobustnessPlotLabels:
    """Axis and title labels for IS-vs-holdout robustness charts."""

    is_sharpe: str = "IS Sharpe"
    holdout_sharpe: str = "Val Sharpe"
    sharpe_title: str = "IS vs Validation Sharpe"
    holdout_bar: str = "Validation bar"
    z_title: str = "Validation return z-scores (IS μ, σ)"
    equity_title: str = "Equity curve confidence bands"
    is_metric: str = "IS metric"
    holdout_metric: str = "Validation metric"
    rank_title: str = "Parameter grid rank correlation"


DEFAULT_VALIDATION_PLOT_LABELS = HoldoutRobustnessPlotLabels()
PORTFOLIO_HOLDOUT_PLOT_LABELS = HoldoutRobustnessPlotLabels(
    is_sharpe="Research Sharpe",
    holdout_sharpe="Holdout Sharpe",
    sharpe_title="Research vs Holdout Sharpe",
    holdout_bar="Holdout bar",
    z_title="Holdout return z-scores (research μ, σ)",
    equity_title="Holdout equity curve confidence bands",
    is_metric="Research metric",
    holdout_metric="Holdout metric",
    rank_title="Parameter grid rank correlation",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_validation_sharpe_comparison(
    report: ValidationRobustnessReport,
    output_path: Path,
    *,
    plot_labels: HoldoutRobustnessPlotLabels = DEFAULT_VALIDATION_PLOT_LABELS,
) -> Path:
    """§5.2 — IS vs holdout Sharpe bars with confidence intervals."""

    sharpe = report.sharpe_comparison
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [plot_labels.is_sharpe, plot_labels.holdout_sharpe]
    values = [sharpe.sr_is, sharpe.sr_val]
    ci_lowers = [sharpe.ci_is.lower, sharpe.ci_val.lower]
    ci_uppers = [sharpe.ci_is.upper, sharpe.ci_val.upper]
    y_pos = np.arange(len(labels))
    colors = ["#2563eb", "#059669"]
    ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.45)
    for index, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers, strict=True)):
        left_err = max(0.0, float(values[index]) - float(lower))
        right_err = max(0.0, float(upper) - float(values[index]))
        ax.errorbar(
            values[index],
            y_pos[index],
            xerr=[[left_err], [right_err]],
            fmt="none",
            color="#111827",
            capsize=4,
            linewidth=1.5,
        )
    if sharpe.ci_overlap:
        overlap_lower = max(sharpe.ci_is.lower, sharpe.ci_val.lower)
        overlap_upper = min(sharpe.ci_is.upper, sharpe.ci_val.upper)
        if overlap_upper > overlap_lower:
            ax.axvspan(overlap_lower, overlap_upper, color="#fbbf24", alpha=0.25, label="CI overlap")
            ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title(plot_labels.sharpe_title)
    degradation_pct = sharpe.degradation_ratio * 100.0 if math.isfinite(sharpe.degradation_ratio) else float("nan")
    ax.text(
        0.02,
        0.02,
        (
            f"Degradation ratio: {degradation_pct:.0f}%  |  "
            f"CI overlap: {'Yes' if sharpe.ci_overlap else 'No'}  |  "
            f"Pass: {'Yes' if sharpe.passed else 'No'}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    ax.axvline(0.0, color="#6b7280", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_z_cusum_rolling_sr(
    z_cusum_csv: Path,
    report: ValidationRobustnessReport,
    output_path: Path,
    *,
    plot_labels: HoldoutRobustnessPlotLabels = DEFAULT_VALIDATION_PLOT_LABELS,
) -> Path:
    """§5.3 — Three-panel z-score, CUSUM, and rolling Sharpe z chart."""

    frame = pd.read_csv(z_cusum_csv)
    _ensure_parent(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    x = np.arange(frame.shape[0])

    ax_z = axes[0]
    z_values = frame["z_series"].to_numpy(dtype=float)
    z_mean = frame["z_rolling_mean_60"].to_numpy(dtype=float)
    ax_z.bar(x, z_values, color="#93c5fd", width=1.0, label="Daily z")
    ax_z.plot(x, z_mean, color="#1d4ed8", linewidth=1.5, label=f"{ROLLING_Z_MEAN_WINDOW}-bar mean")
    ax_z.axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.8)
    ax_z.axhline(-1.0, color="#6b7280", linestyle="--", linewidth=0.8)
    ax_z.axhline(0.0, color="#111827", linewidth=0.6)
    ax_z.set_ylabel("Return z")
    ax_z.set_title(plot_labels.z_title)
    ax_z.legend(loc="upper right", fontsize=8)

    ax_cusum = axes[1]
    cusum_values = frame["cusum_series"].to_numpy(dtype=float)
    ax_cusum.plot(x, cusum_values, color="#7c3aed", linewidth=1.5, label="CUSUM")
    cusum = report.cusum
    threshold_line = cusum.critical_value * math.sqrt(max(cusum.n_obs, 1))
    ax_cusum.axhline(threshold_line, color="#ef4444", linestyle="--", linewidth=1.0, label="5% envelope")
    ax_cusum.axhline(-threshold_line, color="#ef4444", linestyle="--", linewidth=1.0)
    ax_cusum.axhline(0.0, color="#111827", linewidth=0.6)
    ax_cusum.set_ylabel("CUSUM")
    ax_cusum.set_title(
        f"CUSUM statistic {cusum.statistic:.3f} / critical {cusum.critical_value:.3f}"
    )
    ax_cusum.legend(loc="upper right", fontsize=8)

    ax_sr = axes[2]
    rolling_z = frame["rolling_sharpe_z"].to_numpy(dtype=float)
    threshold = float(report.rolling_sharpe_zscore.z_threshold)
    ax_sr.plot(x, rolling_z, color="#059669", linewidth=1.2, label="Rolling SR z")
    ax_sr.axhline(threshold, color="#ef4444", linestyle="--", linewidth=1.0, label=f"z = {threshold:.1f}")
    below = np.isfinite(rolling_z) & (rolling_z < threshold)
    ax_sr.fill_between(x, threshold, rolling_z, where=below, color="#fecaca", alpha=0.5)
    ax_sr.set_ylabel("Rolling SR z")
    ax_sr.set_xlabel(plot_labels.holdout_bar)
    ax_sr.set_title(
        f"Rolling Sharpe z — {report.rolling_sharpe_zscore.fraction_below_threshold:.0%} below threshold"
    )
    ax_sr.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_equity_bands(
    bands_csv: Path,
    report: ValidationRobustnessReport,
    output_path: Path,
    *,
    plot_labels: HoldoutRobustnessPlotLabels = DEFAULT_VALIDATION_PLOT_LABELS,
) -> Path:
    """§5.4 — Cumulative equity curve with IS-predicted confidence bands."""

    frame = pd.read_csv(bands_csv)
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = frame["bar_index"].to_numpy(dtype=int)
    actual = frame["actual"].to_numpy(dtype=float)
    expected = frame["expected"].to_numpy(dtype=float)
    upper = frame["upper_band"].to_numpy(dtype=float)
    lower = frame["lower_band"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        lower,
        upper,
        color="#3b82f6",
        alpha=0.32,
        edgecolor="#1d4ed8",
        linewidth=0.8,
        label="95% band",
        zorder=1,
    )
    ax.plot(
        x,
        upper,
        color="#1e40af",
        linewidth=1.4,
        linestyle="-",
        alpha=0.95,
        label="Upper band",
        zorder=2,
    )
    ax.plot(
        x,
        lower,
        color="#1e40af",
        linewidth=1.4,
        linestyle="-",
        alpha=0.95,
        label="Lower band",
        zorder=2,
    )
    ax.plot(
        x,
        expected,
        color="#2563eb",
        linestyle="--",
        linewidth=1.4,
        label="Expected (IS μ)",
        zorder=3,
    )
    ax.plot(
        x,
        actual,
        color="#111827",
        linewidth=1.8,
        label="Actual cumulative return",
        zorder=4,
    )
    bands = report.equity_curve_bands
    ax.text(
        0.02,
        0.02,
        (
            f"Below lower band: {bands.fraction_below_lower:.0%}  |  "
            f"Pass: {'Yes' if bands.passed else 'No'}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    ax.set_xlabel(plot_labels.holdout_bar)
    ax.set_ylabel("Cumulative return")
    ax.set_title(plot_labels.equity_title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_rank_scatter(
    rank_csv: Path,
    report: ValidationRobustnessReport,
    output_path: Path,
    *,
    chosen_combo_label: str,
    plot_labels: HoldoutRobustnessPlotLabels = DEFAULT_VALIDATION_PLOT_LABELS,
) -> Path:
    """§5.5 — IS vs validation metric scatter with chosen combo highlighted."""

    frame = pd.read_csv(rank_csv)
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    is_metrics = frame["is_metric"].to_numpy(dtype=float)
    val_metrics = frame["val_metric"].to_numpy(dtype=float)
    chosen_mask = frame["is_chosen"].astype(bool).to_numpy()
    other_mask = ~chosen_mask
    ax.scatter(
        is_metrics[other_mask],
        val_metrics[other_mask],
        color="#93c5fd",
        s=48,
        alpha=0.85,
        label="Other combos",
    )
    if chosen_mask.any():
        ax.scatter(
            is_metrics[chosen_mask],
            val_metrics[chosen_mask],
            color="#dc2626",
            s=96,
            edgecolors="#111827",
            linewidths=1.0,
            label=f"Chosen ({chosen_combo_label})",
            zorder=3,
        )
    if is_metrics.shape[0] >= 2:
        slope, intercept = np.polyfit(is_metrics, val_metrics, deg=1)
        x_line = np.linspace(is_metrics.min(), is_metrics.max(), num=50)
        ax.plot(x_line, slope * x_line + intercept, color="#374151", linestyle="--", linewidth=1.0, label="Trend")
    rank = report.rank_correlation
    ax.set_xlabel(plot_labels.is_metric)
    ax.set_ylabel(plot_labels.holdout_metric)
    ax.set_title(plot_labels.rank_title)
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ = {rank.spearman_rho:.3f}  (p = {rank.p_value:.4f})",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(0.0, color="#d1d5db", linewidth=0.6)
    ax.axvline(0.0, color="#d1d5db", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_validation_robustness_plots(
    report: ValidationRobustnessReport,
    *,
    plot_csv_dir: Path,
    output_dir: Path,
    chosen_combo_label: str = "",
    plot_labels: HoldoutRobustnessPlotLabels = DEFAULT_VALIDATION_PLOT_LABELS,
    include_rank_scatter: bool = True,
    artifact_prefix: str = "validation",
) -> list[Path]:
    """Render holdout robustness Matplotlib artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_csv = plot_csv_dir / f"{artifact_prefix}_rank_correlation.csv"
    z_cusum_csv = plot_csv_dir / f"{artifact_prefix}_z_cusum_sr.csv"
    bands_csv = plot_csv_dir / f"{artifact_prefix}_equity_bands.csv"

    generated = [
        plot_validation_sharpe_comparison(
            report,
            output_dir / f"{artifact_prefix}_sharpe_comparison.png",
            plot_labels=plot_labels,
        ),
        plot_validation_z_cusum_rolling_sr(
            z_cusum_csv,
            report,
            output_dir / f"{artifact_prefix}_z_cusum_rolling_sr.png",
            plot_labels=plot_labels,
        ),
        plot_validation_equity_bands(
            bands_csv,
            report,
            output_dir / f"{artifact_prefix}_equity_bands.png",
            plot_labels=plot_labels,
        ),
    ]
    if include_rank_scatter and rank_csv.exists():
        generated.append(
            plot_validation_rank_scatter(
                rank_csv,
                report,
                output_dir / f"{artifact_prefix}_rank_scatter.png",
                chosen_combo_label=chosen_combo_label,
                plot_labels=plot_labels,
            )
        )
    return generated
