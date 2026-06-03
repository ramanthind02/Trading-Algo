"""Matplotlib reports for the portfolio addition gate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


def write_portfolio_addition_plots(
    payload: Mapping[str, object],
    *,
    bootstrap_csv: Path | None,
    corr_matrix_csv: Path | None = None,
    drawdown_detail_csv: Path | None = None,
    output_dir: Path,
) -> list[Path]:
    """Write portfolio addition gate plots from persisted JSON/CSV artifacts."""

    if payload.get("skipped"):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if bootstrap_csv is not None and bootstrap_csv.exists():
        generated.append(
            _write_delta_sr_histogram(
                bootstrap_csv=bootstrap_csv,
                payload=payload,
                output_dir=output_dir,
            )
        )
    if corr_matrix_csv is not None and corr_matrix_csv.exists():
        generated.append(
            _write_correlation_heatmap(
                corr_matrix_csv=corr_matrix_csv,
                payload=payload,
                output_dir=output_dir,
            )
        )
    hurdle = payload.get("analytical_hurdle", {})
    if isinstance(hurdle, Mapping) and hurdle:
        generated.append(
            _write_drawdown_correlation_chart(
                hurdle=hurdle,
                drawdown_detail_csv=drawdown_detail_csv,
                output_dir=output_dir,
            )
        )
    risk = payload.get("risk_impact", {})
    if isinstance(risk, Mapping) and risk:
        generated.append(
            _write_risk_impact_delta_chart(
                risk=risk,
                empirical=payload.get("empirical_comparison", {}),
                output_dir=output_dir,
            )
        )
    return generated


def _write_risk_impact_delta_chart(
    *,
    risk: Mapping[str, object],
    empirical: object,
    output_dir: Path,
) -> Path:
    empirical_map = empirical if isinstance(empirical, Mapping) else {}
    labels = ["ΔSR", "ΔmaxDD", "Δulcer", "Δstress DD"]
    values = [
        float(empirical_map.get("delta_sr", 0.0)),
        float(risk.get("delta_max_dd", 0.0)),
        float(risk.get("delta_ulcer", 0.0)),
        float(risk.get("delta_stress_max_dd", 0.0)),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.barh(labels, values, color=colors, alpha=0.9)
    axis.set_title("Portfolio addition risk & return deltas (sleeve)")
    axis.set_xlabel("Delta (SR in ratio units; drawdowns in fraction)")
    fig.tight_layout()
    out = output_dir / "portfolio_addition_risk_impact_deltas.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _write_delta_sr_histogram(
    *,
    bootstrap_csv: Path,
    payload: Mapping[str, object],
    output_dir: Path,
) -> Path:
    frame = pd.read_csv(bootstrap_csv)
    samples = frame["delta_sr"].astype(float)
    empirical = payload.get("empirical_comparison", {})
    threshold = float(empirical.get("delta_sr_threshold", 0.02)) if isinstance(empirical, Mapping) else 0.02
    point = float(empirical.get("delta_sr", 0.0)) if isinstance(empirical, Mapping) else 0.0
    ci = empirical.get("delta_sr_ci", {}) if isinstance(empirical, Mapping) else {}
    ci_lower = float(ci.get("lower", 0.0)) if isinstance(ci, Mapping) else 0.0
    ci_upper = float(ci.get("upper", 0.0)) if isinstance(ci, Mapping) else 0.0

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axvspan(0.0, max(float(samples.max()), threshold), color="#f58518", alpha=0.10)
    axis.hist(samples, bins=40, color="#4c78a8", alpha=0.85, edgecolor="white")
    axis.axvline(threshold, color="#e45756", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.2f})")
    axis.axvline(point, color="#f58518", linestyle="-", linewidth=1.5, label=f"Point ΔSR ({point:+.2f})")
    axis.axvline(ci_lower, color="#666666", linestyle=":", linewidth=1.2, label=f"CI lower ({ci_lower:+.2f})")
    axis.axvline(ci_upper, color="#666666", linestyle=":", linewidth=1.2, label=f"CI upper ({ci_upper:+.2f})")
    axis.axvline(0.0, color="#333333", linestyle="-", linewidth=0.8)
    axis.set_title("Portfolio addition ΔSR bootstrap distribution")
    axis.set_xlabel("ΔSR")
    axis.set_ylabel("Count")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = output_dir / "portfolio_addition_delta_sr_histogram.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _truncate_label(label: str, *, max_len: int = 22) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 1]}…"


def _correlation_heatmap_axis_labels(
    labels: Sequence[str],
    *,
    candidate_key: str,
) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Return axis labels, (short, full) legend rows, and whether codes are used."""
    max_label_len = max((len(label) for label in labels), default=0)
    use_codes = len(labels) > 5 or max_label_len > 24
    if use_codes:
        display = [
            f"S{index}{'*' if label == candidate_key else ''}"
            for index, label in enumerate(labels)
        ]
    else:
        display = [_truncate_label(label) for label in labels]
    legend = list(zip(display, labels, strict=True))
    return display, legend, use_codes


def _correlation_heatmap_figure_size(
    n: int,
    *,
    use_codes: bool,
    max_label_len: int,
) -> tuple[float, float]:
    cell = 0.72
    heatmap_span = max(4.0, n * cell)
    width = heatmap_span + 1.4
    legend_rows = max(1, (n + 1) // 2)
    legend_height = 0.55 + legend_rows * 0.22 if use_codes else 0.0
    if not use_codes:
        width += min(max_label_len * 0.07, 2.8)
    height = heatmap_span + legend_height + 1.0
    return (width, height)


def _write_correlation_heatmap(
    *,
    corr_matrix_csv: Path,
    payload: Mapping[str, object],
    output_dir: Path,
) -> Path:
    matrix = pd.read_csv(corr_matrix_csv, index_col=0)
    labels = [str(label) for label in matrix.index.tolist()]
    context = payload.get("context", {})
    candidate_key = str(
        (context.get("candidate_key") if isinstance(context, Mapping) else None)
        or (payload.get("meta") or {}).get("candidate_key")
        or labels[0]
    )
    display_labels, legend_rows, use_codes = _correlation_heatmap_axis_labels(
        labels,
        candidate_key=candidate_key,
    )
    max_label_len = max((len(label) for label in labels), default=0)
    fig_width, fig_height = _correlation_heatmap_figure_size(
        len(labels),
        use_codes=use_codes,
        max_label_len=max_label_len,
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#ffffff")
    if use_codes:
        grid = GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.0, max(0.28, len(labels) * 0.045)],
            width_ratios=[1.0, 0.05],
            hspace=0.28,
            wspace=0.08,
        )
        axis = fig.add_subplot(grid[0, 0])
        color_axis = fig.add_subplot(grid[0, 1])
        legend_axis = fig.add_subplot(grid[1, :])
    else:
        grid = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.05], wspace=0.08)
        axis = fig.add_subplot(grid[0, 0])
        color_axis = fig.add_subplot(grid[0, 1])
        legend_axis = None

    values = matrix.astype(float).values
    axis.set_facecolor("#fbfbfc")
    image = axis.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(display_labels, fontsize=9)
    axis.tick_params(length=0)
    axis.grid(False)

    annotate_cells = len(labels) <= 10
    annotation_fontsize = max(7, min(10, int(72 / max(len(labels), 1))))
    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            if row_label == candidate_key or col_label == candidate_key:
                axis.add_patch(
                    plt.Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#f58518",
                        linewidth=1.5,
                    )
                )
            if not annotate_cells:
                continue
            value = float(values[row_index, col_index])
            text_color = "white" if abs(value) > 0.55 else "#111827"
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=annotation_fontsize,
            )

    axis.set_title(
        "Strategy return correlation (IS + validation)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.colorbar(image, cax=color_axis, label="Pearson ρ")

    if legend_axis is not None:
        legend_axis.axis("off")
        legend_axis.set_title("Strategy labels (* = inclusion candidate)", loc="left", fontsize=9)
        legend_text = "\n".join(f"{short}: {full}" for short, full in legend_rows)
        legend_axis.text(
            0.0,
            1.0,
            legend_text,
            va="top",
            ha="left",
            fontsize=8,
            color="#334155",
            family="monospace",
            wrap=True,
        )

    out = output_dir / "portfolio_addition_correlation_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _write_drawdown_correlation_chart(
    *,
    hurdle: Mapping[str, object],
    drawdown_detail_csv: Path | None,
    output_dir: Path,
) -> Path:
    detail = dict(hurdle)
    if drawdown_detail_csv is not None and drawdown_detail_csv.exists():
        detail.update(pd.read_csv(drawdown_detail_csv).iloc[0].to_dict())

    labels = ["Unconditional", "Drawdown stress", "Effective"]
    values = [
        float(detail.get("corr_unconditional", 0.0)),
        float(detail.get("corr_drawdown_conditional", 0.0)),
        float(detail.get("corr_effective", 0.0)),
    ]
    colors = ["#4c78a8", "#e45756", "#f58518"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.2),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    bar_axis = axes[0]
    bars = bar_axis.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")
    bar_axis.axhline(0.0, color="#333333", linewidth=0.8)
    bar_axis.set_ylim(-1.0, 1.0)
    bar_axis.set_ylabel("Correlation with portfolio")
    bar_axis.set_title("Drawdown correlation analysis")
    for bar, value in zip(bars, values, strict=True):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.04 if value >= 0 else -0.08),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    metrics_axis = axes[1]
    metrics_axis.axis("off")
    overlap = float(detail.get("drawdown_overlap", 0.0))
    joint_dd = float(detail.get("joint_drawdown_depth", 0.0))
    uplift = float(detail.get("corr_drawdown_uplift", values[1] - values[0]))
    n_stress = int(detail.get("n_stress_periods", 0))
    metric_lines = [
        ("Drawdown overlap ratio", overlap, bool(detail.get("drawdown_overlap_flagged", False))),
        ("Joint drawdown depth", joint_dd, bool(detail.get("joint_dd_depth_flagged", False))),
        ("Drawdown corr uplift", uplift, bool(detail.get("drawdown_corr_uplift_flagged", False))),
        ("Stress periods", float(n_stress), False),
    ]
    y_pos = 0.85
    for label, value, flagged in metric_lines:
        color = "#e45756" if flagged else "#333333"
        if label == "Stress periods":
            text = f"{label}: {int(value)}"
        elif "depth" in label.lower():
            text = f"{label}: {value:.1%}"
        elif "overlap" in label.lower():
            text = f"{label}: {value:.0%}"
        else:
            text = f"{label}: {value:+.2f}"
        metrics_axis.text(0.02, y_pos, text, fontsize=10, color=color, transform=metrics_axis.transAxes)
        y_pos -= 0.18
    metrics_axis.set_title("Drawdown stress diagnostics")

    fig.tight_layout()
    out = output_dir / "portfolio_addition_drawdown_correlation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
