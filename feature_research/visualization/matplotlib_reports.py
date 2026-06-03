"""Generate Matplotlib plots from CSV-first research exports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from collections.abc import Callable, Sequence
from pathlib import Path
import re
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from feature_research.research_table_exports import (
    AGGREGATE_EQUITY_TICKER,
    canonical_in_sample_visualization_dir,
)


@dataclass(frozen=True)
class MatplotlibReportOptions:
    top_n: int = 12


PlotOutput = Union[Path, Sequence[Path]]
CsvPlotter = Callable[[Path, Path, MatplotlibReportOptions], PlotOutput | None]


def _normalize_plot_outputs(result: PlotOutput | None) -> list[Path]:
    if result is None:
        return []
    if isinstance(result, Path):
        return [result]
    return [path for path in result if isinstance(path, Path)]


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_style(title: str, *, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfc")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    return fig, ax


# ---------------------------------------------------------------------------
# Filter gate A / B / C comparison helpers
# ---------------------------------------------------------------------------

def _plot_filter_gate_comparison(annotated: pd.DataFrame, base_output_path: Path) -> Path | None:
    """Two-subplot chart: Sharpe (top) and n_nonzero_signal (bottom) for all A/B/C combos."""
    if not {"short_label", "gate_type", "sharpe"}.issubset(annotated.columns):
        return None
    sharpe_vals = pd.to_numeric(annotated["sharpe"], errors="coerce")
    if sharpe_vals.dropna().empty:
        return None

    n_combos = len(annotated)
    labels = annotated["short_label"].tolist()
    details = (
        annotated["filter_detail"].astype(str).tolist()
        if "filter_detail" in annotated.columns
        else [""] * n_combos
    )
    gate_types = annotated["gate_type"].tolist()
    gate_colors: dict[str, str] = {"A": "#60a5fa", "B": "#22c55e", "C": "#f97316"}
    bar_colors = [gate_colors.get(g, "#94a3b8") for g in gate_types]
    x = np.arange(n_combos)
    bar_width = 0.65

    has_counts = "n_nonzero_signal" in annotated.columns
    n_rows = 2 if has_counts else 1
    fig_h = 9.0 if has_counts else 5.5
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(8.0, n_combos * 1.2), fig_h), sharex=True)
    fig.patch.set_facecolor("#ffffff")
    ax_top = axes[0] if has_counts else axes

    ax_top.set_facecolor("#fbfbfc")
    ax_top.set_title(
        "Filter exploration A/B/C — Sharpe (IS)\n"
        "A=baseline · B=entry-only gate · C=entry+exit gate",
        fontsize=11,
        fontweight="bold",
    )
    ax_top.bar(x, sharpe_vals.fillna(0.0), width=bar_width, color=bar_colors, edgecolor="#334155")
    baseline_mask = annotated["gate_type"].eq("A")
    baseline_sharpe_series = sharpe_vals[baseline_mask]
    if not baseline_sharpe_series.empty and pd.notna(baseline_sharpe_series.iloc[0]):
        bv = float(baseline_sharpe_series.iloc[0])
        ax_top.axhline(bv, color="#ef4444", linestyle="--", linewidth=1.3,
                       label=f"Baseline {bv:.2f}")
        ax_top.legend(fontsize=8)
    ax_top.set_ylabel("Sharpe")
    ax_top.grid(alpha=0.2, linestyle="--", axis="y")

    if has_counts:
        count_vals = pd.to_numeric(annotated["n_nonzero_signal"], errors="coerce")
        ax_bot = axes[1]
        ax_bot.set_facecolor("#fbfbfc")
        ax_bot.set_title("Trade Count Proxy (n_nonzero_signal)", fontsize=11, fontweight="bold")
        ax_bot.bar(x, count_vals.fillna(0.0), width=bar_width, color=bar_colors, edgecolor="#334155")

        baseline_count_series = count_vals[baseline_mask]
        if not baseline_count_series.empty and pd.notna(baseline_count_series.iloc[0]):
            base_n = float(baseline_count_series.iloc[0])
            if base_n > 0:
                for xi, (n, gate) in enumerate(zip(count_vals.tolist(), gate_types, strict=False)):
                    if gate in ("B", "C") and pd.notna(n):
                        pct_red = (base_n - float(n)) / base_n * 100
                        ax_bot.text(xi, float(n) + base_n * 0.015,
                                    f"{pct_red:.0f}%↓", ha="center", va="bottom",
                                    fontsize=7.5, color="#475569")

        ax_bot.set_ylabel("n_nonzero_signal")
        ax_bot.set_xticks(x)
        tick_labels = [f"{lbl}\n{det}" if det else lbl for lbl, det in zip(labels, details, strict=False)]
        ax_bot.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
        ax_bot.grid(alpha=0.2, linestyle="--", axis="y")
        legend_patches = [
            Patch(facecolor=gate_colors["A"], label="A: Baseline (unfiltered)"),
            Patch(facecolor=gate_colors["B"], label="B: filter_gate_entry_only"),
            Patch(facecolor=gate_colors["C"], label="C: filter_gate (entry + exit)"),
        ]
        ax_bot.legend(handles=legend_patches, fontsize=8, loc="upper right")
    else:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    fig.tight_layout()
    out = base_output_path.with_name("filter_gate_comparison.png")
    return _save_figure(fig, out)


def plot_filter_exploration_summary_csv(
    csv_path: Path,
    output_path: Path,
    options: MatplotlibReportOptions,
) -> Path | None:
    """Render A/B/C filter comparison from ``filter_exploration_summary.csv`` (winner follow-up)."""
    _ = options
    frame = _read_csv_if_nonempty(csv_path)
    if frame is None:
        return None
    required = {"research_display_label", "gate_type", "sharpe"}
    if not required.issubset(frame.columns):
        return None
    annotated = frame.assign(short_label=frame["research_display_label"].astype(str))
    if "filter_detail" not in annotated.columns:
        annotated = annotated.assign(filter_detail="")
    return _plot_filter_gate_comparison(annotated, output_path)


# ---------------------------------------------------------------------------
# Binning phase — ATR% decile chart (reads Parquet, called from pipeline)
# ---------------------------------------------------------------------------

def plot_binning_decile_parquet(parquet_path: Path, output_path: Path) -> Path | None:
    """Decile bar chart from bin_metrics.parquet: ATR% bin vs mean return per ticker.

    When Pass 1 runs with an investigation strategy (e.g. locked cumRSI), ``mean_return``
    is the mean strategy return (signal × EWSD) on days with non-zero signal in that ATR
    decile.  Otherwise it is unconditional mean forward return.
    """
    if not parquet_path.exists():
        return None
    manifest: dict[str, object] = {}
    manifest_path = parquet_path.parent / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    try:
        bin_metrics = pd.read_parquet(parquet_path)
    except Exception:
        return None
    required = {"bin_index", "mean_return"}
    if bin_metrics.empty or not required.issubset(bin_metrics.columns):
        return None

    bin_metrics = bin_metrics.copy()
    bin_metrics["bin_index"] = pd.to_numeric(bin_metrics["bin_index"], errors="coerce")
    bin_metrics["mean_return"] = pd.to_numeric(bin_metrics["mean_return"], errors="coerce")
    bin_metrics = bin_metrics.dropna(subset=["bin_index", "mean_return"])
    if bin_metrics.empty:
        return None

    has_ticker = "ticker" in bin_metrics.columns
    tickers: list[str] = (
        sorted(bin_metrics["ticker"].astype(str).unique()) if has_ticker else ["all"]
    )
    n_tickers = len(tickers)
    fig, axes = plt.subplots(1, n_tickers, figsize=(max(7.0, 5.0 * n_tickers), 5.5), squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    strategy_conditioned = manifest.get("metrics_return_col") == "strategy_return"
    investigation = str(manifest.get("investigation_strategy", "") or "").strip()
    if strategy_conditioned and investigation:
        chart_title = (
            f"ATR% Decile vs Mean Strategy Return — {investigation} (Pass 1)"
        )
        y_label = "Mean Strategy Return (signal × EWSD)"
    elif strategy_conditioned:
        chart_title = "ATR% Decile vs Mean Strategy Return (Pass 1)"
        y_label = "Mean Strategy Return (signal × EWSD)"
    else:
        chart_title = "ATR% Decile vs Mean Forward Return (Pass 1 — Vol Regime EDA)"
        y_label = "Mean Forward Return"
    active_only = bool(manifest.get("active_signal_only"))
    if active_only:
        chart_title = f"{chart_title}\n(non-zero signal days only)"
    fig.suptitle(chart_title, fontsize=11, fontweight="bold")

    for i, ticker in enumerate(tickers):
        ax = axes[0][i]
        ax.set_facecolor("#fbfbfc")
        subset = (
            bin_metrics[bin_metrics["ticker"].astype(str) == ticker]
            if has_ticker else bin_metrics
        )
        if subset.empty:
            continue
        grouped = (
            subset.groupby("bin_index", observed=True)["mean_return"]
            .mean()
            .sort_index()
        )
        bins = grouped.index.tolist()
        returns = grouped.to_numpy(dtype=float).tolist()
        colors = ["#22c55e" if r > 0 else "#ef4444" for r in returns]
        ax.bar(range(1, len(bins) + 1), returns, color=colors, edgecolor="#334155", width=0.72)
        ax.axhline(0, color="#64748b", linewidth=0.9, linestyle="--")
        ax.set_xlabel("ATR% Decile  (1 = low vol → 10 = high vol)", fontsize=9)
        ax.set_ylabel(y_label if i == 0 else "")
        ax.set_title(ticker, fontsize=10, fontweight="bold")
        ax.set_xticks(range(1, len(bins) + 1))
        ax.set_xticklabels([str(b) for b in range(1, len(bins) + 1)], fontsize=8)
        ax.grid(alpha=0.2, linestyle="--", axis="y")

    fig.tight_layout()
    return _save_figure(fig, output_path)


def _prepare_equity_curve_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    required_cols = {"datetime", "cumulative_strategy_return"}
    label_columns = [col for col in ("param_combo_label", "ticker") if col in frame.columns]
    if not required_cols.issubset(frame.columns) or not label_columns:
        return None
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["cumulative_strategy_return"] = pd.to_numeric(
        work["cumulative_strategy_return"],
        errors="coerce",
    )
    work = work.dropna(subset=["datetime", "cumulative_strategy_return"])
    return None if work.empty else work


def _top_equity_curve_rows(work: pd.DataFrame, *, top_n: int, label_column: str) -> pd.DataFrame:
    final_scores = (
        work.sort_values("datetime")
        .groupby(label_column, sort=False)["cumulative_strategy_return"]
        .last()
        .abs()
        .sort_values(ascending=False)
    )
    top_labels = final_scores.head(min(top_n, len(final_scores))).index
    return work[work[label_column].isin(top_labels)].sort_values("datetime")


def _save_equity_curve_plot(
    work: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    label_column: str,
) -> Path:
    fig, ax = _plot_style(title, figsize=(11.0, 6.0))
    _ = [
        ax.plot(
            group["datetime"],
            group["cumulative_strategy_return"],
            linewidth=1.6,
            label=str(series_label),
        )
        for series_label, group in work.groupby(label_column, sort=False)
    ]
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Cumulative strategy return")
    ax.legend(loc="best", fontsize=8)
    return _save_figure(fig, output_path)


def _equity_curve_instrument_tickers(work: pd.DataFrame) -> list[str]:
    if "ticker" not in work.columns:
        return []
    return sorted(
        ticker
        for ticker in work["ticker"].astype(str).unique()
        if ticker != AGGREGATE_EQUITY_TICKER
    )


def _combined_equity_curve_rows(work: pd.DataFrame) -> pd.DataFrame | None:
    if "ticker" not in work.columns:
        return None
    combined = work.loc[work["ticker"].astype(str).eq(AGGREGATE_EQUITY_TICKER)].copy()
    if combined.empty:
        return None
    combined["series_label"] = combined["param_combo_label"].astype(str)
    return combined


def plot_equity_curve_csv(
    csv_path: Path,
    output_path: Path,
    options: MatplotlibReportOptions,
) -> list[Path] | None:
    frame = _read_csv_if_nonempty(csv_path)
    if frame is None:
        return None
    work = _prepare_equity_curve_frame(frame)
    if work is None:
        return None

    outputs: list[Path] = []
    instrument_tickers = _equity_curve_instrument_tickers(work)
    combined = _combined_equity_curve_rows(work)
    if combined is not None:
        top_combined = _top_equity_curve_rows(
            combined,
            top_n=options.top_n,
            label_column="series_label",
        )
        aggregate_title = f"Equity Curves: {csv_path.stem} (Combined Tickers)"
        outputs.append(
            _save_equity_curve_plot(
                top_combined,
                title=aggregate_title,
                output_path=output_path,
                label_column="series_label",
            )
        )
    else:
        aggregate = work.copy()
        aggregate["series_label"] = (
            aggregate[["param_combo_label", "ticker"]].astype(str).agg(" | ".join, axis=1)
            if "ticker" in aggregate.columns
            else aggregate["param_combo_label"].astype(str)
        )
        top_aggregate = _top_equity_curve_rows(
            aggregate,
            top_n=options.top_n,
            label_column="series_label",
        )
        aggregate_title = (
            f"Equity Curves: {csv_path.stem} (All Tickers)"
            if instrument_tickers
            else f"Equity Curves: {csv_path.stem}"
        )
        outputs.append(
            _save_equity_curve_plot(
                top_aggregate,
                title=aggregate_title,
                output_path=output_path,
                label_column="series_label",
            )
        )

    return outputs


def plot_permutation_vector_shuffle_csv(
    csv_path: Path,
    output_path: Path,
    options: MatplotlibReportOptions,
) -> Path | None:
    frame = _read_csv_if_nonempty(csv_path)
    required_cols = {"param_combo_label", "observed_metric", "p_value", "passed"}
    if frame is None or not required_cols.issubset(frame.columns):
        return None
    work = frame.assign(
        observed_value=pd.to_numeric(frame["observed_metric"], errors="coerce"),
        p_value_num=pd.to_numeric(frame["p_value"], errors="coerce"),
    ).dropna(subset=["observed_value", "p_value_num"])
    if work.empty:
        return None
    top = work.nsmallest(min(options.top_n, len(work)), "p_value_num").sort_values("observed_value")
    colors = ["#22c55e" if passed else "#ef4444" for passed in top["passed"].tolist()]
    fig, ax = _plot_style("Permutation Vector Shuffle Summary", figsize=(10.0, 6.0))
    bars = ax.barh(
        top["param_combo_label"].astype(str),
        top["observed_value"],
        color=colors,
        edgecolor="#334155",
    )
    _ = [
        ax.text(
            float(bar.get_width()),
            bar.get_y() + bar.get_height() / 2,
            f"  p={p_value:.3f}",
            va="center",
            fontsize=8,
        )
        for bar, p_value in zip(bars, top["p_value_num"].tolist(), strict=False)
    ]
    ax.set_xlabel("Observed metric")
    ax.set_ylabel("Param combo")
    return _save_figure(fig, output_path)


def plot_robustness_rolling_cusum_csv(
    csv_path: Path,
    output_path: Path,
    options: MatplotlibReportOptions,
) -> Path | None:
    from feature_research.visualization.cusum_stability import plot_rolling_cusum_stability_csv

    return plot_rolling_cusum_stability_csv(csv_path, output_path)


def plot_vault_correlation_csv(
    csv_path: Path,
    output_path: Path,
    options: MatplotlibReportOptions,
) -> Path | None:
    frame = _read_csv_if_nonempty(csv_path)
    required_cols = {"vault_feature_name", "metric_name", "metric_value", "ticker"}
    if frame is None or not required_cols.issubset(frame.columns):
        return None
    preferred = frame[frame["metric_name"].astype(str) == "pearson_return_corr"]
    work = preferred if not preferred.empty else frame
    work = work.assign(metric_num=pd.to_numeric(work["metric_value"], errors="coerce")).dropna(
        subset=["metric_num"]
    )
    if work.empty:
        return None
    ranked = work.assign(abs_metric=work["metric_num"].abs()).nlargest(
        min(options.top_n, len(work)),
        "abs_metric",
    )
    ranked = ranked.sort_values("metric_num")
    labels = ranked[["vault_feature_name", "ticker"]].astype(str).agg(" | ".join, axis=1)
    fig, ax = _plot_style("Vault vs Research Correlation", figsize=(10.0, 6.0))
    ax.barh(labels, ranked["metric_num"], color="#a78bfa", edgecolor="#6d28d9")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Vault feature / ticker")
    return _save_figure(fig, output_path)


_CSV_PLOTTERS: tuple[tuple[re.Pattern[str], CsvPlotter], ...] = (
    (re.compile(r"^robustness_rolling_cusum\.csv$"), plot_robustness_rolling_cusum_csv),
    (re.compile(r"^filter_exploration_summary\.csv$"), plot_filter_exploration_summary_csv),
    (re.compile(r"^filter_exploration_equity_curve\.csv$"), plot_equity_curve_csv),
    (re.compile(r"^permutation_vector_shuffle\.csv$"), plot_permutation_vector_shuffle_csv),
    (re.compile(r"^vault_correlation_long\.csv$"), plot_vault_correlation_csv),
    (re.compile(r"^equity_curve.*\.csv$"), plot_equity_curve_csv),
)


def generate_detected_matplotlib_plots(
    input_dir: Path,
    output_dir: Path,
    *,
    top_n: int = 12,
) -> list[Path]:
    """Scan a visualization CSV directory and render supported Matplotlib plots."""
    options = MatplotlibReportOptions(top_n=top_n)
    csv_paths = sorted(path for path in input_dir.rglob("*.csv") if path.is_file())

    def _plot_for_csv(csv_path: Path) -> Path | None:
        relative_png = csv_path.relative_to(input_dir).with_suffix(".png")
        output_path = output_dir / relative_png
        match = next(
            (
                plotter
                for pattern, plotter in _CSV_PLOTTERS
                if pattern.match(csv_path.name) is not None
            ),
            None,
        )
        return None if match is None else match(csv_path, output_path, options)

    generated = [
        path
        for csv_path in csv_paths
        for path in _normalize_plot_outputs(_plot_for_csv(csv_path))
    ]
    manifest_path = output_dir / "matplotlib_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "generated_plots": [str(path) for path in generated],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return generated


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Matplotlib plots from visualization CSV exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=canonical_in_sample_visualization_dir(),
        help="Root directory containing visualization CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PNGs. Defaults to <input-dir>/matplotlib.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Maximum number of series/rows to include per chart.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "matplotlib"
    generated = generate_detected_matplotlib_plots(
        input_dir=input_dir,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n)),
    )
    print(f"Generated {len(generated)} Matplotlib plot(s) under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
