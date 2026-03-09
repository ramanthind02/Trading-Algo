"""Weight Layer report exporter.

Persists WeightLayer diagnostics to CSV, JSON, and an HTML visual report
after each pipeline phase.

Outputs (written to ``output_dir/``):
- ``report.html``             self-contained visual report (charts + tables)
- ``weights_by_group.csv``    per-(ticker × model) flat table with group weights
- ``signal_cross_ticker.csv`` models that appear in multiple tickers
- ``summary.csv``             per-ticker aggregate stats (FDM, n_models, n_groups)
- ``diagnostics.json``        full raw diagnostics dict for programmatic use
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ensemble.weight_layer import BaseWeightLayer
    from ensemble.global_weight_layer import GlobalWeightLayer as _GlobalWeightLayer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (qualitative, up to 20 groups)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _color_for(family: str, palette: dict[str, str]) -> str:
    if family not in palette:
        palette[family] = _PALETTE[len(palette) % len(_PALETTE)]
    return palette[family]


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert diagnostics to JSON-serializable form (str keys, no Enum/numpy)."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if hasattr(obj, "name"):
        return getattr(obj, "name", str(obj))
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ---------------------------------------------------------------------------
# Data-building helpers
# ---------------------------------------------------------------------------

def _clean_ticker(s: str) -> str:
    """Strip enum class prefix from ticker strings (e.g. 'Ticker.ES' → 'ES')."""
    s = str(s)
    return s.split(".")[-1] if "." in s else s


def _abbrev_family(name: str, max_len: int = 12) -> str:
    """Truncate long family names for chart legends."""
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def _family_id(model_name: str) -> str:
    """Extract feature-family group ID from a model name."""
    left = model_name.split("::")[0].strip() if "::" in model_name else model_name
    return left.split("_")[0] if "_" in left else left


def _build_weights_by_group(diagnostics: dict, phase: str) -> pd.DataFrame:
    rows = []
    for ticker, info in diagnostics.get("tickers", {}).items():
        weights: dict = info.get("weights") or {}
        fdm = info.get("fdm", 1.0)
        mean_corr = info.get("mean_forecast_correlation", float("nan"))

        family_of = {model: _family_id(model) for model in weights}
        group_totals: dict[str, float] = {}
        for model, w in weights.items():
            grp = family_of[model]
            group_totals[grp] = group_totals.get(grp, 0.0) + w

        for model, model_weight in sorted(weights.items()):
            grp = family_of[model]
            rows.append({
                "phase": phase,
                "ticker": _clean_ticker(ticker),
                "feature_family": grp,
                "model_name": model,
                "model_weight": round(model_weight, 6),
                "group_weight": round(group_totals[grp], 6),
                "fdm": round(fdm, 4),
                "mean_forecast_correlation": round(mean_corr, 4),
            })
    return pd.DataFrame(rows)


def _build_cross_ticker(weights_df: pd.DataFrame) -> pd.DataFrame:
    if weights_df.empty:
        return pd.DataFrame()

    all_tickers = sorted(weights_df["ticker"].unique())
    rows = []
    for model_name, grp in weights_df[["model_name", "feature_family"]].drop_duplicates().values:
        model_rows = weights_df[weights_df["model_name"] == model_name]
        present_tickers = sorted(str(t) for t in model_rows["ticker"].tolist())
        row: dict = {
            "model_name": model_name,
            "feature_family": grp,
            "tickers_present": ",".join(present_tickers),
            "n_tickers": len(present_tickers),
        }
        for t in all_tickers:
            match = model_rows[model_rows["ticker"] == t]
            row[f"weight_{t}"] = round(float(match["model_weight"].iloc[0]), 6) if not match.empty else None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["n_tickers", "feature_family", "model_name"], ascending=[False, True, True]
    )
    return df.reset_index(drop=True)


def _build_summary(diagnostics: dict, phase: str) -> pd.DataFrame:
    rows = []
    for ticker, info in diagnostics.get("tickers", {}).items():
        weights: dict = info.get("weights") or {}
        families = {_family_id(m) for m in weights}
        rows.append({
            "phase": phase,
            "ticker": _clean_ticker(ticker),
            "n_models": info.get("n_models", len(weights)),
            "n_groups": len(families),
            "fdm": round(info.get("fdm", 1.0), 4),
            "mean_forecast_correlation": round(
                info.get("mean_forecast_correlation", float("nan")), 4
            ),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart generators (return base64-encoded PNG strings)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_group_weights(weights_df: pd.DataFrame) -> str:
    """Stacked bar: weight allocated to each feature family, per ticker."""
    tickers = sorted(weights_df["ticker"].unique())
    families = sorted(weights_df["feature_family"].unique())
    palette: dict[str, str] = {}
    for f in families:
        _color_for(f, palette)

    # Group → total weight per ticker
    pivot = (
        weights_df.groupby(["ticker", "feature_family"])["model_weight"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(tickers)
    )

    fig, ax = plt.subplots(figsize=(max(6, len(tickers) * 1.4), 4.5), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bottom = np.zeros(len(tickers))
    x = np.arange(len(tickers))

    for fam in sorted(pivot.columns):
        vals = pivot[fam].values
        bars = ax.bar(x, vals, bottom=bottom, color=palette.get(fam, "#888"),
                      label=_abbrev_family(fam), width=0.6, edgecolor="white", linewidth=0.5)
        # Label bars that are wide enough to read
        for bar, b, v in zip(bars, bottom, vals):
            if v > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{v:.1%}",
                    ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Allocated Weight", fontsize=10)
    ax.set_title("Group Weight Allocation per Ticker", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8, ncol=max(1, len(families) // 8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_fdm(summary_df: pd.DataFrame, fdm_max: float) -> str:
    """Horizontal bar chart of FDM per ticker."""
    df = summary_df.sort_values("fdm", ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(df) * 0.5 + 1)), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    colors = ["#55A868" if v >= 1.0 else "#C44E52" for v in df["fdm"]]
    bars = ax.barh(df["ticker"].astype(str), df["fdm"], color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, df["fdm"]):
        ax.text(max(v + 0.02, 0.05), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.axvline(1.0, color="#888", linestyle="--", linewidth=1, label="FDM = 1.0 (no diversification)")
    ax.axvline(fdm_max, color="#C44E52", linestyle=":", linewidth=1, label=f"Cap = {fdm_max:.1f}")
    ax.set_xlim(0, fdm_max + 0.25)
    ax.set_xlabel("Forecast Diversification Multiplier", fontsize=10)
    ax.set_title("FDM per Ticker", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_model_weight_heatmap(weights_df: pd.DataFrame) -> str | None:
    """Heatmap: model × ticker, value = model weight (NaN = not present)."""
    tickers = sorted(weights_df["ticker"].unique())
    if len(tickers) < 2:
        return None  # only useful for multi-ticker

    pivot = weights_df.pivot_table(
        index="model_name", columns="ticker", values="model_weight", aggfunc="first"
    ).reindex(columns=tickers)
    # Sort rows by feature family then model name
    family_map = weights_df.set_index("model_name")["feature_family"].to_dict()
    pivot["_fam"] = pivot.index.map(family_map)
    pivot = pivot.sort_values(["_fam", pivot.index.name]).drop(columns=["_fam"])

    fig_h = max(4, len(pivot) * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(max(5, len(tickers) * 1.2), fig_h), facecolor="#FAFAFA")
    ax.set_facecolor("#F4F4F4")

    import matplotlib.colors as mcolors
    cmap = plt.get_cmap("YlOrRd")
    cmap.set_bad("#ECECEC")

    data = pivot.values.astype(float)
    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)

    # Cell annotations
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if not np.isnan(v):
                txt_color = "white" if v > vmax * 0.6 else "#333"
                ax.text(c, r, f"{v:.2%}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

    ax.set_title("Model Weight per Ticker (Heatmap)", fontsize=12, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Model Weight")
    # Draw family group separators
    prev_fam = None
    for r, mname in enumerate(pivot.index):
        fam = family_map.get(mname, "")
        if fam != prev_fam and r > 0:
            ax.axhline(r - 0.5, color="#555", linewidth=0.8, linestyle="--")
        prev_fam = fam

    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_group_pie(weights_df: pd.DataFrame, ticker: str, palette: dict[str, str]) -> str:
    """Donut chart of group weights for a single ticker."""
    t_df = weights_df[weights_df["ticker"] == ticker]
    grp = t_df.groupby("feature_family")["model_weight"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#FAFAFA")
    colors = [_color_for(f, palette) for f in grp.index]
    wedges, texts, autotexts = ax.pie(
        grp.values,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title(ticker, fontsize=11, fontweight="bold")
    ax.legend(
        wedges, [f"{_abbrev_family(f)} ({w:.1%})" for f, w in zip(grp.index, grp.values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        fontsize=7.5, ncol=2, framealpha=0.8,
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #F0F2F5; color: #1A1A2E; }
.page { max-width: 1300px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #333; margin: 28px 0 12px; border-left: 4px solid #4C72B0; padding-left: 10px; }
h3 { font-size: 1rem; font-weight: 600; color: #444; margin-bottom: 8px; }
.subtitle { color: #666; font-size: 0.9rem; margin-bottom: 24px; }
.card { background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 20px; margin-bottom: 20px; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi { background: #fff; border-radius: 10px; padding: 16px 22px; flex: 1; min-width: 140px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
.kpi .val { font-size: 2rem; font-weight: 700; color: #4C72B0; }
.kpi .lbl { font-size: 0.78rem; color: #888; margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }
.chart-row { display: flex; gap: 16px; flex-wrap: wrap; }
.chart-row .card { flex: 1; min-width: 320px; text-align: center; }
img.chart { max-width: 100%; height: auto; border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead th { background: #4C72B0; color: #fff; padding: 8px 10px; text-align: left; position: sticky; top: 0; }
tbody tr:nth-child(even) { background: #F7F8FA; }
tbody tr:hover { background: #EAF0FB; }
td { padding: 6px 10px; border-bottom: 1px solid #EEE; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem;
       font-weight: 600; color: #fff; }
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.tbl-wrap { max-height: 480px; overflow-y: auto; border-radius: 6px; border: 1px solid #E5E7EB; }
.section-note { font-size: 0.8rem; color: #888; margin-bottom: 10px; }
"""


def _html_table(df: pd.DataFrame, highlight_col: str | None = None, palette: dict | None = None) -> str:
    pal = palette or {}
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            v = row[col]
            if col == "feature_family" and pal:
                color = pal.get(str(v), "#888")
                cells += f'<td><span class="tag" style="background:{color}">{v}</span></td>'
            elif isinstance(v, float):
                if col.startswith("weight") or col == "model_weight" or col == "group_weight":
                    cells += f"<td>{v:.4f}" + (f" <small style='color:#888'>({v:.1%})</small>" if v == v else "") + "</td>"
                elif col in ("fdm", "mean_forecast_correlation"):
                    cells += f"<td>{v:.4f}</td>"
                else:
                    cells += f"<td>{v}</td>"
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells += '<td style="color:#CCC">—</td>'
            else:
                cells += f"<td>{v}</td>"
        rows_html += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"


def _build_html_report(
    weights_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    diagnostics: dict,
    phase_name: str,
) -> str:
    from datetime import datetime, timezone

    palette: dict[str, str] = {}
    all_families = sorted(weights_df["feature_family"].unique()) if not weights_df.empty else []
    for f in all_families:
        _color_for(f, palette)

    summary = diagnostics.get("summary", {})
    weight_method = diagnostics.get("weight_method", "unknown")
    fdm_max = float(diagnostics.get("fdm_max", 2.0))
    n_tickers = int(summary.get("n_tickers", 0))
    mean_fdm = float(summary.get("mean_fdm", 1.0))
    total_models = int(summary.get("total_models", 0))
    n_groups = len(all_families)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Charts ---
    chart_stacked = _chart_group_weights(weights_df) if not weights_df.empty else None
    chart_fdm = _chart_fdm(summary_df, fdm_max) if not summary_df.empty else None
    chart_heatmap = _chart_model_weight_heatmap(weights_df) if not weights_df.empty else None
    ticker_pies = {
        ticker: _chart_group_pie(weights_df, ticker, palette)
        for ticker in sorted(weights_df["ticker"].unique())
    } if not weights_df.empty else {}

    # --- KPI row ---
    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi"><div class="val">{n_tickers}</div><div class="lbl">Tickers</div></div>
      <div class="kpi"><div class="val">{total_models}</div><div class="lbl">Total Models</div></div>
      <div class="kpi"><div class="val">{n_groups}</div><div class="lbl">Feature Families</div></div>
      <div class="kpi"><div class="val">{mean_fdm:.3f}</div><div class="lbl">Mean FDM</div></div>
      <div class="kpi"><div class="val">{fdm_max:.1f}</div><div class="lbl">FDM Cap</div></div>
    </div>"""

    # --- Family legend ---
    legend_pills = "".join(
        f'<span class="tag" style="background:{palette[f]}" title="{f}">{_abbrev_family(f)}</span>'
        for f in sorted(palette)
    )
    legend_html = f'<div class="pill-row">{legend_pills}</div>'

    # --- Stacked + FDM charts ---
    charts_row_1 = ""
    if chart_stacked:
        charts_row_1 += f'<div class="card"><h3>Group Weight Allocation</h3><img class="chart" src="data:image/png;base64,{chart_stacked}"></div>'
    if chart_fdm:
        charts_row_1 += f'<div class="card"><h3>Forecast Diversification Multiplier</h3><img class="chart" src="data:image/png;base64,{chart_fdm}"></div>'
    if charts_row_1:
        charts_row_1 = f'<div class="chart-row">{charts_row_1}</div>'

    # --- Per-ticker donuts ---
    donut_cards = "".join(
        f'<div class="card" style="flex:0 0 auto;width:260px"><img class="chart" src="data:image/png;base64,{b64}"></div>'
        for b64 in ticker_pies.values()
    )
    donut_section = f"""
    <h2>Group Allocation per Ticker</h2>
    <div class="chart-row" style="flex-wrap:wrap">{donut_cards}</div>
    """ if donut_cards else ""

    # --- Heatmap ---
    heatmap_section = ""
    if chart_heatmap:
        heatmap_section = f"""
        <h2>Model Weight Heatmap (Cross-Ticker)</h2>
        <p class="section-note">Rows = models (grouped by feature family, separated by dashed lines). Columns = tickers. Colour intensity = allocated weight. Grey = model not in that ticker.</p>
        <div class="card" style="text-align:center"><img class="chart" src="data:image/png;base64,{chart_heatmap}"></div>
        """

    # --- Summary table ---
    summary_section = ""
    if not summary_df.empty:
        summary_section = f"""
        <h2>Ticker Summary</h2>
        <div class="card"><div class="tbl-wrap">{_html_table(summary_df, palette=palette)}</div></div>
        """

    # --- Weights detail table ---
    detail_section = ""
    if not weights_df.empty:
        display_df = weights_df.drop(columns=["phase"], errors="ignore")
        detail_section = f"""
        <h2>Model Weights Detail</h2>
        <p class="section-note">One row per (ticker × model). <em>group_weight</em> = sum of all model weights in that feature family for that ticker.</p>
        <div class="card"><div class="tbl-wrap">{_html_table(display_df, palette=palette)}</div></div>
        """

    # --- Cross-ticker table ---
    cross_section = ""
    if not cross_df.empty:
        display_cross = cross_df.drop(columns=["tickers_present"], errors="ignore")
        cross_section = f"""
        <h2>Cross-Ticker Signal Presence</h2>
        <p class="section-note">Models sorted by number of tickers they appear in (descending). Weight columns show the allocated weight in each ticker (— = not present).</p>
        <div class="card"><div class="tbl-wrap">{_html_table(display_cross, palette=palette)}</div></div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weight Layer Report — {phase_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Weight Layer Report</h1>
  <p class="subtitle">Phase: <strong>{phase_name}</strong> &nbsp;|&nbsp; Method: <strong>{weight_method}</strong> &nbsp;|&nbsp; Generated: {now}</p>
  {kpi_html}
  <h2>Feature Families</h2>
  <div class="card">{legend_html}</div>
  {charts_row_1}
  {donut_section}
  {heatmap_section}
  {summary_section}
  {detail_section}
  {cross_section}
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_weight_layer_report(
    weight_layer: "BaseWeightLayer",
    phase_name: str,
    output_dir: Path,
) -> None:
    """Export weight layer diagnostics to ``output_dir``.

    Parameters
    ----------
    weight_layer:
        A fitted ``BaseWeightLayer`` instance (from ``portfolio.weight_layer``).
    phase_name:
        Label for the pipeline phase, e.g. ``"train"``, ``"validation"``, ``"test"``.
    output_dir:
        Directory where report files are written (created if absent).
    """
    if not weight_layer.is_fitted_:
        logger.warning("WeightLayer is not fitted — skipping report for phase '%s'", phase_name)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = weight_layer.get_diagnostics()

    # --- diagnostics.json ---
    diag_path = output_dir / "diagnostics.json"
    with diag_path.open("w") as f:
        json.dump(_to_json_serializable(diagnostics), f, indent=2)
    logger.info("Weight layer diagnostics written to %s", diag_path)

    # --- data frames ---
    weights_df = _build_weights_by_group(diagnostics, phase_name)
    summary_df = _build_summary(diagnostics, phase_name)
    cross_df = _build_cross_ticker(weights_df) if not weights_df.empty else pd.DataFrame()

    if not weights_df.empty:
        weights_df.to_csv(output_dir / "weights_by_group.csv", index=False)
        logger.info("Weights by group written (%d rows)", len(weights_df))
    if not cross_df.empty:
        cross_df.to_csv(output_dir / "signal_cross_ticker.csv", index=False)
        logger.info("Cross-ticker signal table written (%d rows)", len(cross_df))
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        logger.info("Summary written")

    # --- HTML visual report ---
    html = _build_html_report(weights_df, summary_df, cross_df, diagnostics, phase_name)
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("Visual report written to %s", html_path)


# ---------------------------------------------------------------------------
# Cross-TF weight diagnostics (GlobalPortfolio / GlobalWeightLayer)
# ---------------------------------------------------------------------------

def _build_global_html_report(diagnostics: dict, phase_name: str) -> str:
    """Build a self-contained HTML report for a fitted GlobalWeightLayer."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tf_weights: dict = diagnostics.get("tf_weights", {})
    fdm = float(diagnostics.get("fdm", 1.0))
    mean_corr = diagnostics.get("mean_cross_tf_correlation", float("nan"))
    mean_corr_str = f"{float(mean_corr):.4f}" if mean_corr == mean_corr else "N/A"  # NaN check
    n_tfs = int(diagnostics.get("daily_grid_len", 0))

    # --- Bar chart of TF weights ---
    chart_b64 = ""
    if tf_weights:
        fig, ax = plt.subplots(figsize=(max(4, len(tf_weights) * 1.2), 3))
        tfs = sorted(tf_weights.keys())
        vals = [tf_weights[t] for t in tfs]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(tfs))]
        bars = ax.bar(tfs, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel("Weight")
        ax.set_title("Cross-TF HRP Weights")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        chart_b64 = base64.b64encode(buf.getvalue()).decode()

    chart_html = (
        f'<div class="card" style="text-align:center">'
        f'<img class="chart" src="data:image/png;base64,{chart_b64}"></div>'
        if chart_b64 else ""
    )

    # --- Weights table ---
    rows_html = "".join(
        f"<tr><td>{tf}</td><td>{w:.6f}</td></tr>"
        for tf, w in sorted(tf_weights.items())
    )
    table_html = (
        f'<table><thead><tr><th>Timeframe</th><th>HRP Weight</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Global Weight Layer Report — {phase_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Global Weight Layer Report</h1>
  <p class="subtitle">Phase: <strong>{phase_name}</strong> &nbsp;|&nbsp; Generated: {now}</p>
  <div class="kpi-row">
    <div class="kpi"><div class="val">{len(tf_weights)}</div><div class="lbl">Timeframes</div></div>
    <div class="kpi"><div class="val">{fdm:.3f}</div><div class="lbl">Cross-TF FDM</div></div>
    <div class="kpi"><div class="val">{mean_corr_str}</div><div class="lbl">Mean Downside Corr</div></div>
    <div class="kpi"><div class="val">{n_tfs}</div><div class="lbl">Training Days</div></div>
  </div>
  <h2>Timeframe Weights</h2>
  {chart_html}
  <div class="card"><div class="tbl-wrap">{table_html}</div></div>
</div>
</body>
</html>"""


def export_global_weight_layer_report(
    global_weight_layer: "_GlobalWeightLayer",
    phase_name: str,
    output_dir: Path,
) -> None:
    """Export cross-TF weight diagnostics from a fitted ``GlobalWeightLayer``.

    Writes to ``output_dir``:
    - ``report.html``             — visual HTML report (bar chart + table)
    - ``global_tf_weights.csv``   — per-timeframe weights + FDM summary
    - ``global_diagnostics.json`` — full raw diagnostics dict
    """
    if not getattr(global_weight_layer, "is_fitted_", False):
        logger.warning(
            "GlobalWeightLayer is not fitted — skipping cross-TF report for phase '%s'",
            phase_name,
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = global_weight_layer.get_diagnostics()

    # --- global_diagnostics.json ---
    diag_path = output_dir / "global_diagnostics.json"
    with diag_path.open("w") as f:
        json.dump(_to_json_serializable(diagnostics), f, indent=2)
    logger.info("Global weight layer diagnostics written to %s", diag_path)

    # --- global_tf_weights.csv ---
    tf_weights: dict = diagnostics.get("tf_weights", {})
    if tf_weights:
        rows = [
            {
                "phase": phase_name,
                "timeframe": str(tf),
                "weight": round(float(w), 6),
                "fdm": round(float(diagnostics.get("fdm", 1.0)), 4),
                "mean_cross_tf_correlation": round(
                    float(diagnostics.get("mean_cross_tf_correlation", float("nan"))), 4
                ),
            }
            for tf, w in sorted(tf_weights.items())
        ]
        tf_df = pd.DataFrame(rows)
        csv_path = output_dir / "global_tf_weights.csv"
        tf_df.to_csv(csv_path, index=False)
        logger.info("Global TF weights written to %s", csv_path)

    # --- report.html ---
    html = _build_global_html_report(diagnostics, phase_name)
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("Global weight layer HTML report written to %s", html_path)
