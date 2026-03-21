"""Validate Norgate migration by comparing returns and prices against Kibot backup.

Generates Plotly HTML reports per ticker in docs/library/Data/comparisons/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fetch_norgate_data import TICKER_TO_NORGATE

ROOT = Path(__file__).resolve().parent.parent
OHLC_DIR = ROOT / "data" / "ohlc_data"
BACKUP_DIR = ROOT / "data" / "ohlc_data_kibot_backup"
REPORT_DIR = ROOT / "docs" / "library" / "Data" / "comparisons"


def _load_close(directory: Path, ticker: str) -> pd.Series:
    """Load daily close prices indexed by datetime string."""
    path = directory / ticker / f"D_{ticker}.parquet"
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")["close"].sort_index()


def compare_returns(kibot_close: pd.Series, norgate_close: pd.Series) -> dict:
    """Compare daily returns on overlapping dates."""
    common = kibot_close.index.intersection(norgate_close.index)
    if len(common) < 10:
        return {"error": "Too few overlapping dates", "n_overlapping_days": len(common)}

    kb_ret = kibot_close.loc[common].pct_change().dropna()
    ng_ret = norgate_close.loc[common].pct_change().dropna()

    # Align after pct_change
    common_ret = kb_ret.index.intersection(ng_ret.index)
    kb_ret = kb_ret.loc[common_ret]
    ng_ret = ng_ret.loc[common_ret]

    corr = float(np.corrcoef(kb_ret.values, ng_ret.values)[0, 1])
    diff = kb_ret.values - ng_ret.values

    # Divergent dates: abs return diff > 5%
    divergent_mask = np.abs(diff) > 0.05
    divergent_dates = [str(d.date()) for d in common_ret[divergent_mask]]

    return {
        "return_correlation": round(corr, 6),
        "mean_return_diff": round(float(np.mean(np.abs(diff))), 6),
        "max_return_diff": round(float(np.max(np.abs(diff))), 6),
        "n_overlapping_days": len(common_ret),
        "n_divergent_dates": len(divergent_dates),
        "divergent_dates": divergent_dates[:20],  # cap for readability
    }


def plot_price_overlay(ticker: str, kibot_close: pd.Series, norgate_close: pd.Series) -> go.Figure:
    """Dual y-axes price overlay: Kibot blue, Norgate orange."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=kibot_close.index, y=kibot_close.values, name="Kibot", line=dict(color="blue", width=1)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=norgate_close.index, y=norgate_close.values, name="Norgate", line=dict(color="orange", width=1)),
        secondary_y=True,
    )
    fig.update_layout(title=f"{ticker} — Price Overlay (Kibot vs Norgate)", height=500)
    fig.update_yaxes(title_text="Kibot Close", secondary_y=False)
    fig.update_yaxes(title_text="Norgate Close", secondary_y=True)
    return fig


def plot_return_scatter(ticker: str, kibot_close: pd.Series, norgate_close: pd.Series) -> go.Figure:
    """Scatter of daily returns: Kibot vs Norgate with 45° line."""
    common = kibot_close.index.intersection(norgate_close.index)
    kb_ret = kibot_close.loc[common].pct_change().dropna()
    ng_ret = norgate_close.loc[common].pct_change().dropna()
    common_ret = kb_ret.index.intersection(ng_ret.index)
    kb_ret = kb_ret.loc[common_ret]
    ng_ret = ng_ret.loc[common_ret]

    r2 = float(np.corrcoef(kb_ret.values, ng_ret.values)[0, 1]) ** 2
    rng = max(abs(kb_ret.max()), abs(ng_ret.max()), abs(kb_ret.min()), abs(ng_ret.min()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kb_ret.values, y=ng_ret.values, mode="markers",
        marker=dict(size=3, opacity=0.4, color="steelblue"),
        name=f"R²={r2:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng], mode="lines",
        line=dict(color="red", dash="dash"), name="45° line",
    ))
    fig.update_layout(
        title=f"{ticker} — Return Scatter (R²={r2:.4f})",
        xaxis_title="Kibot Return", yaxis_title="Norgate Return",
        height=500, width=600,
    )
    return fig


def plot_divergence_timeline(ticker: str, kibot_close: pd.Series, norgate_close: pd.Series) -> go.Figure:
    """Price ratio over time — jumps indicate roll adjustment differences."""
    common = kibot_close.index.intersection(norgate_close.index)
    ratio = kibot_close.loc[common] / norgate_close.loc[common]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common, y=ratio.values, mode="lines",
        line=dict(color="purple", width=1), name="Kibot/Norgate ratio",
    ))
    fig.update_layout(
        title=f"{ticker} — Price Ratio (Kibot / Norgate)", height=400,
        yaxis_title="Ratio",
    )
    return fig


def generate_report(ticker: str, kibot_close: pd.Series, norgate_close: pd.Series, report_dir: Path) -> dict:
    """Save Plotly HTMLs and JSON stats for a single ticker."""
    ticker_dir = report_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    stats = compare_returns(kibot_close, norgate_close)

    # Save stats
    with open(ticker_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save plots
    plot_price_overlay(ticker, kibot_close, norgate_close).write_html(str(ticker_dir / "price_overlay.html"))
    plot_return_scatter(ticker, kibot_close, norgate_close).write_html(str(ticker_dir / "return_scatter.html"))
    plot_divergence_timeline(ticker, kibot_close, norgate_close).write_html(str(ticker_dir / "divergence_timeline.html"))

    return stats


def main() -> None:
    if not BACKUP_DIR.exists():
        print(f"ERROR: Kibot backup not found at {BACKUP_DIR}")
        print("Run migrate_norgate_to_ohlc.py first.")
        return

    print(f"Validation reports will be saved to {REPORT_DIR}\n")

    all_stats: dict[str, dict] = {}
    flagged: list[str] = []

    for ticker in sorted(TICKER_TO_NORGATE):
        kibot_path = BACKUP_DIR / ticker / f"D_{ticker}.parquet"
        norgate_path = OHLC_DIR / ticker / f"D_{ticker}.parquet"

        if not kibot_path.exists():
            print(f"  SKIP: {ticker} — no Kibot backup")
            continue
        if not norgate_path.exists():
            print(f"  SKIP: {ticker} — no Norgate data in ohlc_data")
            continue

        kibot_close = _load_close(BACKUP_DIR, ticker)
        norgate_close = _load_close(OHLC_DIR, ticker)

        stats = generate_report(ticker, kibot_close, norgate_close, REPORT_DIR)
        all_stats[ticker] = stats

        corr = stats.get("return_correlation", 0)
        flag = " *** FLAGGED" if corr < 0.999 else ""
        if corr < 0.999:
            flagged.append(ticker)

        print(f"  {ticker}: corr={corr:.6f}  mean_diff={stats.get('mean_return_diff', 'N/A')}  overlap={stats.get('n_overlapping_days', 0)}{flag}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Validated: {len(all_stats)} tickers")
    if flagged:
        print(f"FLAGGED (return corr < 0.999): {flagged}")
    else:
        print("All tickers passed return correlation threshold (>= 0.999)")

    # Save combined stats
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSummary saved to {REPORT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
