"""Compare back-adjusted legacy data against Norgate continuous futures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils.enums import Ticker

_DEFAULT_REPORT_DIR = Path("docs/library/Data/comparisons")


def _align_daily(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Align two DataFrames on date index, return close series."""
    if "datetime" in old_df.columns:
        old_dates = pd.to_datetime(old_df["datetime"]).dt.normalize()
        old_close = old_df["close"].values if "close" in old_df.columns else old_df["Close"].values
    else:
        old_dates = pd.to_datetime(old_df.index).normalize()
        old_close = old_df["close"].values if "close" in old_df.columns else old_df["Close"].values

    old_s = pd.Series(old_close, index=old_dates, dtype=np.float64, name="old_close")
    old_s = old_s[~old_s.index.duplicated(keep="first")]

    if "Date" in new_df.columns:
        new_dates = pd.to_datetime(new_df["Date"]).dt.normalize()
    elif "datetime" in new_df.columns:
        new_dates = pd.to_datetime(new_df["datetime"]).dt.normalize()
    else:
        new_dates = pd.to_datetime(new_df.index).normalize()

    new_close_col = "Close" if "Close" in new_df.columns else "close"
    new_s = pd.Series(new_df[new_close_col].values, index=new_dates, dtype=np.float64, name="new_close")
    new_s = new_s[~new_s.index.duplicated(keep="first")]

    common = old_s.index.intersection(new_s.index)
    return old_s.loc[common], new_s.loc[common]


def compare_price_levels(
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
) -> Dict[str, float]:
    """Compare price levels between old and new data.
    Returns dict with: correlation, mean_diff, max_divergence, rmse.
    """
    old_s, new_s = _align_daily(old_data, new_data)

    if len(old_s) == 0:
        return {"correlation": 0.0, "mean_diff": 0.0, "max_divergence": 0.0, "rmse": 0.0}

    diff = new_s.values - old_s.values
    return {
        "correlation": float(np.corrcoef(old_s.values, new_s.values)[0, 1]),
        "mean_diff": float(np.mean(diff)),
        "max_divergence": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def compare_roll_dates(
    old_metadata_path: Path,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare roll dates between legacy metadata and Norgate Delivery Month."""
    meta = json.loads(old_metadata_path.read_text())
    old_roll_dates = [pd.Timestamp(d) for d in meta.get("roll_dates", [])]

    new_roll_dates: list[pd.Timestamp] = []
    if "Delivery Month" in new_data.columns:
        date_col = "Date" if "Date" in new_data.columns else "datetime"
        dates = pd.to_datetime(new_data[date_col])
        dm = new_data["Delivery Month"]
        changes = dm != dm.shift(1)
        new_roll_dates = [pd.Timestamp(d) for d in dates[changes].iloc[1:].values]

    rows: list[dict] = []
    for old_dt in old_roll_dates:
        best_new = None
        best_delta = None
        for new_dt in new_roll_dates:
            delta = abs((new_dt - old_dt).days)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_new = new_dt
        rows.append({
            "roll_date_old": old_dt,
            "roll_date_new": best_new,
            "delta_days": best_delta,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["roll_date_old", "roll_date_new", "delta_days"]
    )


def generate_comparison_report(
    ticker: Ticker,
    old_data_path: Path,
    new_data_path: Path,
    metadata_path: Path,
    output_dir: Path = _DEFAULT_REPORT_DIR,
) -> str:
    """Generate a markdown comparison report for a ticker."""
    old_df = pd.read_parquet(old_data_path)
    new_df = pd.read_parquet(new_data_path)

    stats = compare_price_levels(old_df, new_df)
    roll_df = compare_roll_dates(metadata_path, new_df)

    lines = [
        f"# {ticker.name} — Back-Adjustment Comparison Report",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Price Level Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Correlation | {stats['correlation']:.6f} |",
        f"| Mean Difference | {stats['mean_diff']:.2f} |",
        f"| Max Divergence | {stats['max_divergence']:.2f} |",
        f"| RMSE | {stats['rmse']:.2f} |",
        "",
    ]

    if not roll_df.empty:
        lines.extend([
            "## Roll Date Comparison",
            "",
            "| Old Roll Date | New Roll Date | Delta (days) |",
            "|---------------|---------------|--------------|",
        ])
        for _, row in roll_df.iterrows():
            old_str = str(row["roll_date_old"].date()) if pd.notna(row["roll_date_old"]) else "N/A"
            new_str = str(row["roll_date_new"].date()) if pd.notna(row["roll_date_new"]) else "N/A"
            delta_str = str(row["delta_days"]) if pd.notna(row["delta_days"]) else "N/A"
            lines.append(f"| {old_str} | {new_str} | {delta_str} |")
        lines.append("")

    lines.extend([
        "## Summary",
        "",
        f"- Correlation of {stats['correlation']:.4f} between legacy back-adjusted and Norgate data.",
        f"- Mean price difference: {stats['mean_diff']:.2f} points.",
        f"- {len(roll_df)} roll dates compared.",
        "",
    ])

    report = "\n".join(lines)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{ticker.name}_comparison.md"
    report_path.write_text(report)

    return report
