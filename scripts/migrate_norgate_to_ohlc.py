#!/usr/bin/env python3
"""Build repository candle store from Norgate-only continuous futures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.fetch_norgate_data import TICKER_TO_NORGATE

ROOT = Path(__file__).resolve().parent.parent
NORGATE_ADJ_DIR = ROOT / "data" / "norgate" / "continuous_futures" / "adjusted"
OHLC_DIR = ROOT / "data" / "ohlc_data"


def _normalize_daily_frame(norgate_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Norgate daily columns to Trading-Algo candle schema."""
    df = norgate_df.rename(
        columns={
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).copy()
    required = ["datetime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from Norgate frame: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_localize(None)
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float64")
    out["volume"] = out["volume"].fillna(0).astype("int64")

    out["timestamp"] = (out["datetime"].astype("int64") // 10**9).astype("int64")
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _aggregate(daily_df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    dt = pd.to_datetime(daily_df["datetime"])
    agg = (
        daily_df.assign(_dt=dt)
        .set_index("_dt")
        .resample(frequency)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open"]) 
        .reset_index()
    )
    agg["datetime"] = agg["_dt"].dt.strftime("%Y-%m-%d")
    agg["timestamp"] = (agg["_dt"].astype("int64") // 10**9).astype("int64")
    agg = agg.drop(columns=["_dt"])
    for col in ("open", "high", "low", "close"):
        agg[col] = agg[col].astype("float64")
    agg["volume"] = agg["volume"].astype("int64")
    return agg[["datetime", "open", "high", "low", "close", "volume", "timestamp"]].reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="fastparquet")


def rebuild_ticker(ticker: str) -> dict[str, object]:
    src = NORGATE_ADJ_DIR / f"{ticker}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing Norgate adjusted parquet: {src}")

    norgate_df = pd.read_parquet(src)
    daily = _normalize_daily_frame(norgate_df)
    weekly = _aggregate(daily, "W-SUN")
    monthly = _aggregate(daily, "ME")

    tdir = OHLC_DIR / ticker
    _write(daily, tdir / f"D_{ticker}.parquet")
    _write(weekly, tdir / f"W_{ticker}.parquet")
    _write(monthly, tdir / f"M_{ticker}.parquet")

    return {
        "ticker": ticker,
        "daily": len(daily),
        "weekly": len(weekly),
        "monthly": len(monthly),
        "start": daily["datetime"].iloc[0],
        "end": daily["datetime"].iloc[-1],
    }


def main() -> None:
    results = [rebuild_ticker(ticker) for ticker in sorted(TICKER_TO_NORGATE)]
    print("=" * 60)
    print("Norgate-only repository candle rebuild complete")
    for row in results:
        print(
            f"  {row['ticker']}: D={row['daily']} W={row['weekly']} M={row['monthly']} "
            f"({row['start']} -> {row['end']})"
        )


if __name__ == "__main__":
    main()
