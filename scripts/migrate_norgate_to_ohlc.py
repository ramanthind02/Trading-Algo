"""Migrate ohlc_data from Kibot to Norgate back-adjusted continuous futures.

Reads daily Norgate parquets, converts to Kibot schema, generates W/M
aggregates, and writes to data/ohlc_data/. Skips tickers not available
in Norgate (NG, TLT). Backs up existing Kibot data first.

Hybrid splice: Kibot data before Norgate start date is preserved so we
don't lose pre-2005 history. Norgate data replaces everything from its
first available date onward.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fetch_norgate_data import TICKER_TO_NORGATE

ROOT = Path(__file__).resolve().parent.parent
NORGATE_DIR = ROOT / "data" / "norgate" / "continuous_futures" / "adjusted"
OHLC_DIR = ROOT / "data" / "ohlc_data"
BACKUP_DIR = ROOT / "data" / "ohlc_data_kibot_backup"

KIBOT_ONLY = {"NG", "TLT"}


def backup_existing_data(ohlc_dir: Path, backup_dir: Path) -> None:
    """Back up existing ohlc_data to a sibling directory. Skip if already done."""
    if backup_dir.exists():
        print(f"Backup already exists at {backup_dir}, skipping.")
        return
    print(f"Backing up {ohlc_dir} -> {backup_dir} ...")
    shutil.copytree(ohlc_dir, backup_dir)
    print("Backup complete.")


def convert_norgate_daily(norgate_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Norgate daily DataFrame to Kibot schema.

    Norgate columns: Date, Open, High, Low, Close, Volume, Delivery Month, Open Interest
    Kibot columns:   datetime, open, high, low, close, volume, timestamp
    """
    df = norgate_df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    # Cast to match Kibot dtypes
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("int64")

    # datetime as string YYYY-MM-DD
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

    # Unix timestamp (UTC midnight)
    df["timestamp"] = (
        pd.to_datetime(df["datetime"]).astype("int64") // 10**9
    ).astype("int64")

    return df.sort_values("timestamp").reset_index(drop=True)


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily candles to weekly (W-SUN) matching Kibot convention."""
    df = daily_df.copy()
    df["_dt"] = pd.to_datetime(df["datetime"])
    df = df.set_index("_dt")

    weekly = df.resample("W-SUN").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    weekly = weekly.reset_index()
    weekly["datetime"] = weekly["_dt"].dt.strftime("%Y-%m-%d")
    weekly["timestamp"] = (
        weekly["_dt"].astype("int64") // 10**9
    ).astype("int64")
    weekly = weekly.drop(columns=["_dt"])

    for col in ("open", "high", "low", "close"):
        weekly[col] = weekly[col].astype("float64")
    weekly["volume"] = weekly["volume"].astype("int64")

    return weekly[["datetime", "open", "high", "low", "close", "volume", "timestamp"]].reset_index(drop=True)


def aggregate_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily candles to monthly (ME = month-end) matching Kibot convention."""
    df = daily_df.copy()
    df["_dt"] = pd.to_datetime(df["datetime"])
    df = df.set_index("_dt")

    monthly = df.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    monthly = monthly.reset_index()
    monthly["datetime"] = monthly["_dt"].dt.strftime("%Y-%m-%d")
    monthly["timestamp"] = (
        monthly["_dt"].astype("int64") // 10**9
    ).astype("int64")
    monthly = monthly.drop(columns=["_dt"])

    for col in ("open", "high", "low", "close"):
        monthly[col] = monthly[col].astype("float64")
    monthly["volume"] = monthly["volume"].astype("int64")

    return monthly[["datetime", "open", "high", "low", "close", "volume", "timestamp"]].reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame as parquet with fastparquet engine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="fastparquet")


def migrate_ticker(ticker: str, norgate_path: Path, ohlc_dir: Path, backup_dir: Path) -> dict:
    """Migrate a single ticker with hybrid splice.

    Pre-Norgate Kibot data is preserved, then Norgate data replaces
    everything from its first available date onward.
    """
    norgate_df = pd.read_parquet(norgate_path)
    norgate_daily = convert_norgate_daily(norgate_df)

    # Determine Norgate start date for splice cutoff
    norgate_start = norgate_daily["datetime"].iloc[0]  # string YYYY-MM-DD

    # Load Kibot backup daily data (pre-migration original)
    kibot_path = backup_dir / ticker / f"D_{ticker}.parquet"
    kibot_rows = 0
    if kibot_path.exists():
        kibot_df = pd.read_parquet(kibot_path)
        # Keep only Kibot rows strictly before Norgate start date
        kibot_pre = kibot_df[kibot_df["datetime"] < norgate_start].copy()
        kibot_rows = len(kibot_pre)

        if kibot_rows > 0:
            # Splice: Kibot pre-2005 + Norgate 2005+
            daily = pd.concat([kibot_pre, norgate_daily], ignore_index=True)
        else:
            daily = norgate_daily
    else:
        daily = norgate_daily

    daily = daily.sort_values("timestamp").reset_index(drop=True)

    weekly = aggregate_to_weekly(daily)
    monthly = aggregate_to_monthly(daily)

    ticker_dir = ohlc_dir / ticker
    _write_parquet(daily, ticker_dir / f"D_{ticker}.parquet")
    _write_parquet(weekly, ticker_dir / f"W_{ticker}.parquet")
    _write_parquet(monthly, ticker_dir / f"M_{ticker}.parquet")

    return {
        "ticker": ticker,
        "daily_rows": len(daily),
        "weekly_rows": len(weekly),
        "monthly_rows": len(monthly),
        "kibot_pre_rows": kibot_rows,
        "date_range": f"{daily['datetime'].iloc[0]} to {daily['datetime'].iloc[-1]}",
    }


def main() -> None:
    backup_existing_data(OHLC_DIR, BACKUP_DIR)

    results: list[dict] = []
    skipped: list[str] = []

    for ticker in sorted(TICKER_TO_NORGATE):
        norgate_path = NORGATE_DIR / f"{ticker}.parquet"
        if not norgate_path.exists():
            print(f"  SKIP: {ticker} — Norgate file not found at {norgate_path}")
            skipped.append(ticker)
            continue

        summary = migrate_ticker(ticker, norgate_path, OHLC_DIR, BACKUP_DIR)
        results.append(summary)
        kibot_note = f" (Kibot pre-splice: {summary['kibot_pre_rows']})" if summary["kibot_pre_rows"] > 0 else ""
        print(f"  OK: {ticker} — D:{summary['daily_rows']} W:{summary['weekly_rows']} M:{summary['monthly_rows']} ({summary['date_range']}){kibot_note}")

    print(f"\n{'='*60}")
    print(f"Migrated: {len(results)} tickers")
    print(f"Skipped (no Norgate): {skipped}")
    print(f"Kibot-only (untouched): {sorted(KIBOT_ONLY)}")
    print(f"\nBackup at: {BACKUP_DIR}")
    print("Remember to clear the cache (utils/cache/) after verifying.")


if __name__ == "__main__":
    main()
