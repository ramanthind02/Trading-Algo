"""Build the canonical ohlc_data store from Norgate working continuous files.

Reads from data/norgate/working/continuous/{adjusted,unadjusted}/ and writes:
  data/ohlc_data/{TICKER}/D_{TICKER}.parquet        daily, back-adjusted
  data/ohlc_data/{TICKER}/W_{TICKER}.parquet        weekly (W-SUN)
  data/ohlc_data/{TICKER}/M_{TICKER}.parquet        monthly (ME)
  data/ohlc_data/{TICKER}/D_{TICKER}_unadj.parquet  daily, unadjusted (σ denominator)

Schema (all files)
  date    date32 (parquet native DATE, no time component)  ← index
  open    float32
  high    float32
  low     float32
  close   float32
  volume  int32

No redundant timestamp column.  Callers that need a unix epoch can compute
it at read time: (pd.to_datetime(df["date"]).astype("int64") // 10**9).

Run directly:
    python -m data_platform.providers.norgate.migrate
    python -m data_platform.providers.norgate.migrate --tickers ES NQ
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._constants import PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL, TICKER_TO_CCB
from ._paths import ohlc_ticker_dir, working_adjusted_dir, working_unadjusted_dir


@dataclass(frozen=True)
class MigrateResult:
    ticker: str
    daily_rows: int
    weekly_rows: int
    monthly_rows: int
    start: str
    end: str


# ── schema helpers ────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Norgate raw columns to canonical schema with date32 index."""
    out = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    }).copy()

    # Accept either a Date column or a DatetimeIndex
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.set_index("date")
    else:
        out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
        out.index.name = "date"

    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "volume" not in out.columns:
        out["volume"] = 0

    out = out[["open", "high", "low", "close", "volume"]].copy()
    out = out.sort_index().loc[~out.index.duplicated(keep="last")]

    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float32")
    out["volume"] = out["volume"].fillna(0).astype("int32")
    return out


def _aggregate(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (
        daily.resample(freq).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )
    agg.index.name = "date"
    for col in ("open", "high", "low", "close"):
        agg[col] = agg[col].astype("float32")
    agg["volume"] = agg["volume"].astype("int32")
    return agg


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=True),
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


# ── per-ticker migration ──────────────────────────────────────────────────

def migrate_ticker(ticker: str) -> MigrateResult:
    adj_src = working_adjusted_dir() / f"{ticker}.parquet"
    if not adj_src.exists():
        raise FileNotFoundError(f"Missing working adjusted parquet: {adj_src}")

    raw_df = pd.read_parquet(adj_src)
    daily = _normalize(raw_df)
    weekly = _aggregate(daily, "W-SUN")
    monthly = _aggregate(daily, "ME")

    tdir = ohlc_ticker_dir(ticker)
    _write(daily,   tdir / f"D_{ticker}.parquet")
    _write(weekly,  tdir / f"W_{ticker}.parquet")
    _write(monthly, tdir / f"M_{ticker}.parquet")

    # Unadjusted daily — used as σ denominator in EWSD to correct the
    # back-adjustment level inflation (k = P_true/P_adj ≈ 0.69 for ES).
    unadj_src = working_unadjusted_dir() / f"{ticker}.parquet"
    if unadj_src.exists():
        unadj_daily = _normalize(pd.read_parquet(unadj_src))
        _write(unadj_daily, tdir / f"D_{ticker}_unadj.parquet")
    else:
        print(f"  WARN  {ticker}: unadjusted source missing, skipping D_{ticker}_unadj.parquet")

    return MigrateResult(
        ticker=ticker,
        daily_rows=len(daily),
        weekly_rows=len(weekly),
        monthly_rows=len(monthly),
        start=str(daily.index.min().date()),
        end=str(daily.index.max().date()),
    )


def migrate_all(tickers: list[str] | None = None) -> list[MigrateResult]:
    targets = tickers or sorted(TICKER_TO_CCB)
    results = []
    for ticker in targets:
        result = migrate_ticker(ticker)
        results.append(result)
        print(
            f"  {result.ticker:<5}  D={result.daily_rows:>5}  W={result.weekly_rows:>4}"
            f"  M={result.monthly_rows:>3}  ({result.start} -> {result.end})"
        )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main(tickers: list[str] | None = None) -> None:
    print("=== Norgate -> ohlc_data migration ===")
    results = migrate_all(tickers)
    print(f"\nDone. {len(results)} tickers migrated.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Migrate Norgate continuous series to ohlc_data.")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers to migrate (default: all).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(tickers=args.tickers)
