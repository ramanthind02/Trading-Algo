"""Fetch Norgate continuous futures series (adjusted + unadjusted) to the working store.

Writes to data/norgate/working/continuous/{adjusted,unadjusted}/{TICKER}.parquet.
These are ephemeral working copies refreshed on every full rebuild; the permanent
archive lives in data/norgate/archive/.

Run directly:
    python -m data_platform.providers.norgate.fetch_continuous
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

try:
    import norgatedata
except ImportError:
    norgatedata = None  # type: ignore[assignment]
import pandas as pd

from data_platform.storage import write_norgate_bars
from ._constants import TICKER_TO_CCB, TICKER_TO_RAW
from ._paths import working_adjusted_dir, working_unadjusted_dir


@dataclass(frozen=True)
class FetchResult:
    ticker: str
    symbol: str
    rows: int
    start_date: str | None


def ensure_norgate_running() -> None:
    if not norgatedata.status():
        raise RuntimeError("Norgate Data Updater is not running — start NDU and retry.")


def _first_quoted_date(symbol: str) -> str | None:
    try:
        d = norgatedata.first_quoted_date(symbol, datetimeformat="iso")
        return str(d) if d is not None else None
    except Exception:
        return None


def _fetch_series(symbol: str, start_date: str | None) -> pd.DataFrame | None:
    params: dict[str, object] = {
        "interval": "D",
        "timeseriesformat": "pandas-dataframe",
        "padding_setting": norgatedata.PaddingType.NONE,
    }
    if start_date is not None:
        params["start_date"] = start_date
    try:
        return norgatedata.price_timeseries(symbol, **params)
    except Exception as exc:
        print(f"    ERR fetching {symbol}: {exc}")
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to canonical schema: date32 index + float32 OHLC + int32 volume."""
    out = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"}).copy()
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    out.index.name = "date"
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].sort_index()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype("int32")
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    write_norgate_bars(df, path, store="norgate_continuous")


def fetch_one(
    ticker: str,
    symbol: str,
    out_dir: Path,
    *,
    max_retries: int = 4,
    retry_sleep_s: float = 1.5,
) -> FetchResult | None:
    start_date = _first_quoted_date(symbol)
    for attempt in range(1, max_retries + 1):
        df = _fetch_series(symbol, start_date)
        if df is not None and not df.empty:
            normalized = _normalize(df)
            _write(normalized, out_dir / f"{ticker}.parquet")
            print(f"  OK  {ticker:<5} {symbol:<12}  rows={len(normalized):>6}  start={start_date or 'provider_default'}")
            return FetchResult(ticker=ticker, symbol=symbol, rows=len(normalized), start_date=start_date)
        if attempt < max_retries:
            print(f"  RETRY {attempt}/{max_retries}: {ticker} ({symbol})")
            time.sleep(retry_sleep_s * attempt)
    print(f"  ERR  {ticker} ({symbol}) exhausted retries")
    return None


def fetch_all(mapping: dict[str, str], out_dir: Path, label: str) -> list[FetchResult]:
    print(f"\n=== {label} -> {out_dir} ===")
    results = [r for r in (fetch_one(tk, sym, out_dir) for tk, sym in mapping.items()) if r]
    print(f"  -> {len(results)}/{len(mapping)} written, {sum(r.rows for r in results):,} rows total")
    return results


def main() -> None:
    ensure_norgate_running()
    fetch_all(TICKER_TO_CCB, working_adjusted_dir(), "back-adjusted continuous (CCB)")
    fetch_all(TICKER_TO_RAW, working_unadjusted_dir(), "unadjusted continuous")
    print("\nDone.")


if __name__ == "__main__":
    main()
