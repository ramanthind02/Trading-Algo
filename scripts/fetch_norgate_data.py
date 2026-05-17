#!/usr/bin/env python3
"""Fetch Norgate continuous futures with symbol-specific full history."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import norgatedata
import pandas as pd

# Trading-Algo ticker -> Norgate continuous back-adjusted symbol.
TICKER_TO_NORGATE: dict[str, str] = {
    "ES": "&ES_CCB",
    "NQ": "&NQ_CCB",
    "YM": "&YM_CCB",
    "RTY": "&RTY_CCB",
    "CL": "&CL_CCB",
    "HO": "&HO_CCB",
    "GC": "&GC_CCB",
    "HG": "&HG_CCB",
    "SI": "&SI_CCB",
    "PL": "&PL_CCB",
    "EU": "&6E_CCB",
    "JY": "&6J_CCB",
    "BP": "&6B_CCB",
    "CD": "&6C_CCB",
    "SF": "&6S_CCB",
    "C": "&ZC_CCB",
    "S": "&ZS_CCB",
    "W": "&ZW_CCB",
    "GF": "&GF_CCB",
    "TY": "&ZN_CCB",
    "FV": "&ZF_CCB",
    "US": "&ZB_CCB",
    "TU": "&ZT_CCB",
}

# Unadjusted continuous series for roll metadata workflows.
TICKER_TO_NORGATE_RAW: dict[str, str] = {
    key: symbol.replace("_CCB", "")
    for key, symbol in TICKER_TO_NORGATE.items()
}


@dataclass(frozen=True)
class FetchResult:
    ticker: str
    symbol: str
    start_date: str | None
    rows: int


def _resolve_start_date(symbol: str) -> str | None:
    """Resolve the symbol's first quoted date; fallback to provider default history."""
    try:
        start = norgatedata.first_quoted_date(symbol, datetimeformat="iso")
    except Exception:
        return None
    return str(start) if start is not None else None


def _fetch_timeseries(symbol: str, start_date: str | None) -> pd.DataFrame | None:
    params: dict[str, object] = {
        "interval": "D",
        "timeseriesformat": "pandas-dataframe",
        "padding_setting": norgatedata.PaddingType.NONE,
    }
    if start_date is not None:
        params["start_date"] = start_date
    return norgatedata.price_timeseries(symbol, **params)


def fetch_and_save(
    ticker: str,
    norgate_symbol: str,
    output_dir: Path,
    *,
    max_retries: int = 4,
    retry_sleep_s: float = 1.5,
) -> FetchResult | None:
    """Fetch one symbol with retries and write parquet output."""
    start_date = _resolve_start_date(norgate_symbol)
    output_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            df = _fetch_timeseries(norgate_symbol, start_date)
            if df is None or df.empty:
                print(f"  WARN: No data for {ticker} ({norgate_symbol})")
                return None

            df.index.name = "Date"
            out = df.reset_index()
            out_path = output_dir / f"{ticker}.parquet"
            out.to_parquet(out_path, index=False)
            print(
                f"  OK: {ticker:<3} {norgate_symbol:<10} "
                f"start={start_date or 'provider_default'} rows={len(out):>6}"
            )
            return FetchResult(
                ticker=ticker,
                symbol=norgate_symbol,
                start_date=start_date,
                rows=len(out),
            )
        except Exception as exc:
            print(f"  RETRY {attempt}/{max_retries}: {ticker} ({norgate_symbol}) failed: {exc}")
            if attempt == max_retries:
                print(f"  ERR: {ticker} ({norgate_symbol}) exhausted retries")
                return None
            time.sleep(retry_sleep_s * attempt)
    return None


def _run_group(label: str, mapping: dict[str, str], out_dir: Path) -> list[FetchResult]:
    print(f"=== Fetching {label} ===")
    results = [
        result
        for result in (
            fetch_and_save(ticker, symbol, out_dir)
            for ticker, symbol in mapping.items()
        )
        if result is not None
    ]
    print(
        f"  -> {len(results)} symbols written, "
        f"{sum(r.rows for r in results)} total rows\n"
    )
    return results


def _ensure_norgate_running() -> None:
    if not norgatedata.status():
        raise RuntimeError("Norgate Data Updater is not running")


def main() -> None:
    _ensure_norgate_running()

    adj_dir = Path("data/norgate/continuous_futures/adjusted")
    raw_dir = Path("data/norgate/continuous_futures/unadjusted")

    _run_group("back-adjusted continuous futures", TICKER_TO_NORGATE, adj_dir)
    _run_group("unadjusted continuous futures", TICKER_TO_NORGATE_RAW, raw_dir)

    print("Done.")


if __name__ == "__main__":
    main()
