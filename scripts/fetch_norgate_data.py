"""Fetch Norgate continuous futures data for all mapped tickers."""
from __future__ import annotations

import norgatedata
import pandas as pd
from pathlib import Path

# Kibot ticker -> Norgate back-adjusted continuous symbol
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

# Also fetch unadjusted for roll-date detection via Delivery Month
TICKER_TO_NORGATE_RAW: dict[str, str] = {
    k: v.replace("_CCB", "") for k, v in TICKER_TO_NORGATE.items()
}


def fetch_and_save(
    ticker: str,
    norgate_symbol: str,
    output_dir: Path,
    start_date: str = "2005-01-01",
) -> pd.DataFrame | None:
    try:
        df = norgatedata.price_timeseries(
            norgate_symbol,
            start_date=start_date,
            interval="D",
            timeseriesformat="pandas-dataframe",
        )
        if df is None or df.empty:
            print(f"  WARN: No data for {ticker} ({norgate_symbol})")
            return None

        df.index.name = "Date"
        df = df.reset_index()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{ticker}.parquet"
        df.to_parquet(path, index=False)
        print(f"  OK: {ticker} -> {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ERR: {ticker} ({norgate_symbol}): {e}")
        return None


def main() -> None:
    if not norgatedata.status():
        raise RuntimeError("Norgate Data Updater is not running")

    adj_dir = Path("data/norgate/continuous_futures/adjusted")
    raw_dir = Path("data/norgate/continuous_futures/unadjusted")

    print("=== Fetching back-adjusted continuous futures ===")
    for ticker, sym in TICKER_TO_NORGATE.items():
        fetch_and_save(ticker, sym, adj_dir)

    print("\n=== Fetching unadjusted continuous futures ===")
    for ticker, sym in TICKER_TO_NORGATE_RAW.items():
        fetch_and_save(ticker, sym, raw_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
