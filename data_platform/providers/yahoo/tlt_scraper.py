from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf

# Ensure project root (parent of this directory) is on sys.path for package imports
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from utils.core.enums import TimeFrame, Ticker


def _get_project_root() -> Path:
    """Locate the project root by searching for pyproject.toml upwards from this file."""
    current: Path = Path(__file__).resolve()
    return next(
        (p for p in current.parents if (p / "pyproject.toml").exists()),
        current.parents[1],
    )


def fetch_tlt_from_yahoo() -> pd.DataFrame:
    """Fetch full-history daily TLT data from Yahoo Finance using yfinance.

    Returns:
        pd.DataFrame: Raw Yahoo data with a Date column and original Yahoo column names.

    Raises:
        ValueError: If Yahoo returns an empty DataFrame.
    """
    df: pd.DataFrame = yf.download(
        tickers="TLT",
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance for TLT.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "Date"
    return df.reset_index()


def save_raw_tlt_csv(df_raw: pd.DataFrame) -> Path:
    """Save the raw Yahoo TLT data to CSV under data/raw_data.

    Args:
        df_raw: Raw TLT dataframe as returned by fetch_tlt_from_yahoo.

    Returns:
        Path: Path to the written CSV file.
    """
    project_root = _get_project_root()
    output_dir = project_root / "data" / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "TLT_raw.csv"
    df_raw.to_csv(output_path, index=False)
    return output_path


def to_internal_daily_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Yahoo TLT data into the internal daily OHLC schema.

    The internal schema matches other parquet data used by load_data:
    - datetime: string-formatted date (YYYY-MM-DD)
    - timestamp: integer seconds since epoch
    - open, high, low, close, volume: price/volume columns

    Args:
        df_raw: Raw TLT dataframe with a Date column and Yahoo OHLCV columns.

    Returns:
        pd.DataFrame: Dataframe in internal daily OHLC schema.
    """
    # yfinance may return a MultiIndex with (field, ticker). Flatten to field name.
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_flat = df_raw.copy()
        df_flat.columns = [
            col[0] if isinstance(col, tuple) and len(col) > 0 else str(col)
            for col in df_flat.columns
        ]
    else:
        df_flat = df_raw

    if "Date" not in df_flat.columns:
        raise ValueError("Expected 'Date' column in raw Yahoo dataframe.")

    df = df_flat.copy()
    df["datetime"] = pd.to_datetime(df["Date"])

    # Use adjusted close for our close series
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    rename_map: dict[str, str] = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    df["timestamp"] = df["datetime"].astype("int64") // 10**9
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

    base_columns: list[str] = ["datetime", "timestamp", "open", "high", "low", "close"]
    if "volume" in df.columns:
        base_columns.append("volume")

    return df[base_columns]



_TIMEFRAME_TO_PANDAS_RULE: Final[dict[TimeFrame, str]] = {
    TimeFrame.D: "D",
    TimeFrame.W: "W",
    TimeFrame.M: "M",
}


def _aggregate_for_timeframe(
    df_indexed: pd.DataFrame,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    """Aggregate a datetime-indexed daily dataframe to the requested timeframe."""
    rule = _TIMEFRAME_TO_PANDAS_RULE[timeframe]

    if rule == "D":
        return df_indexed.copy()

    available_cols = set(df_indexed.columns)

    agg_spec: dict[str, str] = {}
    if "open" in available_cols:
        agg_spec["open"] = "first"
    if "high" in available_cols:
        agg_spec["high"] = "max"
    if "low" in available_cols:
        agg_spec["low"] = "min"
    if "close" in available_cols:
        agg_spec["close"] = "last"
    if "volume" in available_cols:
        agg_spec["volume"] = "sum"

    if not agg_spec:
        raise ValueError(
            "No OHLCV columns found when aggregating TLT data; "
            f"available columns: {sorted(available_cols)}"
        )

    return df_indexed.resample(rule).agg(agg_spec).dropna(how="any")


def write_ohlc_parquet_from_daily(daily_df: pd.DataFrame) -> dict[TimeFrame, Path]:
    """Write daily, weekly, and monthly OHLC parquet files for TLT.

    Args:
        daily_df: Internal daily OHLC dataframe produced by to_internal_daily_schema.

    Returns:
        dict[TimeFrame, Path]: Mapping from timeframe to written parquet path.
    """
    project_root = _get_project_root()
    output_dir = project_root / "data" / "ohlc_data" / Ticker.TLT.name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = daily_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_indexed = df.set_index("datetime")

    results: dict[TimeFrame, Path] = {}
    for timeframe in (TimeFrame.D, TimeFrame.W, TimeFrame.M):
        aggregated = _aggregate_for_timeframe(df_indexed, timeframe)
        aggregated = aggregated.reset_index()

        aggregated["timestamp"] = aggregated["datetime"].astype("int64") // 10**9
        aggregated["datetime"] = aggregated["datetime"].dt.strftime("%Y-%m-%d")

        output_path = output_dir / f"{timeframe.name}_{Ticker.TLT.name}.parquet"
        aggregated.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )
        results[timeframe] = output_path

    return results


def build_tlt_from_yahoo() -> None:
    """Orchestrate fetching TLT from Yahoo and writing all parquet outputs."""
    raw_df = fetch_tlt_from_yahoo()
    save_raw_tlt_csv(raw_df)

    daily_internal = to_internal_daily_schema(raw_df)
    write_ohlc_parquet_from_daily(daily_internal)


if __name__ == "__main__":
    build_tlt_from_yahoo()

