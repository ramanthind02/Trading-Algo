from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from ensemble.vault_manager import get_ensemble_tickers
from utils.cache.central_cache import CentralCacheStore
from utils.core.enums import TimeFrame


def source_data_available(ensemble_dir: str) -> bool:
    return all(
        (Path("data/ohlc_data") / ticker.name / f"{timeframe.name}_{ticker.name}.parquet").exists()
        for ticker in get_ensemble_tickers(ensemble_dir)
        for timeframe in (TimeFrame.D, TimeFrame.M)
    )


def instrument_returns_from_cache(
    store: CentralCacheStore,
    tickers: Sequence,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    daily_frames = [
        store.query_candles(ticker, TimeFrame.D, start=start, end=end)
        .reset_index()[["datetime", "ticker", "close"]]
        for ticker in tickers
    ]
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily["datetime"] = pd.to_datetime(all_daily["datetime"]).dt.normalize()
    returns = (
        all_daily.sort_values(["ticker", "datetime"])
        .assign(ret=lambda frame: frame.groupby("ticker")["close"].pct_change())
        .pivot(index="datetime", columns="ticker", values="ret")
        .fillna(0.0)
    )
    returns.columns = [
        str(column.name) if hasattr(column, "name") else str(column) for column in returns.columns
    ]
    return returns
