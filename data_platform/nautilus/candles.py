"""Read-back adapter: Nautilus ``ResearchCandle`` -> legacy candle DataFrame.

WP-2 Unit-2, Option B. Mirrors ``data_platform.loaders.load_data`` /
``load_data_multi_ticker`` byte-for-byte by querying the catalog's
:class:`ResearchCandle` records, reconstructing the raw per-ticker frame exactly
as ``pd.read_parquet`` would hand it to the legacy normalizer, then funnelling it
through the **single normalization authority**
:func:`data_platform.loaders._normalize_loaded_frame`. Reusing that function
guarantees identical dtypes, range masking, ``timestamp`` index, and sort order.

Because :class:`ResearchCandle` stores raw ``float64`` (no tick quantization),
the normalizer's ``.astype("float64")`` is a no-op and the output equals the
legacy loader's output exactly (``DataFrame.equals``).
"""
from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd

from nautilus_trader.persistence.catalog import ParquetDataCatalog

from data_platform.loaders import _normalize_loaded_frame
from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.research_candle import (
    ResearchCandle,
    research_instrument_id,
)
from lib.core.enums import Ticker, TimeFrame

# Default open window mirrors data_platform.loaders.load_data.
_DEFAULT_START = datetime(1990, 1, 1)
_DEFAULT_END = datetime(2099, 12, 31)


def _query_candles(
    ticker: Ticker, timeframe: TimeFrame, catalog: ParquetDataCatalog
) -> list[ResearchCandle]:
    """All ResearchCandles for ``(ticker, timeframe)``, unwrapped from CustomData.

    The catalog ``query`` returns ``CustomData`` wrappers ordered by ``ts_init``;
    we pull the inner :class:`ResearchCandle` out of each ``.data`` slot.
    """
    iid = str(research_instrument_id(ticker.name, timeframe.name))
    wrapped = catalog.query(data_cls=ResearchCandle, identifiers=[iid])
    return [getattr(w, "data", w) for w in wrapped]


def _candles_to_raw_frame(candles: list[ResearchCandle]) -> pd.DataFrame:
    """Rebuild the raw ``datetime``+OHLCV frame the legacy normalizer consumes.

    ``ts_event`` is the tz-naive date-label as unix nanoseconds, so
    ``pd.to_datetime(ts_ns)`` reconstructs the exact ``datetime`` the on-disk
    parquet carried. OHLC are already the exact float64 the legacy path produces;
    volume is int64. We hand a ``datetime`` column (not an index) so
    ``_normalize_loaded_frame`` takes its ``"datetime" in df.columns`` branch --
    identical to ``load_data``'s call path.
    """
    datetimes = pd.to_datetime([c.ts_event for c in candles])
    return pd.DataFrame(
        {
            "datetime": datetimes,
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )


def load_data_nautilus(
    ticker: Ticker,
    timeframe: TimeFrame,
    start: datetime = _DEFAULT_START,
    end: datetime = _DEFAULT_END,
    catalog: ParquetDataCatalog | None = None,
) -> pd.DataFrame:
    """Nautilus-backed equivalent of ``data_platform.loaders.load_data``.

    Returns a DataFrame indexed by unix-second ``timestamp`` (int64) with a
    ``datetime`` column and float64 OHLC / int64 volume -- byte-identical to the
    legacy loader for the same ``(ticker, timeframe, start, end)``.

    Raises:
        FileNotFoundError: if no ResearchCandle rows exist for the ticker
            (parity with the legacy loader, which raises when the parquet is
            missing). ``timeframe`` is currently not encoded in the candle's
            instrument id; ingest is expected to populate a single timeframe per
            catalog (see :mod:`data_platform.nautilus.research_ingest`).
    """
    cat = catalog if catalog is not None else get_catalog()
    candles = _query_candles(ticker, timeframe, cat)
    if not candles:
        raise FileNotFoundError(
            f"No ResearchCandle data for ticker={ticker.name} "
            f"timeframe={timeframe.name} in catalog {cat.path}"
        )
    raw = _candles_to_raw_frame(candles)
    return _normalize_loaded_frame(raw, start, end)


def load_data_multi_ticker_nautilus(
    tickers: List[Ticker],
    timeframe: TimeFrame,
    start: datetime = _DEFAULT_START,
    end: datetime = _DEFAULT_END,
    use_millisecond_offset: bool = False,
    catalog: ParquetDataCatalog | None = None,
) -> pd.DataFrame:
    """Nautilus-backed equivalent of ``load_data_multi_ticker``.

    Stacks per-ticker frames with ``ticker`` (Ticker enum object) and
    ``timeframe`` (TimeFrame enum object) columns on a RangeIndex, sorted by
    ``datetime`` -- mirroring the legacy multi-ticker loader exactly.
    """
    cat = catalog if catalog is not None else get_catalog()
    all_dfs = []
    for ticker in tickers:
        ticker_df = load_data_nautilus(
            ticker, timeframe, start=start, end=end, catalog=cat
        ).reset_index()
        ticker_df["ticker"] = ticker
        ticker_df["timeframe"] = timeframe
        all_dfs.append(ticker_df)
    combined = pd.concat(all_dfs, axis=0, ignore_index=True)
    return combined.sort_values("datetime").reset_index(drop=True)
