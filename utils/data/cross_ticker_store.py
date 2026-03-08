"""Cross-ticker data store for multi-asset bias nodes.

Provides a singleton service that lets any bias node look up another ticker's
candle at the current bar.  Data can be pre-loaded in bulk (backtest / EDA) or
auto-loaded from parquet on first access (lazy loading).

Typical lifecycle
-----------------
1. **Orchestrator** optionally pre-loads for performance::

       store = CrossTickerDataStore.get_instance()
       store.load(Ticker.NQ, TimeFrame.D, start, end)

2. **Bias node** queries inside ``_compute_candle``::

       nq = self._store.get_candle(Ticker.NQ, candle.tf, candle.datetime)

   If NQ data wasn't pre-loaded, ``get_candle`` auto-loads from parquet.

3. After the run, call ``store.clear()`` or ``CrossTickerDataStore.reset()``.

Loading modes
-------------
- **Pre-load from filesystem** (backtest/EDA/WF/training): ``store.load()``
- **Set from DataFrame** (testing/permutation): ``store.set_data()``
- **Lazy auto-load**: if ``get_candle`` is called for data that hasn't been
  loaded yet, it calls ``load()`` automatically on first access.
- **Live trading**: call ``store.set_data(ticker, tf, api_df)`` with
  data fetched from your broker API (IB, MT5, etc.) before the forecast run.

Bias-node params contract
-------------------------
- Use exactly one key for cross-ticker dependencies: ``cross_tickers``.
- Type must be ``list[str]`` of ticker symbols (e.g. ``["NQ", "GC"]``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, ClassVar, Dict, Mapping, Optional

import pandas as pd

from utils.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)

CROSS_TICKERS_PARAM_KEY = "cross_tickers"


def extract_cross_ticker_names(params: Mapping[str, Any]) -> set[str]:
    """Return normalized cross-ticker names from canonical params contract.

    Canonical contract:
        params["cross_tickers"] -> list[str]
    """
    raw = params.get(CROSS_TICKERS_PARAM_KEY)
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise TypeError(
            f"'{CROSS_TICKERS_PARAM_KEY}' must be list[str], got {type(raw).__name__}."
        )

    names: set[str] = set()
    for idx, value in enumerate(raw):
        if not isinstance(value, str):
            raise TypeError(
                f"'{CROSS_TICKERS_PARAM_KEY}[{idx}]' must be str, got {type(value).__name__}."
            )
        normalized = value.strip().upper()
        if normalized:
            names.add(normalized)
    return names


class CrossTickerDataStore:
    """Singleton service providing cross-ticker candle lookups.

    Internal storage is ``dict[Ticker, dict[TimeFrame, pd.DataFrame]]``
    where each DataFrame is **datetime-indexed** with columns
    ``open, high, low, close, volume``.
    """

    _instance: ClassVar[Optional["CrossTickerDataStore"]] = None

    def __init__(self) -> None:
        self._data: Dict[Ticker, Dict[TimeFrame, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CrossTickerDataStore":
        """Return the global singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful in tests)."""
        if cls._instance is not None:
            cls._instance._data.clear()
        cls._instance = None

    # ------------------------------------------------------------------
    # Loading API
    # ------------------------------------------------------------------

    def load(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        """Load OHLCV data from parquet files on the filesystem.

        Reads ``data/ohlc_data/{ticker}/{tf}_{ticker}.parquet`` via
        ``helpers.load_data``.
        """
        import utils.core.helpers as helpers

        kwargs: dict = {"ticker": ticker, "timeframe": tf}
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        df = helpers.load_data(**kwargs)
        self.set_data(ticker, tf, df)

    def set_data(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        df: pd.DataFrame,
    ) -> None:
        """Set cross-ticker data from an existing DataFrame.

        Use for **live trading** (DataFrame built from broker API response)
        or any case where you have data that didn't come from parquet.
        """
        if ticker not in self._data:
            self._data[ticker] = {}
        if df.index.name != "datetime" and "datetime" in df.columns:
            df = df.set_index("datetime")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        self._data[ticker][tf] = df
        logger.debug(
            "CrossTickerDataStore: loaded %s %s (%d rows)",
            ticker.name, tf.name, len(df),
        )

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def get_candle(
        self, ticker: Ticker, tf: TimeFrame, dt: datetime
    ) -> Optional["Candle"]:
        """Return a :class:`Candle` at exact *dt*, or ``None`` if missing.

        If *(ticker, tf)* data has not been loaded yet, this method
        **auto-loads from parquet** on first access (lazy loading).
        Returns ``None`` only if the datetime is genuinely absent.
        """
        # Lazy-load on first access
        if not self.is_loaded(ticker, tf):
            try:
                logger.info(
                    "CrossTickerDataStore: auto-loading %s %s (lazy)",
                    ticker.name, tf.name,
                )
                self.load(ticker, tf)
            except Exception as e:
                logger.warning(
                    "CrossTickerDataStore: auto-load failed for %s %s: %s",
                    ticker.name, tf.name, e,
                )
                return None

        df = self._get_df(ticker, tf)
        if df is None:
            return None

        lookup = pd.Timestamp(dt)
        if lookup.tzinfo is not None:
            lookup = lookup.tz_localize(None)
        if lookup not in df.index:
            return None

        row = df.loc[lookup]
        from utils.core.models import Candle

        return Candle(
            datetime=dt, ticker=ticker, tf=tf,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_loaded(self, ticker: Ticker, tf: TimeFrame) -> bool:
        """Check whether data for *(ticker, tf)* has been loaded."""
        return ticker in self._data and tf in self._data[ticker]

    def loaded_tickers(self) -> list[Ticker]:
        """Return list of tickers currently in the store."""
        return list(self._data.keys())

    def clear(self) -> None:
        """Remove all loaded data (keep singleton alive)."""
        self._data.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_df(
        self, ticker: Ticker, tf: TimeFrame
    ) -> Optional[pd.DataFrame]:
        return self._data.get(ticker, {}).get(tf)
