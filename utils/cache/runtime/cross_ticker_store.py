"""Central-cache-owned cross-ticker lookup adapter.

This is the canonical implementation location for cross-ticker candle access.
Compatibility re-exports may exist in older module paths, but the logic lives
under ``utils/cache`` so cache-related behavior is easy to find.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, ClassVar, Mapping, Optional

import pandas as pd

from .central_cache import CentralCacheStore
from .central_cache_errors import ArtifactMissingError
from .central_cache_models import LookupMode
from utils.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)

CROSS_TICKERS_PARAM_KEY = "cross_tickers"

# Params whose values are inherently list[str] and must never be unwrapped
# during grid expansion or normalization.
SCALAR_LIST_PARAM_KEYS: frozenset[str] = frozenset({CROSS_TICKERS_PARAM_KEY})


def extract_cross_ticker_names(params: Mapping[str, Any]) -> set[str]:
    """Return normalized cross-ticker names from canonical params contract."""
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
    """Singleton service providing cross-ticker candle lookups."""

    _instance: ClassVar[Optional["CrossTickerDataStore"]] = None

    @property
    def _central_cache(self) -> CentralCacheStore:
        return CentralCacheStore.get_instance()

    @classmethod
    def get_instance(cls) -> "CrossTickerDataStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton while only purging cached candle state."""
        CentralCacheStore.get_instance().clear_candles(purge_persisted=True)
        cls._instance = None

    def load(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        """Load OHLCV data from repository-backed parquet files."""
        import utils.core.helpers as helpers

        kwargs: dict[str, object] = {"ticker": ticker, "timeframe": tf}
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
        """Set cross-ticker data from an existing DataFrame."""
        self._central_cache.set_candles(ticker, tf, df)
        logger.debug(
            "CrossTickerDataStore: loaded %s %s (%d rows)",
            ticker.name,
            tf.name,
            len(df),
        )

    def query_candle(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        dt: datetime,
        lookup_mode: LookupMode = LookupMode.EXACT,
    ) -> "Candle":
        """Return a candle or raise a typed miss error."""
        return self._central_cache.query_candle(ticker, tf, dt, lookup_mode=lookup_mode)

    def get_candle(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        dt: datetime,
    ) -> Optional["Candle"]:
        """Compatibility wrapper that preserves the historical None-on-miss fallback."""
        try:
            return self.query_candle(ticker, tf, dt)
        except ArtifactMissingError:
            return None

    def is_loaded(self, ticker: Ticker, tf: TimeFrame) -> bool:
        return self._central_cache.is_candle_loaded(ticker, tf)

    def loaded_tickers(self) -> list[Ticker]:
        return self._central_cache.loaded_tickers()

    def clear(self) -> None:
        """Remove only loaded candle data owned by this adapter."""
        self._central_cache.clear_candles(purge_persisted=True)

    def get_candle_as_of(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        dt: datetime,
    ) -> Optional["Candle"]:
        """Compatibility helper for backward-looking lookups."""
        try:
            return self.query_candle(ticker, tf, dt, lookup_mode=LookupMode.AS_OF)
        except ArtifactMissingError:
            return None


__all__ = [
    "CROSS_TICKERS_PARAM_KEY",
    "CrossTickerDataStore",
    "SCALAR_LIST_PARAM_KEYS",
    "extract_cross_ticker_names",
]
