"""Z-scored price spread between two tickers.

Demonstrates the cross-ticker bias-node pattern: the primary ticker's
candles are streamed via the normal ``add_candle()`` path, while the
secondary ticker's data is fetched from the
:class:`~utils.data.cross_ticker_store.CrossTickerDataStore` singleton.

Example
-------
>>> from utils.data.cross_ticker_store import CrossTickerDataStore
>>> store = CrossTickerDataStore.get_instance()
>>> store.load(Ticker.NQ, TimeFrame.D)
>>>
>>> node = SpreadNode(Ticker.ES, TimeFrame.D, cross_tickers=['NQ'], lookback=20)
>>> signal = node.add_candle(es_candle)  # internally fetches NQ close
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

import numpy as np

from nodes import BiasNode
from utils.cache.central_cache_errors import ArtifactMissingError
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore


class SpreadNode(BiasNode):
    """Z-scored price spread between the primary ticker and a cross-ticker.

    The node computes ``primary_close - secondary_close`` each bar, then
    returns the rolling z-score of that spread over *lookback* bars.

    Parameters
    ----------
    ticker : Ticker
        Primary ticker (candles streamed via ``add_candle``).
    tf : TimeFrame
        Timeframe.
    cross_tickers : list[str] | None
        Secondary ticker dependencies from the canonical bias-node params
        contract. This node uses the first ticker in the list.
    lookback : int
        Rolling window for z-score calculation (default ``20``).
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        cross_tickers: list[str] | None = None,
        lookback: int = 20,
    ) -> None:
        super().__init__(ticker, tf)

        configured_cross = cross_tickers or ["NQ"]
        if not configured_cross:
            raise ValueError("cross_tickers must include at least one ticker symbol.")
        normalized_cross: list[str] = []
        for idx, raw in enumerate(configured_cross):
            if not isinstance(raw, str):
                raise TypeError(
                    f"cross_tickers[{idx}] must be str, got {type(raw).__name__}."
                )
            normalized = raw.strip().upper()
            if normalized:
                normalized_cross.append(normalized)
        if not normalized_cross:
            raise ValueError("cross_tickers must include at least one non-empty ticker symbol.")
        cross_ticker_name = normalized_cross[0]
        try:
            self.cross_ticker = Ticker[cross_ticker_name]
        except KeyError as exc:
            raise ValueError(f"Unknown cross ticker '{cross_ticker_name}'.") from exc

        self.cross_tickers = tuple(normalized_cross)
        self.lookback = lookback

        self.module_name = "spread"
        self.output_features = ["signal"]
        self.params = {"cross_tickers": list(self.cross_tickers), "lookback": lookback}
        self.front_bad = lookback

        self._spread_history: deque[float] = deque(maxlen=lookback)
        self._store = CrossTickerDataStore.get_instance()

        self.ensure_standardized_columns()

    # ------------------------------------------------------------------

    def _compute_candle(self, candle: Candle) -> List[float]:
        primary_close = candle.close

        try:
            secondary = self._store.query_candle(
                self.cross_ticker,
                candle.tf,
                candle.datetime,
            )
        except ArtifactMissingError:
            # Secondary data missing for this bar — return neutral
            return [0.0]

        spread = primary_close - secondary.close
        self._spread_history.append(spread)

        if len(self._spread_history) < self.lookback:
            return [0.0]

        mean = float(np.mean(self._spread_history))
        std = float(np.std(self._spread_history))
        if std < 1e-10:
            return [0.0]

        zscore = (spread - mean) / std
        return [zscore]
