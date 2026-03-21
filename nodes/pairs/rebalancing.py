"""Monthly rebalancing pairs-trading node.

Compares percentage gains of the primary ticker and a cross-ticker over the
first 15 calendar days of each month and generates a directional signal for
the primary ticker:

- **Cross-ticker outperforms** → long primary ticker from day 15 to end of
  month (signal = 1).
- **Primary ticker outperforms** → flat during month, then long primary ticker
  from day 25 through the 5th trading day of the *next* month (signal = 1 in
  that window).
- **Observation period (days 1–15)** → signal = 0.

Example
-------
>>> from utils.data.cross_ticker_store import CrossTickerDataStore
>>> store = CrossTickerDataStore.get_instance()
>>> store.load(Ticker.TLT, TimeFrame.D)
>>>
>>> node = RebalancingNode(Ticker.ES, TimeFrame.D, cross_tickers=['TLT'])
>>> signal = node.add_candle(es_candle)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore


class RebalancingNode(BiasNode):
    """Monthly rebalancing pairs node comparing primary ticker vs cross-ticker performance.

    Parameters
    ----------
    ticker : Ticker
        Primary ticker (candles streamed via ``add_candle``).
    tf : TimeFrame
        Timeframe.
    cross_tickers : list[str] | None
        Secondary ticker dependency. Defaults to ``["TLT"]``.
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        cross_tickers: list[str] | None = None,
    ) -> None:
        super().__init__(ticker, tf)

        configured_cross = cross_tickers or ["TLT"]
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

        self.module_name = "rebalancing"
        self.output_features = ["signal"]
        self.params = {"cross_tickers": list(self.cross_tickers)}
        self.front_bad = 0

        self._store = CrossTickerDataStore.get_instance()

        # Monthly state
        self._current_period: Optional[Tuple[int, int]] = None
        self._month_start_ticker: Optional[float] = None
        self._month_start_cross: Optional[float] = None
        self._decision: Optional[str] = None
        self._trading_day_count: int = 0
        self._carry_over_signal: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ------------------------------------------------------------------

    def _compute_candle(self, candle: Candle) -> List[float]:
        dt = candle.datetime
        month = dt.month
        year = dt.year

        # Detect month change → reset state
        if (year, month) != self._current_period:
            self._current_period = (year, month)
            self._month_start_ticker = None
            self._month_start_cross = None
            self._decision = None
            self._trading_day_count = 0

        self._trading_day_count += 1

        # Fetch cross-ticker candle
        cross_candle = self._store.get_candle(self.cross_ticker, candle.tf, dt)
        if cross_candle is None:
            return [0.0]

        # Record first close of month for both tickers
        if self._month_start_ticker is None:
            self._month_start_ticker = candle.close
            self._month_start_cross = cross_candle.close
            # Carry-over from previous month's ticker-wins: long ticker through day 5
            if self._carry_over_signal and self._trading_day_count <= 5:
                return [1.0]
            return [0.0]

        # Carry-over: ticker won last month → long ticker until 5th trading day
        if self._carry_over_signal and self._trading_day_count <= 5:
            if self._trading_day_count == 5:
                self._carry_over_signal = 0
            return [1.0]

        # Observation period: first 15 calendar days (or first trading day after)
        if self._decision is None:
            ticker_pct = (candle.close - self._month_start_ticker) / self._month_start_ticker
            cross_pct = (cross_candle.close - self._month_start_cross) / self._month_start_cross

            # Make decision on day 15, or the first trading day on/after day 15
            should_decide = (
                dt.day >= 15
                or (dt.day >= 13 and self._trading_day_count >= 10)
            )
            if should_decide:
                if cross_pct > ticker_pct:
                    self._decision = "cross_wins"
                else:
                    self._decision = "ticker_wins"
                    self._carry_over_signal = 1
                # For the decision day itself, fall through to act immediately
            else:
                return [0.0]  # still in observation

        # Decision made — act on it
        if self._decision == "cross_wins":
            return [1.0]

        if self._decision == "ticker_wins":
            # Long ticker at end-of-month zone (day 25+) into next month
            if dt.day >= 25:
                return [1.0]
            return [0.0]

        return [0.0]
