"""
Pre-holiday equity calendar bias node (rule-based).

Long signal on ES during D-4 through D0 (trading days) before selected
NYSE equity holiday closures (Good Friday, Independence Day, Thanksgiving).
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from nodes.seasonal.calendar._event_signal import (
    build_active_sessions,
    holiday_d0_dates,
    load_bundle,
)
from utils.calendar.calendar_loader import HolidayAssetBucket
from utils.calendar.trading_day_index import TradingDayIndex, load_es_trading_sessions
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

_DEFAULT_ENTRY = -4
_DEFAULT_EXIT = 0
_EQUITY_TICKERS = frozenset({Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY})


class PreHolidayEquity(BiasNode):
    """Rule-based pre-holiday equity window signal (0/1)."""

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = ()

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        entry_offset: int = _DEFAULT_ENTRY,
        exit_offset: int = _DEFAULT_EXIT,
    ) -> None:
        super().__init__(ticker, tf)
        self.entry_offset = entry_offset
        self.exit_offset = exit_offset
        self.module_name = "preholidayequity"
        self.output_features = ["signal"]
        self.params = {
            "entry_offset": entry_offset,
            "exit_offset": exit_offset,
        }
        self.front_bad = 0

        bundle = load_bundle()
        d0_dates = holiday_d0_dates(bundle, HolidayAssetBucket.EQUITY)
        trading_index = TradingDayIndex(load_es_trading_sessions())
        self._active_sessions = build_active_sessions(
            d0_dates,
            entry_offset,
            exit_offset,
            trading_index,
        )
        self._applies = ticker in _EQUITY_TICKERS

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        session = candle.datetime.date()
        signal = (
            1
            if self._applies and session in self._active_sessions
            else 0
        )
        self.output = [signal]
        return [signal]
