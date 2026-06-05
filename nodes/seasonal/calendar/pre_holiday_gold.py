"""
Pre-holiday gold calendar bias node (rule-based).

Long signal on GC during D-2 through D+1 (trading days) around selected
winter-related NYSE holiday closures (Christmas, New Year, MLK, Presidents Day).
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from nodes.seasonal.calendar._event_signal import (
    build_active_sessions,
    holiday_d0_dates,
    load_bundle,
)
from data_platform.events.calendar_loader import HolidayAssetBucket
from data_platform.events.trading_day_index import TradingDayIndex, load_es_trading_sessions
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

_DEFAULT_ENTRY = -2
_DEFAULT_EXIT = 1
_GOLD_TICKERS = frozenset({Ticker.GC})


class PreHolidayGold(BiasNode):
    """Rule-based pre-holiday gold window signal (0/1)."""

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
        self.module_name = "preholidaygold"
        self.output_features = ["signal"]
        self.params = {
            "entry_offset": entry_offset,
            "exit_offset": exit_offset,
        }
        self.front_bad = 0

        bundle = load_bundle()
        d0_dates = holiday_d0_dates(bundle, HolidayAssetBucket.GOLD)
        trading_index = TradingDayIndex(load_es_trading_sessions())
        self._active_sessions = build_active_sessions(
            d0_dates,
            entry_offset,
            exit_offset,
            trading_index,
        )
        self._applies = ticker in _GOLD_TICKERS

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
