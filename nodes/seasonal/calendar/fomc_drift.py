"""
FOMC drift calendar bias node (rule-based).

Long signal during D-2 through D0 (trading days) ending on the FOMC decision day.
Applies to ES, NQ, and GC (equity index + gold research sleeve; article uses VTI/GLD).
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from nodes.seasonal.calendar._event_signal import build_active_sessions, load_bundle
from data_platform.events.trading_day_index import TradingDayIndex, load_es_trading_sessions
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

_DEFAULT_ENTRY = -2
_DEFAULT_EXIT = 0
_FOMC_TICKERS = frozenset({Ticker.ES, Ticker.NQ, Ticker.GC})


class FomcDrift(BiasNode):
    """Rule-based FOMC drift window signal (0/1)."""

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
        self.module_name = "fomcdrift"
        self.output_features = ["signal"]
        self.params = {
            "entry_offset": entry_offset,
            "exit_offset": exit_offset,
        }
        self.front_bad = 0

        bundle = load_bundle()
        trading_index = TradingDayIndex(load_es_trading_sessions())
        self._active_sessions = build_active_sessions(
            bundle.fomc_decision_dates,
            entry_offset,
            exit_offset,
            trading_index,
        )
        self._applies = ticker in _FOMC_TICKERS

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
