"""
Calendar ensemble bias node (rule-based).

Combines pre-holiday and FOMC calendar windows per asset class. Overlapping
holiday + FOMC exposure on the same session yields a single active signal (OR),
matching the article's no-double-count rule.

Defaults (Beyond Passive Investing, Mar 2026):
- Equity (ES, NQ): pre-holiday D-4…D0; FOMC D-2…D0
- Gold (GC): pre-holiday D-2…D+1; FOMC D-2…D0
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

_DEFAULT_EQUITY_ENTRY = -4
_DEFAULT_EQUITY_EXIT = 0
_DEFAULT_GOLD_ENTRY = -2
_DEFAULT_GOLD_EXIT = 1
_DEFAULT_FOMC_ENTRY = -2
_DEFAULT_FOMC_EXIT = 0

_EQUITY_TICKERS = frozenset({Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY})
_GOLD_TICKERS = frozenset({Ticker.GC})


class CalendarEnsemble(BiasNode):
    """OR of pre-holiday + FOMC calendar signals for the ticker's asset bucket."""

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = ()

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        equity_entry_offset: int = _DEFAULT_EQUITY_ENTRY,
        equity_exit_offset: int = _DEFAULT_EQUITY_EXIT,
        gold_entry_offset: int = _DEFAULT_GOLD_ENTRY,
        gold_exit_offset: int = _DEFAULT_GOLD_EXIT,
        fomc_entry_offset: int = _DEFAULT_FOMC_ENTRY,
        fomc_exit_offset: int = _DEFAULT_FOMC_EXIT,
    ) -> None:
        super().__init__(ticker, tf)

        self.module_name = "calendarensemble"
        self.output_features = ["signal"]
        self.params = {
            "equity_entry_offset": equity_entry_offset,
            "equity_exit_offset": equity_exit_offset,
            "gold_entry_offset": gold_entry_offset,
            "gold_exit_offset": gold_exit_offset,
            "fomc_entry_offset": fomc_entry_offset,
            "fomc_exit_offset": fomc_exit_offset,
        }
        self.front_bad = 0

        self._holiday_node: BiasNode | None = None
        if ticker in _EQUITY_TICKERS:
            self._holiday_node = helpers.create_fresh_bias_node(
                "pre_holiday_equity",
                ticker,
                tf,
                {
                    "entry_offset": equity_entry_offset,
                    "exit_offset": equity_exit_offset,
                },
            )
        elif ticker in _GOLD_TICKERS:
            self._holiday_node = helpers.create_fresh_bias_node(
                "pre_holiday_gold",
                ticker,
                tf,
                {
                    "entry_offset": gold_entry_offset,
                    "exit_offset": gold_exit_offset,
                },
            )

        self._fomc_node = helpers.create_fresh_bias_node(
            "fomc_drift",
            ticker,
            tf,
            {
                "entry_offset": fomc_entry_offset,
                "exit_offset": fomc_exit_offset,
            },
        )

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        holiday_sig = 0
        if self._holiday_node is not None:
            holiday_out = self._holiday_node.add_candle(candle)
            holiday_sig = int(holiday_out[0]) if holiday_out else 0

        fomc_out = self._fomc_node.add_candle(candle)
        fomc_sig = int(fomc_out[0]) if fomc_out else 0

        signal = 1 if (holiday_sig >= 1 or fomc_sig >= 1) else 0
        self.output = [signal]
        return [signal]
