"""
ATR slope filter — binary volatility gate from SMA(ATR) vs prior bar.

Emits ``1.0`` when the chosen direction condition holds (ATR strictly rising or
strictly falling vs the previous bar), else ``0.0``. Intended as the filter leg of
``filter_gate`` / ``filter_and_signal`` (non-zero = gate open).

Uses the same SMA-of-true-range ATR and ``compute_atr_fast`` path as ``ATRNode``.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, List, Union

import numpy as np

from nodes import BiasNode
from lib.compute.fast_nodes import compute_atr_fast
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame


class AtrSlopeDirection(Enum):
    """Which ATR move counts as *filter on* (output ``1.0``)."""

    RISING = "rising"
    FALLING = "falling"


def _coerce_direction(raw: Union[AtrSlopeDirection, str]) -> AtrSlopeDirection:
    if isinstance(raw, AtrSlopeDirection):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"direction must be AtrSlopeDirection or str, got {type(raw).__name__}")
    key = raw.strip().lower()
    aliases = {
        "rising": AtrSlopeDirection.RISING,
        "up": AtrSlopeDirection.RISING,
        "falling": AtrSlopeDirection.FALLING,
        "down": AtrSlopeDirection.FALLING,
    }
    if key not in aliases:
        raise ValueError(
            f'Invalid direction "{raw}". Use rising/up or falling/down.'
        )
    return aliases[key]


class AtrSlopeFilterNode(BiasNode):
    """
    Binary filter: ATR (SMA of true range) vs its prior-bar value.

    Requires a full ``period``-bar ATR window, then one bar to establish a prior
    ATR for comparison. Until then, output is ``0.0``.

    - ``RISING``: ``1.0`` when ``ATR_t > ATR_{t-1}``, else ``0.0``.
    - ``FALLING``: ``1.0`` when ``ATR_t < ATR_{t-1}``, else ``0.0``.

    If ``ATR_t == ATR_{t-1}``, output is ``0.0`` for both modes.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"period"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 14,
        direction: Union[AtrSlopeDirection, str] = AtrSlopeDirection.RISING,
    ) -> None:
        super().__init__(ticker, tf)

        if period < 1:
            raise ValueError("period must be >= 1 for ATR slope filter.")

        self.period = period
        self.direction = _coerce_direction(direction)

        self.module_name = "atr_slope_filter"
        self.output_features = ["signal"]
        self.params = {"period": period, "direction": self.direction.value}
        self.front_bad = period + 1

        self._prev_close: float = -1.0
        self._true_ranges = np.zeros(period, dtype=np.float64)
        self._buffer_idx = 0
        self._n_filled = 0
        self._prev_atr: float | None = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _passes(self, atr: float, prev_atr: float) -> bool:
        if self.direction is AtrSlopeDirection.RISING:
            return atr > prev_atr
        return atr < prev_atr

    def _compute_candle(self, candle: Candle) -> List[float]:
        atr, _, self._buffer_idx, self._n_filled = compute_atr_fast(
            float(candle.high),
            float(candle.low),
            float(candle.close),
            self._prev_close,
            self._true_ranges,
            self._buffer_idx,
            self._n_filled,
            self.period,
        )
        self._prev_close = float(candle.close)

        if self._n_filled < self.period:
            out = 0.0
            self.output.append(out)
            return [out]

        if self._prev_atr is None:
            self._prev_atr = atr
            out = 0.0
            self.output.append(out)
            return [out]

        out = 1.0 if self._passes(atr, self._prev_atr) else 0.0
        self._prev_atr = atr
        self.output.append(out)
        return [out]
