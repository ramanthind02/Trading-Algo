"""
ADX threshold filter bias node.

Computes Wilder-style ADX and emits a binary gate: ``1.0`` when ADX satisfies
the chosen comparison to ``threshold``, else ``0.0``. Intended for use as the
filter leg of ``filter_gate`` / ``filter_and_signal`` (non-zero = gate open).
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, List, Union

from nodes import BiasNode
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame


class AdxFilterCompare(Enum):
    """How ADX is compared to ``threshold`` for the binary output."""

    BELOW = "below"
    ABOVE = "above"


def _coerce_compare(raw: Union[AdxFilterCompare, str]) -> AdxFilterCompare:
    if isinstance(raw, AdxFilterCompare):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"compare must be AdxFilterCompare or str, got {type(raw).__name__}")
    key = raw.strip().lower()
    aliases = {
        "below": AdxFilterCompare.BELOW,
        "<": AdxFilterCompare.BELOW,
        "lt": AdxFilterCompare.BELOW,
        "above": AdxFilterCompare.ABOVE,
        ">": AdxFilterCompare.ABOVE,
        "gt": AdxFilterCompare.ABOVE,
    }
    if key not in aliases:
        raise ValueError(
            f'Invalid compare "{raw}". Use below/< (ADX below threshold -> 1) or above/> (ADX above threshold -> 1).'
        )
    return aliases[key]


class AdxFilterNode(BiasNode):
    """
    Binary filter from ADX vs a fixed threshold.

    - ``compare=BELOW`` (default): output ``1.0`` when ``ADX < threshold``, else ``0.0``.
    - ``compare=ABOVE``: output ``1.0`` when ``ADX > threshold``, else ``0.0``.

    Warmup bars emit ``0.0`` (gate closed). ``front_bad`` is ``2 * length`` candles
    (first bar initializes OHLC; Wilder smoothing for DI/DX and ADX uses ``length`` twice).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"length"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        length: int = 14,
        threshold: float = 20.0,
        compare: Union[AdxFilterCompare, str] = AdxFilterCompare.BELOW,
    ) -> None:
        super().__init__(ticker, tf)

        if length < 2:
            raise ValueError("length must be >= 2 for ADX.")
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

        self.length = length
        self.threshold = float(threshold)
        self.compare = _coerce_compare(compare)

        self.module_name = "adx_filter"
        self.output_features = ["signal"]
        self.params = {
            "length": length,
            "threshold": self.threshold,
            "compare": self.compare.value,
        }
        self.front_bad = 2 * length

        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None

        self._tr_seeded = False
        self._sum_tr = 0.0
        self._sum_pdm = 0.0
        self._sum_mdm = 0.0
        self._acc_count = 0
        self._s_tr = 0.0
        self._s_pdm = 0.0
        self._s_mdm = 0.0

        self._adx_smooth: float | None = None
        self._sum_dx = 0.0
        self._dx_bars = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _passes(self, adx_val: float) -> bool:
        if self.compare is AdxFilterCompare.BELOW:
            return adx_val < self.threshold
        return adx_val > self.threshold

    def _compute_candle(self, candle: Candle) -> List[float]:
        h = float(candle.high)
        l_ = float(candle.low)
        c = float(candle.close)

        if self._prev_close is None:
            self._prev_high, self._prev_low, self._prev_close = h, l_, c
            out = 0.0
            self.output.append(out)
            return [out]

        tr = max(h - l_, abs(h - self._prev_close), abs(l_ - self._prev_close))
        up_move = h - self._prev_high
        down_move = self._prev_low - l_
        pdm = up_move if up_move > down_move and up_move > 0 else 0.0
        mdm = down_move if down_move > up_move and down_move > 0 else 0.0

        self._prev_high, self._prev_low, self._prev_close = h, l_, c

        if not self._tr_seeded:
            self._sum_tr += tr
            self._sum_pdm += pdm
            self._sum_mdm += mdm
            self._acc_count += 1
            if self._acc_count < self.length:
                out = 0.0
                self.output.append(out)
                return [out]
            self._s_tr = self._sum_tr
            self._s_pdm = self._sum_pdm
            self._s_mdm = self._sum_mdm
            self._tr_seeded = True
        else:
            self._s_tr = self._s_tr - self._s_tr / self.length + tr
            self._s_pdm = self._s_pdm - self._s_pdm / self.length + pdm
            self._s_mdm = self._s_mdm - self._s_mdm / self.length + mdm

        dip = 100.0 * self._s_pdm / self._s_tr if self._s_tr > 0 else 0.0
        dim = 100.0 * self._s_mdm / self._s_tr if self._s_tr > 0 else 0.0
        di_sum = dip + dim
        dx = 100.0 * abs(dip - dim) / di_sum if di_sum > 0 else 0.0

        if self._adx_smooth is None:
            self._sum_dx += dx
            self._dx_bars += 1
            if self._dx_bars < self.length:
                out = 0.0
                self.output.append(out)
                return [out]
            # Wilder ADX: first value is the simple mean of the first ``length`` DX readings.
            self._adx_smooth = self._sum_dx / self.length
        else:
            n = self.length
            self._adx_smooth = (self._adx_smooth * (n - 1) + dx) / n

        adx_val = self._adx_smooth
        out = 1.0 if self._passes(adx_val) else 0.0
        self.output.append(out)
        return [out]
