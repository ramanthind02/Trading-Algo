"""
RSI signal with ATR slope required on the *entry cross* bar only.

Unlike ``filter_gate``, the ATR filter does not multiply the live signal each bar.

- **Long:** Enter only when RSI **crosses into oversold** on a bar where the ATR
  slope filter is **on**. If the cross happens while the filter is off, that
  dip is skipped — a later ATR-on while still oversold does **not** enter.
- **Short:** Symmetric: enter only on **cross into overbought** with filter on.
- **long_short:** From flat, a cross into oversold/overbought counts only if the
  filter is on that bar. If already long or short, an opposite-threshold cross
  still flips without the filter (same flip semantics as ``RSISignal``).

**Exits** (overbought / oversold threshold flips, bar-based exit) follow the same
rules as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal` — no ATR filter
on exits.

ATR slope logic is delegated to :class:`~nodes.volatility.atr.atr_slope_filter.AtrSlopeFilterNode`.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Union

import numpy as np

from nodes import BiasNode, LookbackContribution
from nodes.volatility.atr.atr_slope_filter import AtrSlopeDirection, _coerce_direction
from lib.compute.rsi_helpers import compute_rsi_initial, update_rsi
from lib.core import helpers
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle


class RSISignalAtrSlopeEntry(BiasNode):
    """
    RSISignal with ATR slope required on the same bar as the RSI entry cross.

    See module docstring for semantics vs ``filter_gate`` and the prior entry-window variant.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"rsiPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
        atr_period: int = 14,
        atr_direction: Union[AtrSlopeDirection, str] = AtrSlopeDirection.FALLING,
    ) -> None:
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)
        self.exit_policy = exit_policy.lower().strip()
        if self.exit_policy not in {"threshold", "threshold_or_bars"}:
            raise ValueError(
                "exit_policy must be 'threshold' or 'threshold_or_bars'"
            )
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        self.exit_bars = exit_bars

        if atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        self.atr_period = atr_period
        self._atr_direction = _coerce_direction(atr_direction)

        direction_arg: Union[AtrSlopeDirection, str] = (
            self._atr_direction
            if isinstance(atr_direction, AtrSlopeDirection)
            else str(atr_direction)
        )
        self._atr_filter = helpers.create_fresh_bias_node(
            "atr_slope_filter",
            ticker,
            tf,
            {"period": atr_period, "direction": direction_arg},
        )

        self.module_name = "rsi_signal_atr_slope_entry"
        self.output_features = ["signal"]
        self.params = {
            "rsiPeriod": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
            "atrPeriod": atr_period,
            "atrDirection": self._atr_direction.value,
        }

        self.front_bad = max(
            rsi_period,
            int(getattr(self._atr_filter, "front_bad", 0)),
        )

        self.buffer_size = rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        self.position = 0
        self.bars_in_position = 0
        self.prev_rsi: Optional[float] = None

        self.n_candles = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        return self._atr_filter.lookback_contributions()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _crossed_below(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi > self.oversold and rsi <= self.oversold

    def _crossed_above(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi < self.overbought and rsi >= self.overbought

    def _filter_on(self, candle: Candle) -> bool:
        out = self._atr_filter.add_candle(candle)
        return bool(out) and float(out[0]) != 0.0

    def _compute_candle(self, candle: Candle) -> List[float]:
        self.n_candles += 1
        filter_on = self._filter_on(candle)

        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            self.position = 0
            self.bars_in_position = 0
            self.prev_rsi = None
            self.output.append(0.0)
            return [0.0]

        if self.n_prices == self.rsi_period:
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[: self.rsi_period]
            else:
                init_prices = np.concatenate(
                    [
                        self.close_buffer[self.buffer_idx :],
                        self.close_buffer[: self.buffer_idx],
                    ]
                )
            self.upsum, self.dnsum = compute_rsi_initial(init_prices, self.rsi_period)
            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_close = curr_close
        else:
            self.upsum, self.dnsum, rsi = update_rsi(
                self.prev_close,
                curr_close,
                self.upsum,
                self.dnsum,
                self.rsi_period,
            )
            self.prev_close = curr_close

        signal = self._apply_position_rules(rsi, filter_on)
        self.prev_rsi = rsi
        self.output.append(float(signal))
        return [float(signal)]

    def _apply_position_rules(self, rsi: float, filter_on: bool) -> int:
        if self.prev_rsi is None:
            return 0

        crossed_below = self._crossed_below(self.prev_rsi, rsi)
        crossed_above = self._crossed_above(self.prev_rsi, rsi)

        next_position = self.position

        if self.strategy_mode == "long":
            if self.position == 0 and crossed_below and filter_on:
                next_position = 1
            elif self.position == 1 and crossed_above:
                next_position = 0
        elif self.strategy_mode == "short":
            if self.position == 0 and crossed_above and filter_on:
                next_position = -1
            elif self.position == -1 and crossed_below:
                next_position = 0
        else:
            if crossed_below:
                if self.position == 0 and not filter_on:
                    next_position = 0
                else:
                    next_position = 1
            elif crossed_above:
                if self.position == 0 and not filter_on:
                    next_position = 0
                else:
                    next_position = -1

        if next_position == self.position and next_position != 0:
            self.bars_in_position += 1
        elif next_position != 0:
            self.bars_in_position = 1
        else:
            self.bars_in_position = 0

        if (
            self.exit_policy == "threshold_or_bars"
            and next_position != 0
            and self.bars_in_position >= self.exit_bars
        ):
            next_position = 0
            self.bars_in_position = 0

        self.position = next_position
        return next_position
