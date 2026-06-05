"""
Cyclical RSI Signal Bias Node

Discrete signals from threshold crosses on cyclical RSI: bandpass-style cycle
(short SMA minus long SMA) then RSI on the cycle, centered on zero (roughly
-50 to +50), with the same entry/exit rules as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`.

Output: -1, 0, or 1
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List, Optional

import numpy as np

from nodes import BiasNode
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle


class CyclicalRSISignal(BiasNode):
    """
    Cyclical RSI as discrete long/short/flat signals from threshold crosses.

    Thresholds apply to the **centered** cyclical RSI (classic RSI minus 50), not
    the 0–100 price RSI scale. Defaults ``oversold=-20`` / ``overbought=+20`` are
    analogous to classic 30 / 70 on the centered scale.

    Parameters
    ----------
    short_period, long_period, rsi_period
        Bandpass and RSI periods (same semantics as :class:`CyclicalRSI`).
    oversold, overbought
        Centered cyclical RSI levels for cross detection.
    strategy_mode, exit_policy, exit_bars
        Same as :class:`RSISignal`.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"shortPeriod", "longPeriod", "rsiPeriod"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        short_period: int = 5,
        long_period: int = 20,
        rsi_period: int = 14,
        oversold: float = -20.0,
        overbought: float = 20.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        self.short_period = short_period
        self.long_period = long_period
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

        self.module_name = "cyclicalrsisignal"
        self.output_features = ["signal"]
        self.params = {
            "shortPeriod": short_period,
            "longPeriod": long_period,
            "rsiPeriod": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        self.front_bad = long_period + rsi_period

        self.close_buffer: deque[float] = deque(maxlen=long_period)
        self.cycle_buffer: deque[float] = deque(maxlen=rsi_period + 1)

        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.prev_cycle: Optional[float] = None
        self.rsi_initialized = False

        self.position = 0
        self.bars_in_position = 0
        self.prev_cyclical_rsi: Optional[float] = None

        self.n_candles = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        if len(self.close_buffer) >= self.long_period:
            closes = list(self.close_buffer)
            short_sma = float(np.mean(closes[-self.short_period :]))
            long_sma = float(np.mean(closes))
            cycle = short_sma - long_sma
            self.cycle_buffer.append(cycle)

        if self.n_candles < self.front_bad:
            if len(self.cycle_buffer) > 0:
                self.prev_cycle = self.cycle_buffer[-1]
            self.prev_cyclical_rsi = None
            self.position = 0
            self.bars_in_position = 0
            self.output.append(0.0)
            return [0.0]

        if not self.rsi_initialized and len(self.cycle_buffer) >= self.rsi_period:
            cycle_array = np.array(list(self.cycle_buffer)[: self.rsi_period])
            self.upsum, self.dnsum = self._compute_rsi_initial(cycle_array)
            self.rsi_initialized = True

        if self.rsi_initialized and self.prev_cycle is not None:
            curr_cycle = self.cycle_buffer[-1]
            diff = curr_cycle - self.prev_cycle

            if diff > 0:
                self.upsum = (
                    (self.rsi_period - 1) * self.upsum + diff
                ) / self.rsi_period
                self.dnsum *= (self.rsi_period - 1.0) / self.rsi_period
            else:
                self.dnsum = (
                    (self.rsi_period - 1) * self.dnsum - diff
                ) / self.rsi_period
                self.upsum *= (self.rsi_period - 1.0) / self.rsi_period

            rsi_raw = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_cycle = curr_cycle
        else:
            rsi_raw = 50.0
            if len(self.cycle_buffer) > 0:
                self.prev_cycle = self.cycle_buffer[-1]

        cyclical_rsi = max(-50.0, min(50.0, rsi_raw - 50.0))

        signal = self._apply_position_rules(cyclical_rsi)
        self.prev_cyclical_rsi = cyclical_rsi
        self.output.append(float(signal))
        return [float(signal)]

    def _compute_rsi_initial(self, values: np.ndarray) -> tuple[float, float]:
        upsum = 1e-60
        dnsum = 1e-60
        for i in range(1, len(values)):
            diff = float(values[i] - values[i - 1])
            if diff > 0:
                upsum += diff
            else:
                dnsum -= diff

        n = len(values) - 1
        upsum /= n
        dnsum /= n
        return upsum, dnsum

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _crossed_below(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi > self.oversold and rsi <= self.oversold

    def _crossed_above(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi < self.overbought and rsi >= self.overbought

    def _apply_position_rules(self, cyclical_rsi: float) -> int:
        if self.prev_cyclical_rsi is None:
            return 0

        crossed_below = self._crossed_below(self.prev_cyclical_rsi, cyclical_rsi)
        crossed_above = self._crossed_above(self.prev_cyclical_rsi, cyclical_rsi)

        next_position = self.position

        if self.strategy_mode == "long":
            if self.position == 0 and crossed_below:
                next_position = 1
            elif self.position == 1 and crossed_above:
                next_position = 0
        elif self.strategy_mode == "short":
            if self.position == 0 and crossed_above:
                next_position = -1
            elif self.position == -1 and crossed_below:
                next_position = 0
        else:
            if crossed_below:
                next_position = 1
            elif crossed_above:
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
