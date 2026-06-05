"""
Cumulative RSI Signal Bias Node

Discrete signals from RSI threshold crosses applied to **smoothed** RSI: same RSI
engine and rolling average as :class:`~nodes.mean_reversion.rsi.cumulative_rsi.CumulativeRSI`,
with the same entry/exit rules as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`.

Crosses compare consecutive values of the cumulative (averaged) RSI, not raw RSI.

Output: -1, 0, or 1
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import numpy as np

from nodes import BiasNode
from nodes.mean_reversion.rsi.cumulative_rsi import compute_avg_rsi
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle
from lib.compute.fast_nodes import compute_rsi_initial_fast, update_rsi_fast


class CumulativeRSISignal(BiasNode):
    """
    Threshold crosses on rolling-average RSI (discrete long/short/flat).

    Parameters
    ----------
    lookback, avg_period
        Same as :class:`~nodes.mean_reversion.rsi.cumulative_rsi.CumulativeRSI`.
    oversold, overbought
        Classic 0–100 RSI scale (same as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`).
    strategy_mode, exit_policy, exit_bars
        Same as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback", "avgPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 14,
        avg_period: int = 5,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if avg_period < 1:
            raise ValueError("avg_period must be >= 1")

        self.lookback = lookback
        self.avg_period = avg_period
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

        self.module_name = "cumulative_rsi_signal"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "avgPeriod": avg_period,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        self.front_bad = lookback + avg_period - 1

        self.upsum = 1e-60
        self.dnsum = 1e-60
        self.rsi_buffer = np.zeros(avg_period, dtype=np.float64)
        self.rsi_buffer_idx = 0
        self.n_rsi_values = 0

        self.buffer_size = lookback + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0

        self.position = 0
        self.bars_in_position = 0
        self.prev_avg_rsi: Optional[float] = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _crossed_below(self, prev_avg: float, avg: float) -> bool:
        return prev_avg > self.oversold and avg <= self.oversold

    def _crossed_above(self, prev_avg: float, avg: float) -> bool:
        return prev_avg < self.overbought and avg >= self.overbought

    def _apply_position_rules(self, avg_rsi: float) -> int:
        if self.prev_avg_rsi is None:
            return 0

        crossed_below = self._crossed_below(self.prev_avg_rsi, avg_rsi)
        crossed_above = self._crossed_above(self.prev_avg_rsi, avg_rsi)

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

    def _compute_candle(self, candle: Candle) -> List:
        curr_close = candle.close
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        if self.n_prices < self.lookback:
            self.prev_close = curr_close
            self.position = 0
            self.bars_in_position = 0
            self.prev_avg_rsi = None
            if self.n_rsi_values < self.avg_period:
                self.rsi_buffer[self.rsi_buffer_idx] = 50.0
                self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.avg_period
                self.n_rsi_values += 1
            self.output.append(0.0)
            return [0.0]

        if self.n_prices == self.lookback:
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[: self.lookback]
            else:
                init_prices = np.concatenate(
                    [
                        self.close_buffer[self.buffer_idx :],
                        self.close_buffer[: self.buffer_idx],
                    ]
                )
            self.upsum, self.dnsum = compute_rsi_initial_fast(init_prices, self.lookback)

        self.upsum, self.dnsum, current_rsi = update_rsi_fast(
            self.prev_close,
            curr_close,
            self.upsum,
            self.dnsum,
            self.lookback,
        )

        self.rsi_buffer[self.rsi_buffer_idx] = current_rsi
        self.rsi_buffer_idx = (self.rsi_buffer_idx + 1) % self.avg_period
        if self.n_rsi_values < self.avg_period:
            self.n_rsi_values += 1

        avg_rsi = compute_avg_rsi(
            self.rsi_buffer, min(self.n_rsi_values, self.avg_period)
        )

        signal = self._apply_position_rules(avg_rsi)
        self.prev_avg_rsi = avg_rsi
        self.prev_close = curr_close

        self.output.append(float(signal))
        return [float(signal)]
