"""
RSI Signal Bias Node

Discrete signals from RSI threshold crosses. Generates -1, 0, or 1
signals based on oversold/overbought RSI cross events and optional
fixed-exit rules.

Output: Rule-based -1, 0, or 1
"""

from typing import List, Optional
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from utils.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


class RSISignal(BiasNode):
    """
    RSI Signal - Discrete signals from RSI threshold crosses.

    Converts continuous RSI into discrete trading signals:
    - Signal = 1 (Long) when RSI crosses below oversold threshold
    - Signal = -1 (Short) when RSI crosses above overbought threshold
    - Signal = 0 (Neutral) when flat or fixed-exit fires

    Output Range: -1, 0, or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - oversold: RSI threshold for long signal (default: 30)
    - overbought: RSI threshold for short signal (default: 70)
    - strategy_mode: "long", "short", or "long-short" (default: "long")
    - exit_policy: "threshold" or "threshold_or_bars" (default: "threshold_or_bars")
    - exit_bars: bars in position before fixed exit (default: 5)
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: str = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ):
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

        # Standardized naming metadata
        self.module_name = 'rsisignal'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'strategyMode': self.strategy_mode,
            'exitPolicy': self.exit_policy,
            'exitBars': exit_bars,
        }

        # Warmup needs RSI period
        self.front_bad = rsi_period

        # RSI calculation state
        self.buffer_size = rsi_period + 1
        self.close_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.n_prices = 0
        self.prev_close = 0.0
        self.upsum = 1e-60
        self.dnsum = 1e-60

        # Position state
        self.position = 0
        self.bars_in_position = 0
        self.prev_rsi: Optional[float] = None

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI Signal for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the signal (-1, 0, or 1)
        """
        self.n_candles += 1
        curr_close = candle.close

        # Store current close in circular buffer
        self.close_buffer[self.buffer_idx] = curr_close
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.n_prices += 1

        # Calculate RSI
        if self.n_prices < self.rsi_period:
            self.prev_close = curr_close
            self.position = 0
            self.bars_in_position = 0
            self.prev_rsi = None
            self.output.append(0.0)
            return [0.0]
        elif self.n_prices == self.rsi_period:
            # Initialize RSI
            if self.buffer_idx == 0:
                init_prices = self.close_buffer[:self.rsi_period]
            else:
                init_prices = np.concatenate([
                    self.close_buffer[self.buffer_idx:],
                    self.close_buffer[:self.buffer_idx]
                ])
            self.upsum, self.dnsum = compute_rsi_initial(init_prices, self.rsi_period)
            rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
            self.prev_close = curr_close
        else:
            # Update RSI
            self.upsum, self.dnsum, rsi = update_rsi(
                self.prev_close,
                curr_close,
                self.upsum,
                self.dnsum,
                self.rsi_period
            )
            self.prev_close = curr_close

        signal = self._apply_position_rules(rsi)
        self.prev_rsi = rsi
        self.output.append(float(signal))
        return [float(signal)]

    def _normalize_strategy_mode(self, strategy_mode: str) -> str:
        normalized = strategy_mode.lower().strip()
        if normalized == "long_short":
            normalized = "long-short"
        if normalized not in {"long", "short", "long-short"}:
            raise ValueError(
                "strategy_mode must be 'long', 'short', or 'long-short'"
            )
        return normalized

    def _crossed_below(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi > self.oversold and rsi <= self.oversold

    def _crossed_above(self, prev_rsi: float, rsi: float) -> bool:
        return prev_rsi < self.overbought and rsi >= self.overbought

    def _apply_position_rules(self, rsi: float) -> int:
        if self.prev_rsi is None:
            return 0

        crossed_below = self._crossed_below(self.prev_rsi, rsi)
        crossed_above = self._crossed_above(self.prev_rsi, rsi)

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
