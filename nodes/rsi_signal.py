"""
RSI Signal Bias Node

Discrete signals from RSI thresholds. Generates -1, 0, or 1 signals
based on oversold/overbought RSI levels.

Output: Rule-based -1, 0, or 1
"""

from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from utils.rsi_helpers import compute_rsi_initial, update_rsi
from nodes import BiasNode


class RSISignal(BiasNode):
    """
    RSI Signal - Discrete signals from RSI thresholds.

    Converts continuous RSI into discrete trading signals:
    - Signal = 1 (Long) when RSI crosses below oversold threshold
    - Signal = -1 (Short) when RSI crosses above overbought threshold
    - Signal = 0 (Neutral) when RSI is between thresholds

    Different modes available:
    - "threshold": Signal based on current RSI level
    - "cross": Signal only on threshold crosses (more selective)

    Output Range: -1, 0, or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - oversold: RSI threshold for long signal (default: 30)
    - overbought: RSI threshold for short signal (default: 70)
    - mode: Signal generation mode ("threshold" or "cross", default: "threshold")
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        mode: str = "threshold"
    ):
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.mode = mode.lower()

        # Standardized naming metadata
        self.module_name = 'rsisignal'
        self.output_features = ['signal']
        self.params = {
            'rsiPeriod': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'mode': self.mode
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

        # Previous RSI for cross detection
        self.prev_rsi = None

        # Current signal state (for cross mode persistence)
        self.current_signal = 0

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

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

        # Generate signal based on mode
        if self.mode == "threshold":
            # Signal based on current RSI level
            if rsi <= self.oversold:
                signal = 1  # Oversold, bullish
            elif rsi >= self.overbought:
                signal = -1  # Overbought, bearish
            else:
                signal = 0  # Neutral zone

        elif self.mode == "cross":
            # Signal only on threshold crosses
            if self.prev_rsi is not None:
                # Cross below oversold (entering oversold)
                if self.prev_rsi > self.oversold and rsi <= self.oversold:
                    self.current_signal = 1
                # Cross above overbought (entering overbought)
                elif self.prev_rsi < self.overbought and rsi >= self.overbought:
                    self.current_signal = -1
                # Exit conditions: cross back to neutral zone
                elif self.current_signal == 1 and rsi >= 50:
                    self.current_signal = 0
                elif self.current_signal == -1 and rsi <= 50:
                    self.current_signal = 0

            signal = self.current_signal
        else:
            # Default to threshold mode
            if rsi <= self.oversold:
                signal = 1
            elif rsi >= self.overbought:
                signal = -1
            else:
                signal = 0

        self.prev_rsi = rsi

        self.output.append(float(signal))
        return [float(signal)]
