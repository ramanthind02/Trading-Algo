from __future__ import annotations

import math
from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class TanhZMeanReversion(BiasNode):
    """
    Continuous z-score mean reversion via tanh scaling.

    Computes the z-score of close price relative to its rolling SMA, then maps
    it to a continuous signal in [-2, 2]:

        z      = (close - mean(close, lookback)) / std(close, lookback)
        signal = -2 * tanh(z * k)

    Positive signal = long  (price below mean, expect reversion up).
    Negative signal = short (price above mean, expect reversion down).
    Zero             = at equilibrium.

    The ``k`` parameter controls how quickly the signal builds:
      - k=1.0  → z=2 gives signal ≈ -1.93 (gradual; never fully saturates)
      - k=1.5  → z=2 gives signal ≈ -2.00 (full position at ~2σ)
      - k=2.0  → z=1.5 already near ±2    (aggressive)

    Parameters
    ----------
    lookback : int   Rolling window for mean and std (default 20).
    k        : float Tanh steepness — controls how fast the signal saturates.
                     (default 1.5)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 20,
        k: float = 1.5,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if k <= 0.0:
            raise ValueError(f"k must be > 0, got {k}")

        self.lookback = lookback
        self.k = k

        self.module_name = "tanh_z_mr"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "k": k,
        }

        self.front_bad = lookback

        # O(1) running sum / sum-of-squares for mean and variance
        self._closes: deque[float] = deque(maxlen=lookback)
        self._price_sum: float = 0.0
        self._price_sum_sq: float = 0.0
        self._n: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _push(self, close: float) -> None:
        if len(self._closes) == self.lookback:
            evicted = self._closes[0]
            self._price_sum -= evicted
            self._price_sum_sq -= evicted * evicted
        self._closes.append(close)
        self._price_sum += close
        self._price_sum_sq += close * close

    def _z_score(self) -> float:
        n = len(self._closes)
        if n < 2:
            return 0.0
        mean = self._price_sum / n
        variance = (self._price_sum_sq / n) - (mean * mean)
        std = variance ** 0.5 if variance > 0.0 else 0.0
        if std < 1e-10:
            return 0.0
        return (self._closes[-1] - mean) / std

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._push(float(candle.close))
        self._n += 1

        if self._n < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        z = self._z_score()
        signal = -2.0 * math.tanh(z * self.k)
        self.output.append(signal)
        return [signal]
