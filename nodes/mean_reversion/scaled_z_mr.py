"""
Scaled Z-Score Mean Reversion — three-tranche scale-in on price deviation.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

_TRANCHE_WEIGHT: float = 1.0 / 3.0


class ScaledZMeanReversion(BiasNode):
    """
    Scales into a long mean-reversion position in three equal tranches as
    price falls progressively further below its rolling mean.

    z = (close − mean(close, lookback)) / std(close, lookback)

    Tranche 1 activates when z ≤ z1  (mild oversold)
    Tranche 2 activates when z ≤ z2  (moderate oversold)
    Tranche 3 activates when z ≤ z3  (extreme oversold)

    All active tranches exit together when z ≥ exit_z.
    Once a tranche activates it stays active until the group exit fires —
    it does not deactivate if z drifts back above its own entry threshold.

    Output is 0, 1/3, 2/3, or 1.0 — the fraction of full position held.
    Combine with FilterGateNode(SmaAboveFilterNode(252), ...) for bear-market gating.

    Parameters
    ----------
    lookback : int   Rolling window for mean and std (default 20).
    z1       : float First entry threshold, must be < 0 (default -1.0).
    z2       : float Second entry threshold, must be ≤ z1 (default -1.5).
    z3       : float Third entry threshold, must be ≤ z2 (default -2.0).
    exit_z   : float Z-score above which all tranches exit (default 0.0).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 20,
        z1: float = -1.0,
        z2: float = -1.5,
        z3: float = -2.0,
        exit_z: float = 0.0,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if z1 >= 0.0:
            raise ValueError(f"z1 must be negative, got {z1}")
        if z2 > z1:
            raise ValueError(f"z2 must be <= z1, got z2={z2}, z1={z1}")
        if z3 > z2:
            raise ValueError(f"z3 must be <= z2, got z3={z3}, z2={z2}")

        self.lookback = lookback
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.exit_z = exit_z

        self.module_name = "scaled_z_mr"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "exitZ": exit_z,
        }

        self.front_bad = lookback

        # Rolling window for z-score — running sum/sum_sq for O(1) updates
        self._closes: deque[float] = deque(maxlen=lookback)
        self._price_sum: float = 0.0
        self._price_sum_sq: float = 0.0
        self._n: int = 0

        # Independent tranche activation state
        self._t1: bool = False
        self._t2: bool = False
        self._t3: bool = False

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

        if z >= self.exit_z:
            self._t1 = self._t2 = self._t3 = False
        else:
            if z <= self.z1:
                self._t1 = True
            if z <= self.z2:
                self._t2 = True
            if z <= self.z3:
                self._t3 = True

        out = _TRANCHE_WEIGHT * (self._t1 + self._t2 + self._t3)
        self.output.append(out)
        return [out]
