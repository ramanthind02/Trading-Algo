"""
Scaled Z-Score Mean Reversion — symmetric long/short three-tranche scale-in.

Symmetric extension of ``nodes/mean_reversion/scaled_z_mr.ScaledZMeanReversion``
(long-only, outputs 0 -> 1). This node trades both sides of a raw price z-score:
it scales LONG as price stretches below its rolling mean and scales SHORT as
price stretches above it, exiting all tranches when the deviation reverts past
``exit_z`` toward the mean (crossing zero).

Output is in [-1.0, +1.0]:
    -1, -2/3, -1/3 (short tranches), 0 (flat), +1/3, +2/3, +1 (long tranches).

Research artifact (agent-designed). Promote into ``nodes/mean_reversion/`` if proven.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

_TRANCHE_WEIGHT: float = 1.0 / 3.0


class ScaledZMeanReversionLongShort(BiasNode):
    """
    Scales into a mean-reversion position in three equal tranches on BOTH sides
    as price stretches progressively further from its rolling mean.

    z = (close - mean(close, lookback)) / std(close, lookback)

    LONG side (price below mean, z < 0):
        Tranche 1 activates when z <= z1  (mild oversold)
        Tranche 2 activates when z <= z2  (moderate oversold)
        Tranche 3 activates when z <= z3  (extreme oversold)

    SHORT side (price above mean, z > 0) — symmetric mirror of the long thresholds:
        Tranche 1 activates when z >= -z1  (mild overbought)
        Tranche 2 activates when z >= -z2  (moderate overbought)
        Tranche 3 activates when z >= -z3  (extreme overbought)

    All active tranches (on whichever side is open) exit together when z reverts
    past ``exit_z`` toward flat — i.e. a long exits when z >= exit_z and a short
    exits when z <= -exit_z. With the default ``exit_z = 0.0`` both sides flatten
    on the mean cross. Long and short cannot be open simultaneously: a side only
    arms while z is on its half of the distribution, and the opposite side's
    arming threshold lies beyond the exit.

    Once a tranche activates it stays active until the group exit fires — it does
    not deactivate if z drifts back toward its own entry threshold (matching the
    parent node's "ratchet" tranche semantics).

    Output is the signed fraction of full position held, in [-1.0, +1.0].

    Parameters
    ----------
    lookback : int   Rolling window for mean and std (default 20).
    z1       : float First (long) entry threshold, must be < 0 (default -1.0).
                     The short side mirrors it at -z1 = +1.0.
    z2       : float Second (long) entry threshold, must be <= z1 (default -1.5).
    z3       : float Third (long) entry threshold, must be <= z2 (default -2.0).
    exit_z   : float Z-score magnitude at which all tranches exit toward the mean
                     (default 0.0 — revert to the mean). Must be >= 0.
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
        if exit_z < 0.0:
            raise ValueError(f"exit_z must be >= 0, got {exit_z}")

        self.lookback = lookback
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.exit_z = exit_z

        self.module_name = "scaled_z_mr_long_short"
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

        # Independent tranche activation state, per side
        self._long_t1: bool = False
        self._long_t2: bool = False
        self._long_t3: bool = False
        self._short_t1: bool = False
        self._short_t2: bool = False
        self._short_t3: bool = False

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

    def _reset_long(self) -> None:
        self._long_t1 = self._long_t2 = self._long_t3 = False

    def _reset_short(self) -> None:
        self._short_t1 = self._short_t2 = self._short_t3 = False

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._push(float(candle.close))
        self._n += 1

        if self._n < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        z = self._z_score()

        # Symmetric exit toward the mean: a long flattens once z has reverted up
        # to >= exit_z; a short flattens once z has reverted down to <= -exit_z.
        if z >= self.exit_z:
            self._reset_long()
        if z <= -self.exit_z:
            self._reset_short()

        # Arm LONG tranches on the oversold (negative-z) side.
        if z <= self.z1:
            self._long_t1 = True
        if z <= self.z2:
            self._long_t2 = True
        if z <= self.z3:
            self._long_t3 = True

        # Arm SHORT tranches on the overbought (positive-z) side — mirror of the
        # long thresholds (z1/z2/z3 are negative, so -z1/-z2/-z3 are positive).
        if z >= -self.z1:
            self._short_t1 = True
        if z >= -self.z2:
            self._short_t2 = True
        if z >= -self.z3:
            self._short_t3 = True

        long_units = self._long_t1 + self._long_t2 + self._long_t3
        short_units = self._short_t1 + self._short_t2 + self._short_t3
        out = _TRANCHE_WEIGHT * (long_units - short_units)

        self.output.append(out)
        return [out]
