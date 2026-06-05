"""
Z-Score RSI Signal Bias Node

Discrete signals from z-score RSI threshold crosses (same entry/exit
semantics as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`),
applied to the z-score of RSI from :class:`ZScoreRSIEngine`.

Thresholds are z-score levels (e.g. oversold -2, overbought +2), not 0–100 RSI.

Output: Rule-based -1, 0, or 1
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

from nodes import BiasNode
from nodes.mean_reversion.rsi.zscore_rsi import ZScoreRSIEngine
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle


class ZScoreRSISignal(BiasNode):
    """
    Z-Score RSI Signal - discrete signals from z-score threshold crosses.

    Uses the same RSI + rolling z-score series as :class:`~nodes.mean_reversion.rsi.zscore_rsi.ZScoreRSI`.
    Entry/exit logic matches :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`, but
    ``oversold`` / ``overbought`` are z-score levels (default -2 / +2).

    Output Range: -1, 0, or 1 (rule-based discrete signal)
    Neutral Value: 0 (during warmup)

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - zscore_period: Lookback for z-score of RSI (default: 100)
    - oversold: z-score at or below which a long entry cross is detected (default: -2)
    - overbought: z-score at or above which a short entry / long exit cross is detected (default: 2)
    - strategy_mode: "long", "short", or "long_short" (default: "long")
    - exit_policy: "threshold" or "threshold_or_bars" (default: "threshold_or_bars")
    - exit_bars: bars in position before fixed exit (default: 5)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"rsiPeriod", "zscorePeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        zscore_period: int = 100,
        oversold: float = -2.0,
        overbought: float = 2.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        self.rsi_period = rsi_period
        self.zscore_period = zscore_period
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

        self.module_name = "zscore_rsi_signal"
        self.output_features = ["signal"]
        self.params = {
            "rsiPeriod": rsi_period,
            "zscorePeriod": zscore_period,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        self.front_bad = rsi_period + zscore_period

        self._engine = ZScoreRSIEngine(rsi_period=rsi_period, zscore_period=zscore_period)

        self.position = 0
        self.bars_in_position = 0
        self.prev_z: Optional[float] = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _crossed_below(self, prev_z: float, z: float) -> bool:
        return prev_z > self.oversold and z <= self.oversold

    def _crossed_above(self, prev_z: float, z: float) -> bool:
        return prev_z < self.overbought and z >= self.overbought

    def _apply_position_rules(self, z: float) -> int:
        if self.prev_z is None:
            return 0

        crossed_below = self._crossed_below(self.prev_z, z)
        crossed_above = self._crossed_above(self.prev_z, z)

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

    def _compute_candle(self, candle: Candle) -> List[float]:
        z = self._engine.step(candle)

        if self._engine.n_candles < self.front_bad:
            self.position = 0
            self.bars_in_position = 0
            self.prev_z = None
            self.output.append(0.0)
            return [0.0]

        signal = self._apply_position_rules(z)
        self.prev_z = z
        self.output.append(float(signal))
        return [float(signal)]
