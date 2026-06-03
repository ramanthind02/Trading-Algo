"""
Market-regime filter + Laguerre RSI (γ=0) mean-reversion signal.

Port of CL Daily Strategy 1.22.280 (EasyLanguage / NinjaScript):

- Regime: ``MR = sum(H-L) / (HH - LL)`` smoothed, percentile-ranked vs its own MA.
- Momentum: 3-bar close gain/loss ratio (Laguerre RSI with γ=0).
- Long: LRSI crosses down through ``long_threshold`` while regime rank above its MA.
- Short: LRSI crosses up through ``short_threshold`` while regime rank below its MA.
- Exit: fixed bar hold (optional tick stop only when ``stop_loss_ticks`` > 0).
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle


def _round6(value: float) -> float:
    return round(value, 6)


def _laguerre_rsi_gamma_zero(closes: tuple[float, ...]) -> float:
    """LRSI from four consecutive closes (current bar is index 0)."""
    if len(closes) < 4:
        return 0.0
    cu = 0.0
    cd = 0.0
    for left, right in ((closes[0], closes[1]), (closes[1], closes[2]), (closes[2], closes[3])):
        if left >= right:
            cu += left - right
        else:
            cd += right - left
    total = cu + cd
    return cu / total if total > 0.0 else 0.0


def _percentile_rank(current: float, history: tuple[float, ...]) -> float:
    if not history:
        return 0.0
    rank = sum(1 for prior in history if _round6(current) > _round6(prior))
    return 100.0 * rank / len(history)


class RegimeLrsiSignal(BiasNode):
    """Discrete -1/0/1 from market-regime rank + Laguerre RSI crosses."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {
            "regimeMaPeriod",
            "hlSumPeriod",
            "hlRangePeriod",
            "mrAvgPeriod",
            "mrPrankPeriod",
        }
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        regime_ma_period: int = 58,
        hl_sum_period: int = 76,
        hl_range_period: int = 64,
        mr_avg_period: int = 29,
        mr_prank_period: int = 66,
        long_threshold: float = 0.6767,
        short_threshold: float = 0.3233,
        exit_bars: int = 1,
        stop_loss_ticks: int = 0,
        tick_size: float = 0.01,
        strategy_mode: DirectionInput = "long_short",
    ) -> None:
        super().__init__(ticker, tf)

        if regime_ma_period < 2:
            raise ValueError("regime_ma_period must be >= 2")
        if hl_sum_period < 1 or hl_range_period < 1:
            raise ValueError("hl_sum_period and hl_range_period must be >= 1")
        if mr_avg_period < 1 or mr_prank_period < 1:
            raise ValueError("mr_avg_period and mr_prank_period must be >= 1")
        if long_threshold <= short_threshold:
            raise ValueError("long_threshold must be > short_threshold")
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        if stop_loss_ticks < 0:
            raise ValueError("stop_loss_ticks must be >= 0")
        if tick_size <= 0.0:
            raise ValueError("tick_size must be > 0")

        self.regime_ma_period = regime_ma_period
        self.hl_sum_period = hl_sum_period
        self.hl_range_period = hl_range_period
        self.mr_avg_period = mr_avg_period
        self.mr_prank_period = mr_prank_period
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.exit_bars = exit_bars
        self.stop_loss_ticks = stop_loss_ticks
        self.tick_size = tick_size
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)

        self.module_name = "regime_lrsi_signal"
        self.output_features = ["signal"]
        self.params = {
            "regimeMaPeriod": regime_ma_period,
            "hlSumPeriod": hl_sum_period,
            "hlRangePeriod": hl_range_period,
            "mrAvgPeriod": mr_avg_period,
            "mrPrankPeriod": mr_prank_period,
            "longThreshold": long_threshold,
            "shortThreshold": short_threshold,
            "exitBars": exit_bars,
            "stopLossTicks": stop_loss_ticks,
            "tickSize": tick_size,
            "strategyMode": self.strategy_mode,
        }

        min_history = hl_sum_period + hl_range_period + 2
        self.front_bad = max(
            350,
            min_history + mr_avg_period + mr_prank_period + regime_ma_period + 5,
        )

        history_len = self.front_bad + 8
        self.highs: deque[float] = deque(maxlen=history_len)
        self.lows: deque[float] = deque(maxlen=history_len)
        self.closes: deque[float] = deque(maxlen=history_len)
        self.mr_raw_hist: deque[float] = deque(maxlen=history_len)
        self.mr_avg_hist: deque[float] = deque(maxlen=history_len)
        self.mr_prank_hist: deque[float] = deque(maxlen=history_len)

        self.n_candles = 0
        self.position = 0
        self.bars_in_position = 0
        self.entry_price = 0.0
        self.prev_lrsi = 0.0
        self._lrsi_ready = False

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        mode = coerce_direction(strategy_mode, field_name="strategy_mode").value
        if mode == "long":
            return "long"
        if mode == "short":
            return "short"
        return "long_short"

    def _recent_closes(self, offset: int, count: int) -> tuple[float, ...]:
        end = len(self.closes) - offset
        start = end - count
        if start < 0:
            return tuple()
        return tuple(list(self.closes)[start:end])

    def _compute_mr_avg(self) -> float | None:
        if len(self.highs) < self.hl_sum_period + self.hl_range_period + 2:
            return None

        hl_sum = sum(
            self.highs[-2 - idx] - self.lows[-2 - idx]
            for idx in range(self.hl_sum_period + 1)
        )
        range_highs = [self.highs[-1 - idx] for idx in range(1, self.hl_range_period + 1)]
        range_lows = [self.lows[-1 - idx] for idx in range(1, self.hl_range_period + 1)]
        highest = max(range_highs)
        lowest = min(range_lows)
        band = highest - lowest
        if band <= 0.0:
            return 0.0

        mr_raw = hl_sum / band
        self.mr_raw_hist.append(mr_raw)
        if len(self.mr_raw_hist) < self.mr_avg_period:
            return None
        recent_raw = list(self.mr_raw_hist)[-self.mr_avg_period :]
        mr_avg = sum(recent_raw) / self.mr_avg_period
        self.mr_avg_hist.append(mr_avg)
        return mr_avg

    def _compute_mr_prank(self, mr_avg: float) -> float | None:
        if len(self.mr_avg_hist) < self.mr_prank_period + 1:
            return None
        history = tuple(list(self.mr_avg_hist)[-self.mr_prank_period - 1 : -1])
        prank = _percentile_rank(mr_avg, history)
        self.mr_prank_hist.append(prank)
        return prank

    def _regime_flags(self, mr_prank: float) -> tuple[bool, bool]:
        if len(self.mr_prank_hist) < self.regime_ma_period:
            return False, False
        regime_ma = sum(list(self.mr_prank_hist)[-self.regime_ma_period :]) / self.regime_ma_period
        current = _round6(mr_prank)
        ma_val = _round6(regime_ma)
        return current > ma_val, current < ma_val

    def _current_lrsi(self) -> float:
        return _laguerre_rsi_gamma_zero(self._recent_closes(offset=1, count=4))

    def _stop_hit(self, candle: Candle) -> bool:
        if self.stop_loss_ticks <= 0 or self.position == 0:
            return False
        stop_distance = self.stop_loss_ticks * self.tick_size
        if self.position > 0:
            stop_price = self.entry_price - stop_distance
            return float(candle.low) <= stop_price
        stop_price = self.entry_price + stop_distance
        return float(candle.high) >= stop_price

    def _apply_exits(self, candle: Candle) -> None:
        if self.position == 0:
            return
        if self._stop_hit(candle):
            self.position = 0
            self.bars_in_position = 0
            self.entry_price = 0.0
            return
        if self.bars_in_position >= self.exit_bars:
            self.position = 0
            self.bars_in_position = 0
            self.entry_price = 0.0

    def _apply_entries(
        self,
        *,
        long_sig: bool,
        short_sig: bool,
        entry_price: float,
    ) -> None:
        if self.position != 0:
            return

        allow_long = self.strategy_mode in {"long", "long_short"}
        allow_short = self.strategy_mode in {"short", "long_short"}

        if long_sig and allow_long:
            self.position = 1
            self.bars_in_position = 0
            self.entry_price = entry_price
            return

        if short_sig and not long_sig and allow_short:
            self.position = -1
            self.bars_in_position = 0
            self.entry_price = entry_price

    def _compute_candle(self, candle: Candle) -> List[float]:
        self.n_candles += 1
        self.highs.append(float(candle.high))
        self.lows.append(float(candle.low))
        self.closes.append(float(candle.close))

        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        mr_avg = self._compute_mr_avg()
        if mr_avg is None:
            self.output.append(float(self.position))
            return [float(self.position)]

        mr_prank = self._compute_mr_prank(mr_avg)
        if mr_prank is None:
            self.output.append(float(self.position))
            return [float(self.position)]

        is_above_ma, is_below_ma = self._regime_flags(mr_prank)
        lrsi_cur = self._current_lrsi()
        lrsi_prv = self.prev_lrsi if self._lrsi_ready else lrsi_cur

        long_sig = (
            lrsi_cur < self.long_threshold
            and lrsi_prv > self.long_threshold
            and is_above_ma
        )
        short_sig = (
            lrsi_cur > self.short_threshold
            and lrsi_prv < self.short_threshold
            and is_below_ma
        )

        if self.position != 0:
            self.bars_in_position += 1

        self._apply_exits(candle)
        self._apply_entries(
            long_sig=long_sig,
            short_sig=short_sig,
            entry_price=float(candle.close),
        )

        self.prev_lrsi = lrsi_cur
        self._lrsi_ready = True
        signal = float(self.position)
        self.output.append(signal)
        return [signal]
