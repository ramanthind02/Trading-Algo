"""Stacked SMA regime for 1–5 simple moving averages (MA complexity / overfitting experiments).

- **One MA:** price vs SMA; strength depends on :class:`~utils.core.enums.PositionMode`.
- **Two MAs:** discrete **+1 / 0 / −1** from ``SMA_fast`` vs ``SMA_slow`` (shorter vs longer period),
  then mode clamp (same contract as :class:`~nodes.regime.sma.sma_regime_signal.SmaRegimeSignalNode`).
- **Three or more MAs:** continuous strength in **[−2, 2]** from the fraction of pairwise
  comparisons with ``SMA_fast > SMA_slow``: e.g. three MAs → three pairs → **1** vote →
  ``2/3`` scaling → **~0.67** in ``LONG_ONLY`` / ``SHORT_ONLY`` strength form, **2** votes → **~1.33**;
  ``LONG_SHORT`` maps vote share linearly to **[−2, 2]** (50 % votes → **0**).

Optional **realized volatility filter** (``vol_period`` and ``vol_threshold`` both set): annualized
sample std of the last ``vol_period`` simple daily returns must be **strictly below**
``vol_threshold`` or the signal is flattened to **0**.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from itertools import combinations
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import PositionMode, Ticker, TimeFrame
from utils.core.models import Candle

_SIGNAL_ABS_MAX = 2.0


def stacked_sma_period_combos(layer_count: int, candidates: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """
    Strictly increasing period tuples of length ``layer_count`` from ``candidates``.

    Use this to build a bias-spec catalog for grid search without invalid
    ``period_1 >= period_2`` combinations from a full Cartesian product.
    """
    if not 1 <= layer_count <= 5:
        raise ValueError("layer_count must be between 1 and 5")
    vals = tuple(sorted({int(x) for x in candidates}))
    if len(vals) < layer_count:
        raise ValueError("candidates must contain at least layer_count distinct values")
    return tuple(combinations(vals, layer_count))


def _coerce_position_mode(raw: PositionMode | str) -> PositionMode:
    if isinstance(raw, PositionMode):
        return raw
    return PositionMode(raw)


def _annualized_realized_vol(closes: tuple[float, ...], vol_period: int) -> float:
    """
    Annualized realized vol: sample std of the last ``vol_period`` simple returns,
    times ``sqrt(252)``. Requires ``len(closes) >= vol_period + 1``.
    """
    if vol_period < 2:
        raise ValueError("vol_period must be >= 2")
    if len(closes) < vol_period + 1:
        raise ValueError("not enough closes for vol_period")
    window = closes[-(vol_period + 1) :]
    rets = tuple(
        (window[i] / window[i - 1] - 1.0)
        for i in range(1, len(window))
        if window[i - 1] != 0.0
    )
    if len(rets) < vol_period:
        raise ValueError("invalid closes (zero prior) shortened return window")
    if len(rets) < 2:
        return 0.0
    sigma = statistics.stdev(rets)
    return float(sigma * math.sqrt(252.0))


def _pairwise_long_votes(closes: tuple[float, ...], periods: tuple[int, ...]) -> tuple[int, int]:
    """Count pairs (fast, slow) with ``SMA_fast > SMA_slow`` and total pair count."""
    smas = {p: sum(closes[-p:]) / float(p) for p in periods}
    pairs = tuple(combinations(periods, 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return 0, 0
    votes = sum(1 for fast_p, slow_p in pairs if smas[fast_p] > smas[slow_p])
    return votes, n_pairs


def _multi_ma_raw_strength(closes: tuple[float, ...], periods: tuple[int, ...], mode: PositionMode) -> float:
    """Map pairwise vote share to a float in ``[-_SIGNAL_ABS_MAX, _SIGNAL_ABS_MAX]`` (three+ MAs)."""
    votes, n_pairs = _pairwise_long_votes(closes, periods)
    if n_pairs == 0:
        return 0.0
    frac = votes / float(n_pairs)
    match mode:
        case PositionMode.LONG_SHORT:
            raw = _SIGNAL_ABS_MAX * 2.0 * (frac - 0.5)
        case PositionMode.LONG_ONLY:
            raw = _SIGNAL_ABS_MAX * frac
        case PositionMode.SHORT_ONLY:
            raw = _SIGNAL_ABS_MAX * (1.0 - frac)
    return max(-_SIGNAL_ABS_MAX, min(_SIGNAL_ABS_MAX, float(raw)))


def _two_ma_raw_signal(closes: tuple[float, ...], periods: tuple[int, ...]) -> float:
    """Discrete +1 / 0 / −1 from the two SMAs (shorter period = fast)."""
    fast_p, slow_p = periods[0], periods[1]
    s_fast = sum(closes[-fast_p:]) / float(fast_p)
    s_slow = sum(closes[-slow_p:]) / float(slow_p)
    if s_fast > s_slow:
        return 1.0
    if s_fast < s_slow:
        return -1.0
    return 0.0


def _one_ma_raw_signal(close: float, closes: tuple[float, ...], period: int) -> float:
    sma = sum(closes[-period:]) / float(period)
    if close > sma:
        return 1.0
    if close < sma:
        return -1.0
    return 0.0


def _apply_position_mode(raw: float, mode: PositionMode) -> float:
    if mode == PositionMode.LONG_ONLY and raw < 0.0:
        return 0.0
    if mode == PositionMode.SHORT_ONLY and raw > 0.0:
        return 0.0
    return raw


def _apply_realized_vol_gate(
    base: float,
    closes: tuple[float, ...],
    vol_period: int | None,
    vol_threshold: float | None,
) -> float:
    if vol_period is None or vol_threshold is None:
        return base
    rv = _annualized_realized_vol(closes, vol_period)
    return base if rv < vol_threshold else 0.0


def _normalize_vol_params(
    vol_period: int | None,
    vol_threshold: float | None,
) -> tuple[int | None, float | None]:
    has_p = vol_period is not None
    has_t = vol_threshold is not None
    if has_p ^ has_t:
        raise ValueError(
            "stacked_sma_long_only: vol_period and vol_threshold must both be set or both omitted"
        )
    if not has_p:
        return None, None
    vp = int(vol_period)  # type: ignore[arg-type]
    vt = float(vol_threshold)  # type: ignore[arg-type]
    if vp < 2:
        raise ValueError("stacked_sma_long_only: vol_period must be >= 2 when using vol filter")
    if vt <= 0.0:
        raise ValueError("stacked_sma_long_only: vol_threshold must be > 0 when using vol filter")
    return vp, vt


def _period_tuple_from_init(
    period_1: int,
    period_2: int | None,
    period_3: int | None,
    period_4: int | None,
    period_5: int | None,
) -> tuple[int, ...]:
    if period_2 is None and any(x is not None for x in (period_3, period_4, period_5)):
        raise ValueError(
            "stacked_sma_long_only: period_2 must be set before period_3, period_4, or period_5"
        )
    if period_3 is None and any(x is not None for x in (period_4, period_5)):
        raise ValueError("stacked_sma_long_only: period_3 must be set before period_4 or period_5")
    if period_4 is None and period_5 is not None:
        raise ValueError("stacked_sma_long_only: period_4 must be set before period_5")
    layers = [period_1]
    if period_2 is not None:
        layers.append(period_2)
    if period_3 is not None:
        layers.append(period_3)
    if period_4 is not None:
        layers.append(period_4)
    if period_5 is not None:
        layers.append(period_5)
    p = tuple(int(x) for x in layers)
    if any(x < 2 for x in p):
        raise ValueError("each period must be >= 2")
    if any(p[i] >= p[i + 1] for i in range(len(p) - 1)):
        raise ValueError("periods must be strictly increasing (fast < medium < slow, …)")
    return p


class StackedSmaLongOnlyNode(BiasNode):
    """
    Stacked SMA signal for 1–5 periods; see module docstring.

    :attr:`mode` is :class:`~utils.core.enums.PositionMode` (default ``LONG_ONLY`` for
    backward compatibility).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"period_1", "period_2", "period_3", "period_4", "period_5", "vol_period"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period_1: int,
        period_2: int | None = None,
        period_3: int | None = None,
        period_4: int | None = None,
        period_5: int | None = None,
        vol_period: int | None = None,
        vol_threshold: float | None = None,
        mode: PositionMode | str = PositionMode.LONG_ONLY,
    ) -> None:
        super().__init__(ticker, tf)
        self.periods = _period_tuple_from_init(period_1, period_2, period_3, period_4, period_5)
        vp, vt = _normalize_vol_params(vol_period, vol_threshold)
        self._vol_period = vp
        self._vol_threshold = vt
        self.mode = _coerce_position_mode(mode)
        self.module_name = "stacked_sma_long_only"
        self.output_features = ["signal"]
        self.params = {f"period_{i + 1}": self.periods[i] for i in range(len(self.periods))}
        self.params["mode"] = self.mode
        if vp is not None:
            assert vt is not None
            self.params["vol_period"] = vp
            self.params["vol_threshold"] = vt
        ma_need = max(self.periods)
        vol_need = (vp + 1) if vp is not None else 0
        self.front_bad = max(ma_need, vol_need)
        self._closes: deque[float] = deque(maxlen=self.front_bad)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._closes.append(float(candle.close))
        if len(self._closes) < self.front_bad:
            out = 0.0
        else:
            cl = tuple(self._closes)
            c = float(candle.close)
            n = len(self.periods)
            if n == 1:
                p1 = self.periods[0]
                sma = sum(cl[-p1:]) / float(p1)
                match self.mode:
                    case PositionMode.LONG_ONLY:
                        raw = 1.0 if c > sma else 0.0
                    case PositionMode.SHORT_ONLY:
                        raw = -1.0 if c < sma else 0.0
                    case PositionMode.LONG_SHORT:
                        raw = _one_ma_raw_signal(c, cl, p1)
            elif n == 2:
                raw = _two_ma_raw_signal(cl, self.periods)
                raw = _apply_position_mode(raw, self.mode)
            else:
                raw = _multi_ma_raw_strength(cl, self.periods, self.mode)
            out = _apply_realized_vol_gate(raw, cl, self._vol_period, self._vol_threshold)
        self.output.append(out)
        return [out]
