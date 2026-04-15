"""
ATR percentile filter — binary low-volatility gate from rank within a rolling window.

Computes SMA(ATR) over ``atr_period`` true ranges, then ranks the current reading
(``atr`` or ``atrPct``) against the last ``lookback`` values. Emits ``1.0`` when
the empirical bottom fraction is at or below ``max_rank_fraction`` (e.g. ``0.3``
= current metric is in the lowest ~30% of the window), else ``0.0``.

Intended for ``filter_gate`` / ``filter_and_signal`` (non-zero = gate open).
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

import numpy as np

from nodes import BiasNode
from utils.compute.fast_nodes import compute_atr_fast
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def _coerce_rank_metric(raw: str) -> str:
    key = str(raw).strip().lower().replace("-", "_")
    if key in {"atr", "atr_pct", "atrpct"}:
        return "atr_pct" if key != "atr" else "atr"
    raise ValueError(f'rank_metric must be "atr" or "atr_pct", got "{raw}".')


def _strict_less_rank_fraction(window: np.ndarray, current: float) -> float:
    """Share of window values strictly below ``current`` in ``[0, 1]`` (``n>=2``)."""
    n = int(window.shape[0])
    if n < 2:
        return 0.0
    less = int(np.sum(window < current))
    return float(less) / float(n - 1)


class AtrPercentileFilterNode(BiasNode):
    """
    Low-volatility gate: current ATR (or ATR%) is in the bottom fraction of its
    rolling history.

    ``max_rank_fraction=0.3`` → gate open when at most ~30% of window values are
    strictly below the current reading (current is among the lower third).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"atr_period", "lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        atr_period: int = 14,
        lookback: int = 252,
        max_rank_fraction: float = 0.3,
        rank_metric: str = "atr_pct",
    ) -> None:
        super().__init__(ticker, tf)

        if atr_period < 1:
            raise ValueError("atr_period must be >= 1.")
        if lookback < 2:
            raise ValueError("lookback must be >= 2 for percentile rank.")
        if not 0.0 < max_rank_fraction <= 1.0:
            raise ValueError("max_rank_fraction must be in (0, 1].")

        self.atr_period = atr_period
        self.lookback = lookback
        self.max_rank_fraction = float(max_rank_fraction)
        self._rank_metric = _coerce_rank_metric(rank_metric)

        self.module_name = "atr_percentile_filter"
        self.output_features = ["signal"]
        self.params = {
            "atr_period": atr_period,
            "lookback": lookback,
            "max_rank_fraction": self.max_rank_fraction,
            "rank_metric": self._rank_metric,
        }
        self.front_bad = atr_period + lookback - 1

        self._prev_close: float = -1.0
        self._true_ranges = np.zeros(atr_period, dtype=np.float64)
        self._buffer_idx = 0
        self._n_filled = 0
        self._metric_history: deque[float] = deque(maxlen=lookback)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        atr, atr_pct_out, self._buffer_idx, self._n_filled = compute_atr_fast(
            float(candle.high),
            float(candle.low),
            float(candle.close),
            self._prev_close,
            self._true_ranges,
            self._buffer_idx,
            self._n_filled,
            self.atr_period,
        )
        self._prev_close = float(candle.close)

        if self._n_filled < self.atr_period:
            out = 0.0
            self.output.append(out)
            return [out]

        metric = (
            float(atr_pct_out)
            if self._rank_metric == "atr_pct"
            else float(atr)
        )
        self._metric_history.append(metric)

        if len(self._metric_history) < self.lookback:
            out = 0.0
            self.output.append(out)
            return [out]

        arr = np.asarray(self._metric_history, dtype=np.float64)
        rank_frac = _strict_less_rank_fraction(arr, metric)
        out = 1.0 if rank_frac <= self.max_rank_fraction else 0.0
        self.output.append(out)
        return [out]
