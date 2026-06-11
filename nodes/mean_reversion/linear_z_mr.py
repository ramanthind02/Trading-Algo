from __future__ import annotations

from collections import deque
from typing import ClassVar, List, Optional

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class LinearZMeanReversion(BiasNode):
    """
    Linear z-score mean reversion with an optional rate-of-change regime filter.

    A faithful port of the Robot Wealth / Ernie Chan AUD/NZD mean-reversion model
    (Longmore, 2015): hold a position negatively proportional to the moving
    z-score of price, with the lookback set to the half-life of mean reversion.

        z      = (close - mean(close, lookback)) / std(close, lookback)
        signal = clip(-z * scale, -cap, +cap)

    Positive signal = long  (price below mean, expect reversion up).
    Negative signal = short (price above mean, expect reversion down).

    With ``scale=1.0`` and ``cap=2.0`` the position is full (+/-2) at +/-2 sigma,
    matching the framework forecast cap. This is the linear analogue of
    :class:`~nodes.mean_reversion.tanh_z_mr.TanhZMeanReversion`, whose ``k``
    saturates the same band smoothly via ``tanh``.

    Regime filter (optional)
    ------------------------
    The blog tames the strategy's worst drawdown — which occurs when AUD/NZD
    stops mean-reverting and trends to a new range — by flattening when the
    moving average changes too fast. We gate on the *percent* rate of change of
    the rolling mean (scale-invariant, unlike the blog's price-specific
    ``100 * (ma - ma_prev)`` form):

        sma_roc = |SMA[t] / SMA[t-1] - 1|
        if roc_threshold is not None and sma_roc >= roc_threshold: signal = 0

    ``roc_threshold=None`` disables the filter (pure linear ``-z``).

    Parameters
    ----------
    lookback      : int            Rolling window for mean and std (default 150,
                                   the blog's ~half-life/2). Must be >= 2.
    scale         : float          Sensitivity of -z before clipping (default 1.0).
    cap           : float          Forecast cap, |signal| <= cap (default 2.0).
    roc_threshold : float | None   Percent-ROC of the SMA above which the signal
                                   is flattened (default None = filter off).
                                   Must be > 0 when set.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 150,
        scale: float = 1.0,
        cap: float = 2.0,
        roc_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if cap <= 0.0:
            raise ValueError(f"cap must be > 0, got {cap}")
        if roc_threshold is not None and roc_threshold <= 0.0:
            raise ValueError(f"roc_threshold must be > 0 when set, got {roc_threshold}")

        self.lookback = lookback
        self.scale = scale
        self.cap = cap
        self.roc_threshold = roc_threshold

        self.module_name = "linear_z_mr"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "scale": scale,
            "cap": cap,
            "rocThreshold": roc_threshold,
        }

        # The filter reads the previous full-window SMA, so it needs one extra
        # warmup bar; the plain signal only needs a full window.
        self.front_bad = lookback + 1 if roc_threshold is not None else lookback

        # O(1) running sum / sum-of-squares for mean and variance.
        self._closes: deque[float] = deque(maxlen=lookback)
        self._price_sum: float = 0.0
        self._price_sum_sq: float = 0.0
        self._n: int = 0
        self._prev_sma: Optional[float] = None

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

        full = len(self._closes) == self.lookback
        cur_sma = (self._price_sum / self.lookback) if full else None

        if self._n < self.front_bad:
            if full:
                self._prev_sma = cur_sma
            self.output.append(0.0)
            return [0.0]

        z = self._z_score()
        raw = -z * self.scale
        signal = max(-self.cap, min(self.cap, raw))

        if (
            self.roc_threshold is not None
            and self._prev_sma is not None
            and self._prev_sma != 0.0
            and cur_sma is not None
        ):
            sma_roc = abs(cur_sma / self._prev_sma - 1.0)
            if sma_roc >= self.roc_threshold:
                signal = 0.0

        self._prev_sma = cur_sma
        self.output.append(signal)
        return [signal]
