"""Single-ticker, long-only monthly rebalancing flow node.

Decomposes the Robot Wealth monthly SPY/TLT (here ES/TLT) rebalancing strategy
into one configurable, **single-ticker, long-only** bias node. The node trades
only its own ``ticker`` and compares it against one ``peer`` (cross-ticker) over
the first N trading days of each month, then expresses one or both of two
independent month-end "flows":

- **Reversal** (``RebalancingFlow.REVERSAL``): month-end rebalance supply makes
  the first-window *loser* catch up. If the peer outperformed ``ticker`` over the
  first ``decision_trading_day`` trading days, go **long ``ticker`` from the
  decision day through end of month** (signal = 1); otherwise flat.
- **Continuation** (``RebalancingFlow.CONTINUATION``): the first-window move
  resumes at the turn of the month. If ``ticker`` outperformed the peer, stay
  flat until the EOM window (within ``eom_lead_days`` calendar days of month end),
  then go **long ``ticker``** and hold for the first ``continuation_hold`` trading
  days of the next month (signal = 1); otherwise flat.

``RebalancingFlow.BOTH`` enables both legs (only one fires per month, since the
decision is either self-won or peer-won).

By default the node is **long-only** (signal ∈ {0, 1}). With
``direction=RebalancingDirection.LONG_SHORT`` it emits ``+1`` / ``-1``: it shorts
the side it would otherwise sit flat on (short the first-window winner during the
reversal window; short the loser during the continuation window).

Reproduces the legacy nodes with slide-accurate (trading-day / EOM-relative)
timing:

- ``RebalancingFlowNode(Ticker.ES,  D, ["TLT"], flow=BOTH)``      ~ old ``RebalancingNode``
- ``RebalancingFlowNode(Ticker.TLT, D, ["ES"],  flow=REVERSAL)``  ~ old ``RebalancingCrossNode``

Example
-------
>>> from utils.data.cross_ticker_store import CrossTickerDataStore
>>> store = CrossTickerDataStore.get_instance()
>>> store.load(Ticker.TLT, TimeFrame.D)
>>>
>>> node = RebalancingFlowNode(Ticker.ES, TimeFrame.D, cross_tickers=["TLT"], flow="reversal")
>>> signal = node.add_candle(es_candle)
"""

from __future__ import annotations

import calendar
from enum import Enum
from typing import List, Optional, Tuple, Union

from nodes import BiasNode
from utils.cache.runtime.central_cache_errors import ArtifactMissingError
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.data.cross_ticker_store import CrossTickerDataStore


class RebalancingFlow(Enum):
    """Which month-end flow leg(s) the node trades (long-only)."""

    REVERSAL = "reversal"
    """Long self when the peer won the first window; decision day → EOM."""

    CONTINUATION = "continuation"
    """Long self when self won the first window; EOM → next month's open."""

    BOTH = "both"
    """Both legs (reversal when self lost, continuation when self won)."""


def _coerce_flow(raw: Union[RebalancingFlow, str]) -> RebalancingFlow:
    if isinstance(raw, RebalancingFlow):
        return raw
    if not isinstance(raw, str):
        raise TypeError(
            f"flow must be RebalancingFlow or str, got {type(raw).__name__}."
        )
    key = raw.strip().lower()
    aliases = {
        "reversal": RebalancingFlow.REVERSAL,
        "rev": RebalancingFlow.REVERSAL,
        "continuation": RebalancingFlow.CONTINUATION,
        "cont": RebalancingFlow.CONTINUATION,
        "both": RebalancingFlow.BOTH,
        "all": RebalancingFlow.BOTH,
    }
    if key not in aliases:
        raise ValueError(
            f'Invalid flow "{raw}". Use reversal, continuation, or both.'
        )
    return aliases[key]


class RebalancingDirection(Enum):
    """Whether the node trades only the long side or both sides."""

    LONG_ONLY = "long_only"
    """Emit ``1`` (long) or ``0`` (flat) — original behavior."""

    LONG_SHORT = "long_short"
    """Emit ``+1`` / ``-1``: long the bet's favoured side, short the other.

    Reversal window: long the first-window loser, **short the winner**.
    Continuation window: long the first-window winner, **short the loser**.
    """


def _coerce_direction(raw: Union[RebalancingDirection, str]) -> RebalancingDirection:
    if isinstance(raw, RebalancingDirection):
        return raw
    if not isinstance(raw, str):
        raise TypeError(
            f"direction must be RebalancingDirection or str, got {type(raw).__name__}."
        )
    key = raw.strip().lower()
    aliases = {
        "long_only": RebalancingDirection.LONG_ONLY,
        "long": RebalancingDirection.LONG_ONLY,
        "longonly": RebalancingDirection.LONG_ONLY,
        "long_short": RebalancingDirection.LONG_SHORT,
        "longshort": RebalancingDirection.LONG_SHORT,
        "ls": RebalancingDirection.LONG_SHORT,
    }
    if key not in aliases:
        raise ValueError(
            f'Invalid direction "{raw}". Use long_only or long_short.'
        )
    return aliases[key]


def _coerce_positive_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


class RebalancingFlowNode(BiasNode):
    """Single-ticker, long-only monthly rebalancing flow node.

    Parameters
    ----------
    ticker : Ticker
        Primary ticker (candles streamed via ``add_candle``); the only ticker traded.
    tf : TimeFrame
        Timeframe.
    cross_tickers : list[str] | None
        Peer ticker dependency (first entry is used). Defaults to ``["ES"]`` when
        ``ticker`` is ``TLT`` else ``["TLT"]``.
    flow : RebalancingFlow | str
        Which flow leg(s) to trade. Defaults to ``RebalancingFlow.BOTH``.
    direction : RebalancingDirection | str
        ``LONG_ONLY`` (default; emit 1/0) or ``LONG_SHORT`` (emit +1/-1: long the
        bet's favoured side, short the other). Defaults to ``LONG_ONLY``.
    decision_trading_day : int
        Trading day of the month on which the first-window relative performance is
        evaluated (default ``15``).
    eom_lead_days : int
        Continuation leg enters when within this many *calendar* days of month end
        (default ``3``). ``0`` disables the in-month leg (next-month hold only).
    continuation_hold : int
        Number of trading days into the next month the continuation leg is held
        (default ``5``).
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        cross_tickers: list[str] | None = None,
        flow: Union[RebalancingFlow, str] = RebalancingFlow.BOTH,
        direction: Union[RebalancingDirection, str] = RebalancingDirection.LONG_ONLY,
        decision_trading_day: int = 15,
        eom_lead_days: int = 3,
        continuation_hold: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if cross_tickers is None:
            configured_cross: list[str] = ["ES" if ticker == Ticker.TLT else "TLT"]
        else:
            configured_cross = cross_tickers
        if not configured_cross:
            raise ValueError("cross_tickers must include at least one ticker symbol.")
        normalized_cross: list[str] = []
        for idx, raw in enumerate(configured_cross):
            if not isinstance(raw, str):
                raise TypeError(
                    f"cross_tickers[{idx}] must be str, got {type(raw).__name__}."
                )
            normalized = raw.strip().upper()
            if normalized:
                normalized_cross.append(normalized)
        if not normalized_cross:
            raise ValueError(
                "cross_tickers must include at least one non-empty ticker symbol."
            )
        cross_ticker_name = normalized_cross[0]
        try:
            self.cross_ticker = Ticker[cross_ticker_name]
        except KeyError as exc:
            raise ValueError(f"Unknown cross ticker '{cross_ticker_name}'.") from exc
        if self.cross_ticker == ticker:
            raise ValueError(
                f"Peer ticker '{cross_ticker_name}' must differ from the node ticker "
                f"'{ticker.name}'."
            )

        self.cross_tickers = tuple(normalized_cross)

        self._flow = _coerce_flow(flow)
        self._reversal_enabled = self._flow in (
            RebalancingFlow.REVERSAL,
            RebalancingFlow.BOTH,
        )
        self._continuation_enabled = self._flow in (
            RebalancingFlow.CONTINUATION,
            RebalancingFlow.BOTH,
        )
        self._direction = _coerce_direction(direction)
        self._long_short = self._direction is RebalancingDirection.LONG_SHORT
        # Value emitted on the "off" side of a bet: 0 (long-only) or -1 (long/short).
        self._short_val = -1.0 if self._long_short else 0.0
        self._decision_trading_day = _coerce_positive_int(
            "decision_trading_day", decision_trading_day, minimum=1
        )
        self._eom_lead_days = _coerce_positive_int(
            "eom_lead_days", eom_lead_days, minimum=0
        )
        self._continuation_hold = _coerce_positive_int(
            "continuation_hold", continuation_hold, minimum=1
        )

        self.module_name = "rebalancing_flow"
        self.output_features = ["signal"]
        self.params = {
            "cross_tickers": list(self.cross_tickers),
            "flow": self._flow.value,
            "direction": self._direction.value,
            "decision_trading_day": self._decision_trading_day,
            "eom_lead_days": self._eom_lead_days,
            "continuation_hold": self._continuation_hold,
        }
        self.front_bad = 0

        self._store = CrossTickerDataStore.get_instance()

        # Monthly state (reset on month change, except the carry tag + prev closes)
        self._current_period: Optional[Tuple[int, int]] = None
        self._month_start_self: Optional[float] = None
        self._month_start_peer: Optional[float] = None
        self._decision: Optional[str] = None
        self._trading_day_count: int = 0
        # Prior month's last close, used as the first-window performance basis (the
        # conventional monthly-return base: prior close → TD15 close).
        self._prev_self_close: Optional[float] = None
        self._prev_peer_close: Optional[float] = None
        # (year, month) that armed an active continuation carry; consumed in the
        # following month. ``None`` when no carry is pending.
        self._carry_period: Optional[Tuple[int, int]] = None
        # Whether the armed carry's month was self-won (long the winner) or peer-won
        # (short the loser, long/short only).
        self._carry_won: bool = False

        # NOTE: deliberately NOT calling ``_init_cache_after_params()`` — output is
        # state-dependent (accumulated month history), so per-candle/cross-run
        # caching keyed only on the candle would be unsound.
        self.ensure_standardized_columns()

    # ------------------------------------------------------------------

    def _within_eom_window(self, year: int, month: int, day: int) -> bool:
        """True when ``day`` is within ``eom_lead_days`` calendar days of month end."""
        last_day = calendar.monthrange(year, month)[1]
        return (last_day - day) < self._eom_lead_days

    def _compute_candle(self, candle: Candle) -> List[float]:
        dt = candle.datetime
        year, month = dt.year, dt.month

        # Detect month change → reset observation state (carry tag + prev closes survive)
        if (year, month) != self._current_period:
            self._current_period = (year, month)
            self._decision = None
            self._trading_day_count = 0
            # Basis for first-window performance is the prior month's last close.
            self._month_start_self = self._prev_self_close
            self._month_start_peer = self._prev_peer_close

        self._trading_day_count += 1

        # Fetch peer candle
        try:
            cross_candle = self._store.query_candle(self.cross_ticker, candle.tf, dt)
        except ArtifactMissingError:
            self._prev_self_close = candle.close  # keep self-close continuity
            return [0.0]

        # Very first observation ever (no prior close): fall back to this candle's close.
        if self._month_start_self is None:
            self._month_start_self = candle.close
        if self._month_start_peer is None:
            self._month_start_peer = cross_candle.close

        signal = self._signal_for_candle(candle, cross_candle, dt, year, month)

        self._prev_self_close = candle.close
        self._prev_peer_close = cross_candle.close
        return [signal]

    def _signal_for_candle(
        self, candle: Candle, cross_candle: Candle, dt: object, year: int, month: int
    ) -> float:
        # Continuation carry-over: consumed in the month AFTER it was armed.
        # long the winner (carry_won) / short the loser (long/short only).
        if self._carry_period is not None and self._carry_period != (year, month):
            if self._trading_day_count <= self._continuation_hold:
                won = self._carry_won
                if self._trading_day_count >= self._continuation_hold:
                    self._carry_period = None
                return 1.0 if won else self._short_val
            self._carry_period = None  # hold window elapsed

        # Observation window: flat (no position) until the decision trading day
        if self._decision is None and self._trading_day_count < self._decision_trading_day:
            return 0.0

        # Decision on / after the configured trading day
        if self._decision is None:
            self_pct = (candle.close - self._month_start_self) / self._month_start_self
            peer_pct = (cross_candle.close - self._month_start_peer) / self._month_start_peer
            self._decision = "peer_won" if peer_pct > self_pct else "self_won"
            # Arm next-month continuation. Long-only arms only on self-won (long the
            # winner); long/short arms every month (also short the loser next month).
            if self._continuation_enabled and (self._decision == "self_won" or self._long_short):
                self._carry_period = (year, month)
                self._carry_won = self._decision == "self_won"
            # fall through to act on the decision day itself

        won = self._decision == "self_won"

        # Reversal leg, long side: self lost → long self from decision day through EOM
        # (kept first so the long-only reversal holds through the EOM window unchanged).
        if not won and self._reversal_enabled:
            return 1.0

        # Continuation leg, in-month EOM window: long the winner / short the loser.
        if self._continuation_enabled and self._within_eom_window(year, month, dt.day):
            return 1.0 if won else self._short_val

        # Reversal leg, short side: self won → short the winner (long/short only).
        if won and self._reversal_enabled:
            return self._short_val

        # Continuation-only flow, pre-EOM window: flat.
        return 0.0
