"""Pure rollover-cycle scheduling in BROKER wall-clock time.

The MT5 **financing rollover is 00:00 broker** for every symbol on these CFD
venues (see ``data_platform/providers/mt5/cfd_candles`` — *"the daily financing
rollover is 00:00 broker"*). Each symbol additionally has its own session
**reopen** time (most index/metal/energy CFDs: 01:00 broker, after the
00:00–01:00 dead-zone; an FX pair: 01:00; a US-stock CFD: ~16:30 broker).

The overlay (``execution.rollover_overlay``) wants two events per broker-day:

* **EXIT** at ``exit_lead_min`` before the 00:00-broker rollover (default 15 →
  23:45 broker) — flatten the swap-negative legs that are worth overlaying.
* **ENTRY** at each symbol's **reopen + entry_settle_min** — re-establish /
  rebalance to the fresh forecast (delta-based).

Everything here is **pure** and computed in **broker wall-clock** (a naive
``datetime`` whose fields are the broker's local clock). The caller derives
``broker_now`` from a fresh MT5 tick (``tick.time`` IS broker wall-clock),
**never** the host clock — the host may be mis-set, and the broker is the market
truth. This also makes the schedule DST-correct for free: the tick already
reflects the broker's current EET/EEST.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

ROLLOVER_HOUR = 0   # 00:00 broker — the financing rollover boundary
ROLLOVER_MIN = 0

# A weekday the FX/CFD week is closed (Saturday). Sunday evening (broker Monday
# 00:00+) is the weekend reopen; we treat it as a normal session whose entry is
# gated by an actual fresh quote, so no special weekend branch is needed here.
_SATURDAY = 5


@dataclass(frozen=True)
class SymbolSession:
    """A symbol's daily session open/close in BROKER wall-clock (HH:MM)."""

    open_h: int
    open_m: int
    close_h: int
    close_m: int

    @property
    def open_time(self) -> time:
        return time(self.open_h, self.open_m)

    @property
    def close_time(self) -> time:
        return time(self.close_h, self.close_m)


@dataclass(frozen=True)
class RolloverParams:
    """Universal timing knobs (broker minutes); no per-ticker values."""

    exit_lead_min: int = 15        # open the exit window this long before 00:00 broker (T-15)
    # Fallback reopen when a symbol has no inferred session (most CFDs reopen 01:00 broker).
    default_reopen_h: int = 1
    default_reopen_m: int = 0
    # How long after a symbol's reopen we keep trying to enter it (a trading-day band).
    entry_window_min: int = 180


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(ROLLOVER_HOUR, ROLLOVER_MIN))


def next_rollover(broker_now: datetime) -> datetime:
    """The next 00:00-broker financing boundary strictly after ``broker_now``.

    At 23:45 broker on day D this is D+1 00:00; at 00:30 broker on day D it is
    D+1 00:00 (the current day's boundary already passed).
    """
    return _midnight(broker_now.date()) + timedelta(days=1)


def last_rollover(broker_now: datetime) -> datetime:
    """The most recent 00:00-broker boundary at/before ``broker_now``.

    This is the boundary whose **new session** ``broker_now`` belongs to — its
    date is the session date for entry latching.
    """
    return _midnight(broker_now.date())


def exit_time(broker_now: datetime, params: RolloverParams) -> datetime:
    """Broker wall-clock of the T-lead exit, for the upcoming rollover."""
    return next_rollover(broker_now) - timedelta(minutes=params.exit_lead_min)


def in_exit_window(broker_now: datetime, params: RolloverParams) -> bool:
    """True from the T-lead exit time up to the 00:00 rollover: ``[exit_time, rollover)``.

    Starts at ``exit_time`` (e.g. 23:45) — NOT just before midnight — because the
    CFD session itself closes a few minutes before the financing rollover (FTMO
    ~23:49). The first poll in this band, while the market is still open, runs the
    (idempotent, position-driven) exit pass; the quote-freshness gate skips any leg
    whose market has already closed.
    """
    return exit_time(broker_now, params) <= broker_now < next_rollover(broker_now)


def exit_boundary_date(broker_now: datetime) -> date:
    """Latch key for the exit pass = the date of the rollover it precedes."""
    return next_rollover(broker_now).date()


def session_date(broker_now: datetime) -> date:
    """Latch key for the entry pass = the date of the session ``broker_now`` is in."""
    return last_rollover(broker_now).date()


def reopen_at(
    broker_now: datetime,
    session: SymbolSession | None,
    settle_min: int,
    params: RolloverParams,
) -> datetime:
    """Broker wall-clock when ``broker_now``'s session may be (re)entered for a symbol.

    = the symbol's session open (or the default reopen) on the current session
    date, plus the per-leg settle. For a symbol that opens *before* 00:00 (a rare
    near-24h close at 23:xx) we still anchor entry to the post-rollover reopen.
    """
    s_date = session_date(broker_now)
    if session is not None:
        open_t = session.open_time
    else:
        open_t = time(params.default_reopen_h, params.default_reopen_m)
    return datetime.combine(s_date, open_t) + timedelta(minutes=settle_min)


def in_entry_window(
    broker_now: datetime,
    session: SymbolSession | None,
    settle_min: int,
    params: RolloverParams,
) -> bool:
    """True once a symbol's reopen+settle has passed, within the trading-day band.

    The actual "is the market truly open" check is left to the quote-freshness
    gate (a closed market yields no fresh quote → the symbol defers), so this is
    only the schedule gate.
    """
    start = reopen_at(broker_now, session, settle_min, params)
    end = start + timedelta(minutes=params.entry_window_min)
    return start <= broker_now < end


def is_triple_night(broker_now: datetime, swap_rollover3days_weekday: int) -> bool:
    """Whether the rollover ``broker_now`` precedes charges ×3 swap for a symbol.

    MT5's ``symbol_info.swap_rollover3days`` gives the weekday (0=Sun … 6=Sat in
    MT5's convention) the triple swap is booked. The charge is booked on that
    weekday's 00:00 rollover, i.e. the rollover at the END of that weekday. The
    exit decided on day ``D`` (in ``[D 23:48, D+1 00:00)``) precedes the rollover
    booked for day ``D``'s close, so the triple applies when ``D``'s weekday
    matches. ``broker_now`` is on day ``D`` during the exit window.
    """
    # Python weekday(): Mon=0..Sun=6. MT5 swap_rollover3days: Sun=0..Sat=6.
    mt5_weekday = (broker_now.weekday() + 1) % 7
    return mt5_weekday == swap_rollover3days_weekday


def is_weekend_closed(broker_now: datetime) -> bool:
    """True during the Saturday gap (no trading) — purely informational."""
    return broker_now.weekday() == _SATURDAY


__all__ = [
    "ROLLOVER_HOUR",
    "RolloverParams",
    "SymbolSession",
    "exit_boundary_date",
    "exit_time",
    "in_entry_window",
    "in_exit_window",
    "is_triple_night",
    "is_weekend_closed",
    "last_rollover",
    "next_rollover",
    "reopen_at",
    "session_date",
]
