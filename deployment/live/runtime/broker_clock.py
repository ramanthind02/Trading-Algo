"""Broker-aware clock for the live vault runtime.

Two clocks matter, and they are NOT the same:

* **Signal clock — US Eastern.** The vault forecast is a *daily-close* signal
  aligned to the US cash close (16:00 ET). The trading "session date" and the
  once-per-day rebalance decision are therefore reckoned in ``America/New_York``.
* **Broker clock — server EET/EEST.** MT5 brokers (Darwinex/FTMO/FundedNext)
  run a server clock in EET (UTC+2 winter / UTC+3 summer). Market-session checks
  and the nightly financing-rollover dead-zone (stored 00:00–01:00, when the
  book has zero ticks) are reckoned in the broker tz.

Both are derived from real wall-clock UTC via ``zoneinfo`` (correct DST), so we
never hard-code the "ET = stored − 7h" approximation noted in
``docs/library/Data/mt5_timezones.md`` — that offset only holds when EU and US
DST agree; ``zoneinfo`` handles the few mismatched weeks each spring/autumn.

This module is pure (no I/O, no terminal); it reuses ``brokers.broker_tzinfo``
for the broker server tz.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from data_platform.providers.mt5 import brokers

#: US cash-equity timezone — the canonical clock for the daily-close signal.
US_EASTERN: ZoneInfo = ZoneInfo("America/New_York")

#: Final trading weekday in the US (Mon=0 .. Fri=4). Sat/Sun have no new close.
_LAST_TRADING_WEEKDAY: int = 4


def now_utc() -> datetime:
    """Timezone-aware current instant in UTC."""
    return datetime.now(timezone.utc)


def _as_utc(moment: datetime) -> datetime:
    """Coerce to aware UTC (naive datetimes are assumed to already be UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def to_eastern(moment: datetime) -> datetime:
    """Convert an instant to ``America/New_York`` (the signal clock)."""
    return _as_utc(moment).astimezone(US_EASTERN)


def to_broker(moment: datetime, broker: str) -> datetime:
    """Convert an instant to the broker's EET/EEST server clock."""
    return _as_utc(moment).astimezone(brokers.broker_tzinfo(broker))


def in_rollover_deadzone(moment: datetime, broker: str) -> bool:
    """True inside the nightly financing-rollover dead-zone (broker 00:00–01:00).

    The book is effectively empty here (zero ticks); executing into it risks
    a no-price / wide-spike fill, so the strategy holds off rebalancing.
    """
    return to_broker(moment, broker).hour == 0


def parse_hhmm(value: str) -> time:
    """Parse a ``"HH:MM"`` string into a :class:`datetime.time`."""
    hh, mm = value.split(":")
    return time(hour=int(hh), minute=int(mm))


@dataclass(frozen=True)
class DecisionClock:
    """Decides *when* the once-per-day rebalance fires, in US-Eastern terms.

    A decision for trading-session date ``D`` fires the first time the runtime
    observes a moment on weekday ``D`` at or after ``decision_time_et`` for which
    no decision has yet been recorded. Weekends never fire (no new daily close).

    Parameters
    ----------
    decision_time_et:
        Local Eastern time of day at/after which the daily decision is allowed
        to run (e.g. ``16:05`` — just after the 16:00 ET cash close so the final
        daily bar has formed). Default ``16:05``.
    """

    decision_time_et: time = time(hour=16, minute=5)

    @classmethod
    def from_hhmm(cls, hhmm: str) -> "DecisionClock":
        return cls(decision_time_et=parse_hhmm(hhmm))

    def session_date(self, moment: datetime) -> date:
        """The US-Eastern calendar date of ``moment`` (its trading session)."""
        return to_eastern(moment).date()

    def is_trading_day(self, moment: datetime) -> bool:
        """True when ``moment`` falls on a US trading weekday (Mon–Fri, ET)."""
        return to_eastern(moment).weekday() <= _LAST_TRADING_WEEKDAY

    def is_past_decision(self, moment: datetime) -> bool:
        """True on a US weekday at/after ``decision_time_et`` — the time/weekday gate.

        Unlike :meth:`is_due` this carries NO once-per-day latch: the caller owns
        that (e.g. a persisted decision-state marker that survives restarts), so a
        crash + restart can resume an unfinished session rather than being blocked
        by an in-memory flag.
        """
        et = to_eastern(moment)
        return et.weekday() <= _LAST_TRADING_WEEKDAY and et.time() >= self.decision_time_et

    def is_due(self, moment: datetime, last_session_date: date | None) -> bool:
        """Should the daily decision run now?

        Parameters
        ----------
        moment:
            Current instant (any tz; naive treated as UTC).
        last_session_date:
            Eastern date of the most recent decision already taken, or ``None``
            if none has run yet this process. Prevents re-firing within a day.
        """
        et = to_eastern(moment)
        if et.weekday() > _LAST_TRADING_WEEKDAY:
            return False
        if et.time() < self.decision_time_et:
            return False
        return last_session_date is None or et.date() > last_session_date


__all__ = [
    "US_EASTERN",
    "DecisionClock",
    "now_utc",
    "to_eastern",
    "to_broker",
    "in_rollover_deadzone",
    "parse_hhmm",
]
