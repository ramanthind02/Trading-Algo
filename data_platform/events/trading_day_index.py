"""Trading-day offsets relative to event D0 dates."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence


class TradingDayIndex:
    """Maps session dates to integer offsets from each event D0."""

    def __init__(self, trading_sessions: Sequence[date]) -> None:
        ordered = tuple(sorted(set(trading_sessions)))
        self._sessions = ordered
        self._position = {d: i for i, d in enumerate(ordered)}

    @property
    def sessions(self) -> tuple[date, ...]:
        return self._sessions

    def offset(self, session: date, d0: date) -> int | None:
        if session not in self._position or d0 not in self._position:
            return None
        return self._position[session] - self._position[d0]

    def active_sessions(
        self,
        d0_dates: Iterable[date],
        entry_offset: int,
        exit_offset: int,
    ) -> frozenset[date]:
        active: set[date] = set()
        for d0 in d0_dates:
            d0_pos = self._position.get(d0)
            if d0_pos is None:
                continue
            start_pos = d0_pos + entry_offset
            end_pos = d0_pos + exit_offset
            if start_pos < 0:
                start_pos = 0
            if end_pos >= len(self._sessions):
                end_pos = len(self._sessions) - 1
            if start_pos > end_pos:
                continue
            active.update(self._sessions[start_pos : end_pos + 1])
        return frozenset(active)


def load_es_trading_sessions(
    start: date | None = None,
    end: date | None = None,
) -> tuple[date, ...]:
    """Load ES daily session dates from repository OHLC parquet."""
    from datetime import datetime

    from lib.core import helpers
    from lib.core.enums import Ticker, TimeFrame

    start_dt = datetime(start.year, start.month, start.day) if start else datetime(1990, 1, 1)
    end_dt = datetime(end.year, end.month, end.day) if end else datetime(2030, 12, 31)
    df = helpers.load_data(Ticker.ES, TimeFrame.D, start=start_dt, end=end_dt)
    return tuple(sorted(set(df["datetime"].dt.date)))
