"""Broker-wall-clock session constants and helpers for MT5 intraday data.

Promotion source: research/experiments/lafo_kama_mr/engine.py (_bsec, RTH/H24
window constants).
Authority: docs/library/Data/mt5_timezones.md

Every timestamp in data/mt5_data/** is BROKER EET/EEST, mislabelled UTC.  Strip
the false UTC tag with ``.dt.tz_localize(None)`` to obtain broker wall-clock, then
keep it as-is for signal logic — bar boundaries are consistent in broker time and
converting to UTC/ET is only needed for human-facing reporting.

ET ↔ broker offset: ET + 7h = broker (constant year-round; both zones observe DST
within ~2 weeks of each other so the +7 is stable for intraday signal work).
The daily financing rollover sits at broker 00:00 (= 17:00 ET); anchor session
windows to broker midnight, not to a hard-coded ET hour.
"""
from __future__ import annotations

from enum import Enum


class Session(str, Enum):
    """Intraday session identifier."""

    RTH = "rth"  # US regular trading hours: 09:30–16:00 ET  =  16:30–23:00 broker
    H24 = "h24"  # Full CFD session: 01:05–23:45 broker (dead-zone excluded, T-15 flat)


def bsec(et_h: int, et_m: int) -> int:
    """Convert ET hh:mm to broker seconds-of-day.  ET + 7h = broker (constant)."""
    return (et_h + 7) * 3600 + et_m * 60


# ---------------------------------------------------------------------------
# Session boundaries in broker seconds-of-day
# ---------------------------------------------------------------------------
RTH_OPEN_S: int = bsec(9, 30)     # 16:30 broker  =  09:30 ET cash open
RTH_CLOSE_S: int = bsec(16, 0)    # 23:00 broker  =  16:00 ET cash close

# H24 uses broker-native anchors (not ET-derived) so the dead-zone and rollover
# boundaries stay correct even during the ~2-week US/EU DST mismatch windows.
H24_OPEN_S: int = 1 * 3600 + 5 * 60      # 01:05 broker — first bar after 00:00–01:00 dead zone
H24_CLOSE_S: int = 23 * 3600 + 45 * 60   # 23:45 broker — T-15 pre-rollover (swap-avoidance flat)


def session_window(session: Session) -> tuple[int, int]:
    """Return ``(open_s, close_s)`` broker-seconds-of-day for *session*."""
    match session:
        case Session.RTH:
            return RTH_OPEN_S, RTH_CLOSE_S
        case Session.H24:
            return H24_OPEN_S, H24_CLOSE_S
