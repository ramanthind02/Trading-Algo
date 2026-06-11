"""Execution-hours + quote-freshness gate for the live/sandbox rebalance.

The signal is generated from the Darwinex daily close, but orders execute on a
DIFFERENT broker (e.g. FTMO) whose market hours differ — FTMO index/energy CFDs
close ~23:15 EET while Darwinex closes 23:00 EET, and both have a nightly rollover
break. Submitting a market order while the execution venue is closed (or against a
stale quote) risks a rejected or wildly-off fill. So before submitting a symbol we
require BOTH:

1. the **execution** broker's schedule says the market is open, and
2. we hold a **fresh** quote for it.

``brokers.is_market_open`` is the schedule check; its own docstring notes that a
live tick-freshness probe is the ground truth for ad-hoc holidays / early closes —
so we AND the two. Symbols failing either condition are *deferred* (retried on a
later poll within the session), never force-traded into a closed book.

Pure: no Nautilus, no I/O.
"""
from __future__ import annotations

from datetime import datetime

from data_platform.providers.mt5 import brokers

#: A quote older than this many seconds is treated as stale (venue likely closed
#: or the feed gapped) — defer rather than trade on a dead price.
DEFAULT_MAX_QUOTE_AGE_SECS: float = 300.0


def is_quote_fresh(
    quote_age_secs: float | None, max_age_secs: float = DEFAULT_MAX_QUOTE_AGE_SECS
) -> bool:
    """True when a quote exists and is no older than ``max_age_secs``."""
    return quote_age_secs is not None and 0.0 <= quote_age_secs <= max_age_secs


def tradeable(
    exec_broker: str,
    canonical: str,
    now: datetime,
    *,
    quote_age_secs: float | None,
    max_quote_age_secs: float = DEFAULT_MAX_QUOTE_AGE_SECS,
) -> bool:
    """True when ``canonical`` can be safely submitted on ``exec_broker`` at ``now``.

    Requires the execution broker's market to be open on its schedule AND a fresh
    quote. Either failing → defer the symbol.
    """
    if not brokers.is_market_open(exec_broker, canonical, now):
        return False
    return is_quote_fresh(quote_age_secs, max_quote_age_secs)


def market_open(exec_broker: str, canonical: str, now: datetime) -> bool:
    """Schedule-only open check (no quote requirement) — for pre-window filtering."""
    return brokers.is_market_open(exec_broker, canonical, now)


__all__ = ["DEFAULT_MAX_QUOTE_AGE_SECS", "is_quote_fresh", "market_open", "tradeable"]
