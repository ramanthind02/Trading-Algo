"""Demand-driven quote-tick subscription controller.

The vault signal is daily and cache-sourced, so we do NOT stream ticks
continuously for the whole universe. Instead, when a rebalance fires we open a
short *execution window*: subscribe to live quotes for exactly the instruments
about to trade, wait until each has a fresh quote (for sizing + to move the
sandbox book so market orders fill), then unsubscribe once the window closes.

This class is the pure bookkeeping for that window — subscribe/unsubscribe are
injected callables so it is fully testable without a Nautilus node. It is
idempotent: re-opening a window never double-subscribes an already-live symbol.

The generic ``K`` key is whatever the caller uses to identify an instrument
(a Nautilus ``InstrumentId``, its string form, etc.).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Hashable
from typing import Generic, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
Q = TypeVar("Q")


class DemandTickController(Generic[K, Q]):
    """Tracks the active execution window: who is subscribed and who has quoted."""

    def __init__(
        self,
        subscribe: Callable[[K], None],
        unsubscribe: Callable[[K], None],
    ) -> None:
        self._subscribe = subscribe
        self._unsubscribe = unsubscribe
        self._subscribed: set[K] = set()
        self._awaiting: set[K] = set()
        self._fresh: dict[K, Q] = {}

    def open_window(self, instrument_ids: Iterable[K]) -> None:
        """Begin a window: subscribe new instruments, await a fresh quote for all.

        Idempotent — instruments already subscribed are not re-subscribed, but
        they are reset to "awaiting" so a stale quote from a prior window does
        not count as fresh for this one.
        """
        for iid in instrument_ids:
            if iid not in self._subscribed:
                self._subscribe(iid)
                self._subscribed.add(iid)
            self._awaiting.add(iid)
            self._fresh.pop(iid, None)

    def record(self, instrument_id: K, quote: Q) -> None:
        """Record a fresh quote for an instrument in the current window."""
        if instrument_id in self._subscribed:
            self._fresh[instrument_id] = quote

    def ready(self) -> bool:
        """True when every awaited instrument has a fresh quote."""
        return bool(self._awaiting) and self._awaiting.issubset(self._fresh)

    def missing(self) -> set[K]:
        """Awaited instruments still lacking a fresh quote."""
        return {iid for iid in self._awaiting if iid not in self._fresh}

    def latest(self, instrument_id: K) -> Optional[Q]:
        """Most recent quote recorded for an instrument this window, if any."""
        return self._fresh.get(instrument_id)

    def quotes(self) -> dict[K, Q]:
        """All fresh quotes recorded in the current window."""
        return dict(self._fresh)

    def active(self) -> bool:
        """True while any instrument is subscribed (a window is open)."""
        return bool(self._subscribed)

    def close_window(self) -> None:
        """End the window: unsubscribe everything and clear state."""
        for iid in list(self._subscribed):
            self._unsubscribe(iid)
        self._subscribed.clear()
        self._awaiting.clear()
        self._fresh.clear()


__all__ = ["DemandTickController"]
