"""Nautilus-aligned identifiers: Venue, Symbol, InstrumentId.

Mirrors nautilus_trader.model.identifiers semantics without the dependency:
  - InstrumentId renders as ``{symbol}.{venue}`` (e.g. ``AAPL.XNAS``).
  - Venues use MIC-style codes (XNAS, XNYS, XCME, ...) so the mapping to
    NautilusTrader's venue identifiers is 1:1 when we migrate.

When we adopt nautilus_trader, these classes map directly onto
``nautilus_trader.model.identifiers.{Venue, Symbol, InstrumentId}`` — the
string form (``from_str`` / ``str``) is identical, so serialized catalog
rows and BarType strings carry over unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Venue:
    """A trading/listing venue, identified by a MIC-style code (e.g. ``XNAS``)."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or "." in self.value or "-" in self.value:
            raise ValueError(f"Invalid venue code: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Symbol:
    """A venue-native symbol (e.g. ``AAPL``, ``ES`` for the continuous root)."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Symbol cannot be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """Unique instrument identifier rendered as ``{symbol}.{venue}``.

    The ``{symbol}.{venue}`` combination must be unique within the system,
    matching NautilusTrader's symbology rule.
    """

    symbol: Symbol
    venue: Venue

    def __str__(self) -> str:
        return f"{self.symbol}.{self.venue}"

    @property
    def value(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, value: str) -> "InstrumentId":
        """Parse ``{symbol}.{venue}``. The venue is the final dotted segment.

        Symbols may themselves contain dots (e.g. ``BRK.A.XNYS``), so we split
        on the *last* dot only.
        """
        if "." not in value:
            raise ValueError(f"InstrumentId must be 'symbol.venue', got {value!r}")
        symbol_part, _, venue_part = value.rpartition(".")
        return cls(Symbol(symbol_part), Venue(venue_part))
