"""Nautilus-aligned instrument specifications.

A frozen ``Instrument`` dataclass hierarchy mirroring the fields NautilusTrader
exposes on its instrument types (Equity, FuturesContract, Cfd, CurrencyPair,
IndexInstrument). We keep one flat dataclass with an ``instrument_class``
discriminator rather than a deep class tree — it serializes cleanly to the
catalog parquet and maps onto the matching nautilus type at migration time.

Precision/increment are first-class (NautilusTrader enforces them strictly):
  - price_precision / price_increment  (tick size)
  - size_precision  / size_increment
  - multiplier (point value: $/point), lot_size, margins, fees

Optional fields are None when not applicable (e.g. equities have no expiry).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .enums import AssetClass, InstrumentClass, PriceAdjustment
from .identifiers import InstrumentId


@dataclass(frozen=True, slots=True)
class Instrument:
    """Specification for a tradable (or reference) instrument.

    Field names match NautilusTrader's instrument types so a row maps onto the
    matching nautilus instrument at migration. ``raw_symbol`` is the
    provider-native symbol (e.g. Norgate ``&ES_CCB`` or ``AAPL``); ``id`` is
    the canonical ``symbol.venue`` identifier.
    """

    id: InstrumentId
    raw_symbol: str
    asset_class: AssetClass
    instrument_class: InstrumentClass

    # Precision & increments (NautilusTrader enforces these strictly)
    price_precision: int
    price_increment: float          # tick size
    size_precision: int = 0
    size_increment: float = 1.0

    # Economics
    quote_currency: str = "USD"
    multiplier: float = 1.0         # point value ($/point); 1.0 for equities
    lot_size: float = 1.0
    margin_init: float | None = None     # exchange initial margin snapshot
    margin_maint: float | None = None
    maker_fee: float = 0.0
    taker_fee: float = 0.0

    # Lifecycle (futures/options have these; equities/cfd do not)
    underlying: str | None = None
    activation: date | None = None       # first quoted date
    expiration: date | None = None       # last quoted date (None = active/perpetual)

    # Provenance / metadata
    venue_name: str | None = None        # human-readable exchange name
    description: str | None = None
    data_source: str | None = None       # "norgate" | "ib" | "mt5"
    # Adjustment conventions of the *stored* series for this instrument, by
    # series tag (e.g. {"D": "BACK_ADJUSTED", "D_unadj": "NONE"}).
    price_adjustments: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)  # raw provider metadata

    @property
    def symbol(self) -> str:
        return str(self.id.symbol)

    @property
    def venue(self) -> str:
        return str(self.id.venue)

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick = price_increment x multiplier."""
        return self.price_increment * self.multiplier
