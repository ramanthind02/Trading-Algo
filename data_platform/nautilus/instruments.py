"""Map ``data_platform.core`` instruments onto NautilusTrader instruments.

Each :class:`data_platform.core.instruments.Instrument` row carries the
execution-correct precision / tick increment / multiplier / venue, so the
mapping is a direct field copy keyed by ``instrument_class`` (and
``asset_class`` for the ``SPOT`` discriminator):

  * ``FUTURE``                 -> :class:`FuturesContract`
  * ``SPOT`` + ``EQUITY``      -> :class:`Equity`
  * ``SPOT`` + ``FX``          -> :class:`CurrencyPair`
  * ``CFD``                    -> :class:`Cfd`
  * ``WARRANT``                -> :class:`Equity` (US warrants trade like shares)

Continuous-future roots have no real expiry, so we synthesise a far-future
``expiration_ns`` (year 2099) and take ``activation_ns`` from the row's
``activation`` date. This keeps the instrument permanently "active" for the
research lane while remaining a valid Nautilus ``FuturesContract``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AssetClass as NTAssetClass
from nautilus_trader.model.identifiers import InstrumentId as NTInstrumentId
from nautilus_trader.model.identifiers import Symbol as NTSymbol
from nautilus_trader.model.instruments import (
    Cfd,
    CurrencyPair,
    Equity,
    FuturesContract,
)
from nautilus_trader.model.instruments import Instrument as NTInstrument
from nautilus_trader.model.objects import Currency, Price, Quantity

from data_platform.core.catalog import load_catalog
from data_platform.core.enums import AssetClass, InstrumentClass
from data_platform.core.instruments import Instrument

# Continuous-future roots are perpetual; Nautilus still wants a concrete expiry.
# Use a far-future sentinel so the contract never "expires" in the research lane.
_FAR_FUTURE = datetime(2099, 12, 31, tzinfo=timezone.utc)
# Epoch sentinel for a missing activation date (instrument always "active").
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_ASSET_CLASS_MAP: dict[AssetClass, NTAssetClass] = {
    AssetClass.FX: NTAssetClass.FX,
    AssetClass.EQUITY: NTAssetClass.EQUITY,
    AssetClass.COMMODITY: NTAssetClass.COMMODITY,
    AssetClass.DEBT: NTAssetClass.DEBT,
    AssetClass.INDEX: NTAssetClass.INDEX,
    AssetClass.CRYPTOCURRENCY: NTAssetClass.CRYPTOCURRENCY,
    AssetClass.ALTERNATIVE: NTAssetClass.ALTERNATIVE,
}


@dataclass(frozen=True)
class SkippedInstrument:
    """A catalog row that could not (or should not) be mapped, with a reason."""

    instrument_id: str
    instrument_class: str
    asset_class: str
    reason: str


def _to_ns(d: date | None, *, default: datetime) -> int:
    dt = (
        datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if d is not None
        else default
    )
    # Nautilus timestamps are uint64 nanos from the UNIX epoch; pre-1970 dates
    # (some Norgate continuous-root history starts) clamp to 0 (always-active).
    return max(0, int(dt.timestamp() * 1_000_000_000))


def _currency(code: str) -> Currency:
    try:
        return Currency.from_str(code)
    except Exception:
        return USD


def _price_increment(inst: Instrument) -> Price:
    return Price(inst.price_increment, inst.price_precision)


def _size_increment(inst: Instrument) -> Quantity:
    return Quantity(inst.size_increment, inst.size_precision)


def _multiplier(inst: Instrument) -> Quantity:
    # Catalog multipliers are whole point-values; keep as an integer Quantity.
    return Quantity.from_int(int(round(inst.multiplier)))


def _lot_size(inst: Instrument) -> Quantity:
    return Quantity.from_int(int(round(inst.lot_size)))


def _instrument_id(inst: Instrument) -> NTInstrumentId:
    return NTInstrumentId.from_str(str(inst.id))


def _to_futures(inst: Instrument) -> FuturesContract:
    return FuturesContract(
        instrument_id=_instrument_id(inst),
        raw_symbol=NTSymbol(inst.raw_symbol),
        asset_class=_ASSET_CLASS_MAP[inst.asset_class],
        currency=_currency(inst.quote_currency),
        price_precision=inst.price_precision,
        price_increment=_price_increment(inst),
        multiplier=_multiplier(inst),
        lot_size=_lot_size(inst),
        underlying=inst.underlying or inst.symbol,
        activation_ns=_to_ns(inst.activation, default=_EPOCH),
        expiration_ns=_to_ns(inst.expiration, default=_FAR_FUTURE),
        ts_event=0,
        ts_init=0,
        margin_init=Decimal(str(inst.margin_init)) if inst.margin_init is not None else None,
        exchange=inst.venue,
        info=dict(inst.info),
    )


def _to_equity(inst: Instrument) -> Equity:
    return Equity(
        instrument_id=_instrument_id(inst),
        raw_symbol=NTSymbol(inst.raw_symbol),
        currency=_currency(inst.quote_currency),
        price_precision=inst.price_precision,
        price_increment=_price_increment(inst),
        lot_size=_lot_size(inst),
        ts_event=0,
        ts_init=0,
        info=dict(inst.info),
    )


def _fx_base_quote(inst: Instrument) -> tuple[str, str]:
    """Derive (base, quote) ISO currencies for an FX SPOT row.

    MT5 FX symbols are 6-char ``BASEQUOTE`` (e.g. ``EURUSD``); base = first 3.
    Quote currency is authoritative from the row.
    """
    sym = inst.symbol
    base = sym[:3] if len(sym) >= 6 else sym
    return base, inst.quote_currency


def _to_currency_pair(inst: Instrument) -> CurrencyPair:
    base, quote = _fx_base_quote(inst)
    return CurrencyPair(
        instrument_id=_instrument_id(inst),
        raw_symbol=NTSymbol(inst.raw_symbol),
        base_currency=_currency(base),
        quote_currency=_currency(quote),
        price_precision=inst.price_precision,
        size_precision=inst.size_precision,
        price_increment=_price_increment(inst),
        size_increment=_size_increment(inst),
        ts_event=0,
        ts_init=0,
        info=dict(inst.info),
    )


def _to_cfd(inst: Instrument) -> Cfd:
    return Cfd(
        instrument_id=_instrument_id(inst),
        raw_symbol=NTSymbol(inst.raw_symbol),
        asset_class=_ASSET_CLASS_MAP[inst.asset_class],
        quote_currency=_currency(inst.quote_currency),
        price_precision=inst.price_precision,
        size_precision=inst.size_precision,
        price_increment=_price_increment(inst),
        size_increment=_size_increment(inst),
        ts_event=0,
        ts_init=0,
        margin_init=Decimal(str(inst.margin_init)) if inst.margin_init is not None else None,
        margin_maint=Decimal(str(inst.margin_maint)) if inst.margin_maint is not None else None,
        info=dict(inst.info),
    )


def to_nautilus_instrument(inst: Instrument) -> NTInstrument:
    """Map one ``data_platform.core`` instrument to its Nautilus counterpart.

    Raises ``ValueError`` for an unhandled ``(instrument_class, asset_class)``
    combination — callers that want soft-skip behaviour should catch it (see
    :func:`build_all_nautilus_instruments`).
    """
    cls = inst.instrument_class
    if cls is InstrumentClass.FUTURE:
        return _to_futures(inst)
    if cls is InstrumentClass.WARRANT:
        # US warrants are quoted/settled like shares; Equity is the closest map.
        return _to_equity(inst)
    if cls is InstrumentClass.CFD:
        return _to_cfd(inst)
    if cls is InstrumentClass.SPOT:
        if inst.asset_class is AssetClass.EQUITY:
            return _to_equity(inst)
        if inst.asset_class is AssetClass.FX:
            return _to_currency_pair(inst)
        raise ValueError(
            f"SPOT row {inst.id} has unmapped asset_class {inst.asset_class.value}"
        )
    raise ValueError(
        f"Instrument {inst.id} has unmapped instrument_class {cls.value}"
    )


def build_all_nautilus_instruments() -> tuple[list[NTInstrument], list[SkippedInstrument]]:
    """Map every on-disk catalog row to a Nautilus instrument.

    Returns ``(mapped, skipped)`` — *skipped* collects rows that raised during
    mapping, each with a human-readable reason.
    """
    catalog = load_catalog()
    mapped: list[NTInstrument] = []
    skipped: list[SkippedInstrument] = []
    for inst in catalog.all():
        try:
            mapped.append(to_nautilus_instrument(inst))
        except Exception as exc:  # noqa: BLE001 — record & continue
            skipped.append(
                SkippedInstrument(
                    instrument_id=str(inst.id),
                    instrument_class=inst.instrument_class.value,
                    asset_class=inst.asset_class.value,
                    reason=str(exc),
                )
            )
    return mapped, skipped


def write_instruments_to_catalog(catalog) -> list[NTInstrument]:
    """Build all Nautilus instruments and write them to *catalog*.

    Returns the list of instruments written (skips are dropped silently here;
    use :func:`build_all_nautilus_instruments` to inspect skip reasons).
    """
    mapped, _skipped = build_all_nautilus_instruments()
    if mapped:
        catalog.write_data(mapped)
    return mapped
