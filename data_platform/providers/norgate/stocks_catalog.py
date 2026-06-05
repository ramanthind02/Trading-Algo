"""Build data_core.Instrument rows for Norgate US stocks and add them to the catalog.

For each scraped stock this constructs one ``data_core.Instrument`` and upserts it
into the existing ``data/instruments/catalog.parquet`` (loaded, extended via
``add_many``, saved) — it NEVER overwrites the futures/TLT rows already present.

InstrumentId convention for stocks:
    {RAW_SYMBOL}.{VENUE_MIC}      e.g. AAPL.XNAS, SPY.ARCX, AGM.A.XNYS
The symbol is the *raw Norgate symbol* (matching the stock_data store key), so the
catalog resolves back to the partitioned bar store with no data move. Symbols may
contain dots (``AGM.A``); ``InstrumentId.from_str`` splits on the last dot, so the
venue (a MIC, never dotted) is recovered correctly.

Classification:
  - asset_class      = EQUITY for all (equity, ETP/ETF, preferred, warrant, etc.)
  - instrument_class = SPOT for common shares / ETPs (default);
                       WARRANT for Norgate subtype1 "Derivative" warrants.
  - price_precision  = 2, price_increment = 0.01, multiplier = 1.0 (US equities)

Venue: resolved via ``data_platform.core.venue_mic_for_exchange`` (the single
exchange->MIC source of truth). Norgate's "OTC" bucket and any unmapped exchange
fall back to the private "OOTC" MIC, so seeding never crashes.

Run directly to (re)seed the stock portion of the catalog for a sample/universe:
    python -m data_platform.providers.norgate.stocks_catalog --universe active --limit 20
    python -m data_platform.providers.norgate.stocks_catalog --watchlist "S&P 500" --limit 10
    python -m data_platform.providers.norgate.stocks_catalog --scraped-only   (only on-disk symbols)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

import norgatedata

from data_platform.core import (
    AssetClass,
    Instrument,
    InstrumentCatalog,
    InstrumentClass,
    InstrumentId,
    Symbol,
    Venue,
    venue_mic_for_exchange,
)

from .fetch_continuous import ensure_norgate_running
from .stocks import (
    Universe,
    already_fetched,
    list_symbols,
    list_watchlist_symbols,
    read_raw_symbol,
    stock_data_root,
)

# Norgate subtype1 -> data_core InstrumentClass (default SPOT).
_SUBTYPE1_TO_INSTRUMENT_CLASS: dict[str, InstrumentClass] = {
    "Derivative": InstrumentClass.WARRANT,
}


@dataclass(frozen=True)
class _StockMeta:
    symbol: str
    security_name: str | None
    exchange_name: str | None
    subtype1: str | None
    subtype2: str | None
    first_quoted_date: date | None
    last_quoted_date: date | None
    assetid: int | None


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


def _fetch_meta(symbol: str) -> _StockMeta:
    return _StockMeta(
        symbol=symbol,
        security_name=_safe(norgatedata.security_name, symbol),
        exchange_name=_safe(norgatedata.exchange_name, symbol),
        subtype1=_safe(norgatedata.subtype1, symbol),
        subtype2=_safe(norgatedata.subtype2, symbol),
        first_quoted_date=_parse_date(
            _safe(norgatedata.first_quoted_date, symbol, datetimeformat="iso")
        ),
        last_quoted_date=_parse_date(
            _safe(norgatedata.last_quoted_date, symbol, datetimeformat="iso")
        ),
        assetid=_safe(norgatedata.assetid, symbol),
    )


def _venue_mic(exchange_name: str | None) -> str:
    return venue_mic_for_exchange(exchange_name)


def _instrument_class(subtype1: str | None) -> InstrumentClass:
    return _SUBTYPE1_TO_INSTRUMENT_CLASS.get(subtype1 or "", InstrumentClass.SPOT)


def _is_etf(meta: _StockMeta) -> bool:
    return (meta.subtype1 or "") == "Exchange Traded Product"


def build_stock_instrument(symbol: str) -> Instrument:
    meta = _fetch_meta(symbol)
    mic = _venue_mic(meta.exchange_name)
    return Instrument(
        id=InstrumentId(Symbol(symbol), Venue(mic)),
        raw_symbol=symbol,
        asset_class=AssetClass.EQUITY,
        instrument_class=_instrument_class(meta.subtype1),
        price_precision=2,
        price_increment=0.01,
        size_precision=0,
        size_increment=1.0,
        quote_currency="USD",
        multiplier=1.0,
        lot_size=1.0,
        activation=meta.first_quoted_date,
        expiration=meta.last_quoted_date,  # None for active, set for delisted
        venue_name=meta.exchange_name,
        description=meta.security_name,
        data_source="norgate",
        price_adjustments={
            "D_TR": "TOTAL_RETURN",
            "W_TR": "TOTAL_RETURN",
            "M_TR": "TOTAL_RETURN",
            "D_CAP": "CAPITAL",
            "W_CAP": "CAPITAL",
            "M_CAP": "CAPITAL",
            "D_UNADJ": "NONE",
            "W_UNADJ": "NONE",
            "M_UNADJ": "NONE",
        },
        info={
            "subtype1": meta.subtype1,
            "subtype2": meta.subtype2,
            "asset_kind": "ETF" if _is_etf(meta) else "STOCK",
            "assetid": meta.assetid,
            "first_quoted_date": meta.first_quoted_date.isoformat()
            if meta.first_quoted_date
            else None,
            "last_quoted_date": meta.last_quoted_date.isoformat()
            if meta.last_quoted_date
            else None,
            "delisted": meta.last_quoted_date is not None,
        },
    )


def build_stock_instruments(symbols: list[str]) -> list[Instrument]:
    out: list[Instrument] = []
    for symbol in symbols:
        try:
            out.append(build_stock_instrument(symbol))
        except Exception as exc:
            print(f"  ERR building instrument for {symbol}: {exc}")
    return out


def _scraped_symbols() -> list[str]:
    """Raw symbols that already have a primary (D_TR) file under data/stock_data/.

    The raw Norgate symbol is recovered losslessly from each parquet's file-level
    metadata (safe filenames like ``AGM_A`` can't be reversed on their own).
    """
    root = stock_data_root()
    if not root.exists():
        return []
    out: list[str] = []
    for price_file in root.glob("*/*/D_TR_*.parquet"):
        raw = read_raw_symbol(price_file)
        out.append(raw if raw is not None else price_file.parent.name)
    return sorted(set(out))


def seed_stock_catalog(symbols: list[str]) -> tuple[InstrumentCatalog, int]:
    """Load the catalog, add stock instruments for *symbols*, save. Returns (catalog, added)."""
    catalog = InstrumentCatalog.load()
    before = len(catalog)
    catalog.add_many(build_stock_instruments(symbols))
    catalog.save()
    return catalog, len(catalog) - before


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.scraped_only:
        return _scraped_symbols()
    if args.watchlist:
        symbols = list_watchlist_symbols(args.watchlist)
    else:
        symbols = list_symbols(Universe(args.universe))
    if args.on_disk_only:
        symbols = [s for s in symbols if already_fetched(s)]
    if args.limit is not None:
        symbols = symbols[: args.limit]
    return symbols


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add Norgate stock Instrument rows to the existing catalog."
    )
    p.add_argument("--universe", choices=[u.value for u in Universe],
                   default=Universe.ACTIVE.value)
    p.add_argument("--watchlist", default=None, help="Only this watchlist's members.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--on-disk-only", action="store_true",
                   help="Only symbols whose price files already exist.")
    p.add_argument("--scraped-only", action="store_true",
                   help="Derive symbol list from data/stock_data/ folder names.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not args.scraped_only:
        ensure_norgate_running()
    symbols = _resolve_symbols(args)
    print(f"=== Seeding catalog with {len(symbols)} stock instruments ===")
    catalog, added = seed_stock_catalog(symbols)
    equities = catalog.filter(asset_class=AssetClass.EQUITY, data_source="norgate")
    print(f"Catalog now {len(catalog)} total ({added} added this run); "
          f"{len(equities)} EQUITY rows.")


if __name__ == "__main__":
    main()
