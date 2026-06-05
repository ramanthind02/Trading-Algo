"""Build data_core Instrument rows for the existing futures + TLT, seed the catalog.

Reads the persisted Norgate contract specs (data/norgate/archive/contract_specs.json)
and the asset-class mapping, and emits one ``data_core.Instrument`` per repo ticker.

InstrumentId convention for futures continuous roots:
    {REPO_TICKER}.{VENUE_MIC}      e.g. ES.XCME, CL.XNYM, ZN -> TY.XCBT
The symbol is the repo ticker (matching the ohlc_data/{TICKER}/ folder), so the
catalog resolves back to the existing bar store with no data move.

Run directly to (re)seed the futures+TLT portion of the catalog:
    python -m data_platform.providers.norgate.to_catalog
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from data_platform.core import (
    AssetClass,
    Instrument,
    InstrumentCatalog,
    InstrumentClass,
    InstrumentId,
    NORGATE_EXCHANGE_TO_MIC,
    Symbol,
    Venue,
)

from ._constants import TICKER_TO_CONTRACT_PREFIX
from ._paths import norgate_root

# Norgate subtype1 -> data_core AssetClass
_SUBTYPE_TO_ASSET_CLASS: dict[str, AssetClass] = {
    "Stock Index": AssetClass.INDEX,
    "Currency": AssetClass.FX,
    "Energy": AssetClass.COMMODITY,
    "Metal": AssetClass.COMMODITY,
    "Agriculture": AssetClass.COMMODITY,
    "Interest Rate": AssetClass.DEBT,
}

# TLT is an ETF held in the futures-style store (bond proxy); classify as equity ETF.
_TLT_VENUE_MIC = "XNAS"


def _specs_path() -> Path:
    return norgate_root() / "archive" / "contract_specs.json"


def _venue_mic(exchange_name: str) -> str:
    mic = NORGATE_EXCHANGE_TO_MIC.get(exchange_name)
    if mic is None:
        raise ValueError(f"No MIC mapping for Norgate exchange {exchange_name!r}")
    return mic


def _price_precision(tick_size: float) -> int:
    """Decimal places implied by the tick size (e.g. 0.25 -> 2, 5e-05 -> 5)."""
    s = f"{tick_size:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


# IB CONTFUT exchange overrides + ETF proxy + MT5 CFD-equivalent, sourced from the
# live_forecast configs (the authoritative trading mapping). IB uses COMEX for
# metals where Norgate reports NYMEX; configs capture that. Tickers absent here
# fall back to {ib root = contract prefix, ib exchange = Norgate exchange}.
_IB_EXCHANGE_OVERRIDE: dict[str, str] = {"GC": "COMEX", "SI": "COMEX"}
_ETF_PROXY: dict[str, str] = {
    "ES": "SPY", "NQ": "QQQ", "YM": "DIA", "RTY": "IWM", "GC": "GLD", "TLT": "TLT",
}
_MT5_SYMBOL: dict[str, str] = {
    # Darwinex terminal uses non-standard index names — confirmed 2026-06-04
    "ES": "SP500",       # was US500.cash — Darwinex uses SP500
    "NQ": "NDX",         # was US100.cash — Darwinex uses NDX (not NAS100/US100)
    "GC": "XAUUSD",
    "SI": "XAGUSD",
    "CL": "XTIUSD",
}


def _source_symbols(ticker: str, norgate_symbol: str, norgate_exchange: str) -> dict[str, str | None]:
    """Map a futures InstrumentId to its native symbol at each source."""
    ib_root = TICKER_TO_CONTRACT_PREFIX.get(ticker, ticker)
    return {
        "norgate_adj": norgate_symbol,                 # &ES_CCB
        "norgate_unadj": norgate_symbol.replace("_CCB", ""),  # &ES
        "ib_contfut": ib_root,                          # ES (CONTFUT root)
        "ib_exchange": _IB_EXCHANGE_OVERRIDE.get(ticker, norgate_exchange),
        "ib_sec_type": "CONTFUT",
        "etf_proxy": _ETF_PROXY.get(ticker),            # SPY / None
        "mt5": _MT5_SYMBOL.get(ticker),                 # US500.cash / None
    }


def build_futures_instruments() -> list[Instrument]:
    specs = json.loads(_specs_path().read_text())
    instruments: list[Instrument] = []
    for row in specs:
        ticker = row["ticker"]
        mic = _venue_mic(row["exchange_name"])
        tick = float(row["tick_size"])
        asset_class = _SUBTYPE_TO_ASSET_CLASS.get(row.get("subtype1", ""), AssetClass.COMMODITY)
        inst = Instrument(
            id=InstrumentId(Symbol(ticker), Venue(mic)),
            raw_symbol=row["norgate_symbol"],
            asset_class=asset_class,
            instrument_class=InstrumentClass.FUTURE,
            price_precision=_price_precision(tick),
            price_increment=tick,
            size_precision=0,
            size_increment=1.0,
            quote_currency=row.get("currency", "USD"),
            multiplier=float(row["point_value"]),
            lot_size=1.0,
            margin_init=row.get("margin"),
            margin_maint=None,
            underlying=ticker,
            activation=_parse_date(row.get("history_start")),
            expiration=None,  # continuous root has no expiry
            venue_name=row["exchange_name"],
            description=row.get("futures_market_name"),
            data_source="norgate",
            price_adjustments={"D": "BACK_ADJUSTED", "W": "BACK_ADJUSTED",
                                "M": "BACK_ADJUSTED", "D_unadj": "NONE"},
            info={"norgate_symbol": row["norgate_symbol"],
                  "subtype1": row.get("subtype1"),
                  "tick_value": row.get("tick_value"),
                  "source_symbols": _source_symbols(
                      ticker, row["norgate_symbol"], row["exchange_name"])},
        )
        instruments.append(inst)
    return instruments


def build_tlt_instrument() -> Instrument:
    """TLT is stored in ohlc_data as an ETF (bond proxy), not a futures root."""
    return Instrument(
        id=InstrumentId(Symbol("TLT"), Venue(_TLT_VENUE_MIC)),
        raw_symbol="TLT",
        asset_class=AssetClass.EQUITY,
        instrument_class=InstrumentClass.SPOT,
        price_precision=2,
        price_increment=0.01,
        size_precision=0,
        size_increment=1.0,
        quote_currency="USD",
        multiplier=1.0,
        lot_size=1.0,
        venue_name="Nasdaq",
        description="iShares 20+ Year Treasury Bond ETF",
        data_source="norgate",
        price_adjustments={"D": "TOTAL_RETURN", "W": "TOTAL_RETURN", "M": "TOTAL_RETURN"},
        info={"asset_kind": "ETF", "bond_proxy": True,
              "source_symbols": {"norgate_adj": "TLT", "ib_contfut": "TLT",
                                  "ib_exchange": "SMART", "ib_sec_type": "STK",
                                  "etf_proxy": "TLT", "mt5": None}},
    )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


def seed_catalog() -> InstrumentCatalog:
    """Load (or create) the catalog and upsert the futures + TLT instruments."""
    catalog = InstrumentCatalog.load()
    catalog.add_many(build_futures_instruments())
    catalog.add(build_tlt_instrument())
    catalog.save()
    return catalog


def main() -> None:
    catalog = seed_catalog()
    futures = catalog.filter(instrument_class=InstrumentClass.FUTURE)
    print(f"Catalog seeded: {len(catalog)} total instruments")
    print(f"  futures continuous roots: {len(futures)}")
    for inst in sorted(futures, key=lambda i: str(i.id)):
        print(f"    {str(inst.id):<12} mult=${inst.multiplier:<10} tick={inst.price_increment:<10} {inst.description}")


if __name__ == "__main__":
    main()
