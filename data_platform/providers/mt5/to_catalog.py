"""Seed the instrument catalog with Darwinex MT5 CFD instruments.

These are CFD instruments available on the Darwinex Live account that have
no direct futures equivalent in the repo (or where the CFD is the primary
trading vehicle on this broker). Instruments that already exist as futures
(ES→SP500, NQ→NDX, GC→XAUUSD etc.) are cross-referenced via the
`source_symbols["mt5"]` field on the futures instrument — those are handled
by data_platform/providers/norgate/to_catalog.py.

This module seeds instruments where Darwinex is the *primary* data source:
  - Index CFDs with no repo futures equivalent (NDX as standalone CFD entry)
  - FX pairs (EURUSD, USDJPY, etc.) for which we have MT5 tick history
  - Commodities traded as CFDs (XTIUSD, XNGUSD)

Confirmed symbol names (Darwinex terminal, 2026-06-04):
  Nasdaq 100 → NDX   (NOT NAS100 / US100 / NDX100)
  S&P 500    → SP500
  DAX 40     → GDAXI (NOT GER40 / DAX40 / DE40)
  Dow Jones  → WS30
  FTSE 100   → UK100

Run to (re)seed the MT5 portion of the catalog:
    python -m data_platform.providers.mt5.to_catalog
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from data_platform.core import (
    AssetClass,
    Instrument,
    InstrumentCatalog,
    InstrumentClass,
    InstrumentId,
    Symbol,
    Venue,
    load_catalog,
)

# ---------------------------------------------------------------------------
# Tick history start dates (confirmed via binary-search probe, 2026-06-04)
# ---------------------------------------------------------------------------
_TICK_STARTS: dict[str, date] = {
    "NDX":     date(2018,  1, 24),
    "SP500":   date(2023,  3, 27),
    "GDAXI":   date(2018,  1, 25),
    "WS30":    date(2018,  1, 31),
    "UK100":   date(2018,  1, 25),
    "XAUUSD":  date(2018,  1, 25),
    "XAGUSD":  date(2018,  1, 25),
    "XTIUSD":  date(2019, 12, 27),
    "XNGUSD":  date(2020,  1,  1),
    "EURUSD":  date(2011, 12, 19),
    "GBPUSD":  date(2011, 12, 19),
    "USDJPY":  date(2011, 12, 19),
    "AUDUSD":  date(2011, 12, 19),
    "USDCHF":  date(2011, 12, 19),
    "AUDNZD":  date(2011, 12, 19),
    "EURCHF":  date(2011, 12, 19),
}

# Venue for CFD instruments — use XNAS as a proxy venue for Darwinex CFDs
# (no official MIC; XNAS is used to avoid collision with futures XCME/XNYM)
_DARWINEX = Venue("XNAS")


def _info(mt5_symbol: str) -> dict:
    return {
        "source_symbols": {"mt5": mt5_symbol},
        "tick_history_start": _TICK_STARTS.get(mt5_symbol, "unknown"),
        "broker": "Darwinex",
        "server": "liveUK-mt5.darwinex.com",
    }


def build_mt5_instruments() -> list[Instrument]:
    """Return Instrument rows for Darwinex CFD instruments."""
    return [
        # ── Index CFDs ───────────────────────────────────────────────────────
        Instrument(
            id=InstrumentId(Symbol("NDX"), _DARWINEX),
            raw_symbol="NDX",
            asset_class=AssetClass.INDEX,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="USD",
            multiplier=1.0,
            margin_init=0.20,       # 5:1 leverage = 20% margin (confirmed)
            venue_name="Darwinex",
            description="Nasdaq 100 Index CFD (Darwinex NDX)",
            data_source="mt5",
            activation=_TICK_STARTS["NDX"],
            info=_info("NDX"),
        ),
        Instrument(
            id=InstrumentId(Symbol("SP500"), _DARWINEX),
            raw_symbol="SP500",
            asset_class=AssetClass.INDEX,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="USD",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="S&P 500 Index CFD (Darwinex SP500)",
            data_source="mt5",
            activation=_TICK_STARTS["SP500"],
            info=_info("SP500"),
        ),
        Instrument(
            id=InstrumentId(Symbol("GDAXI"), _DARWINEX),
            raw_symbol="GDAXI",
            asset_class=AssetClass.INDEX,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="EUR",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="DAX 40 Index CFD (Darwinex GDAXI)",
            data_source="mt5",
            activation=_TICK_STARTS["GDAXI"],
            info=_info("GDAXI"),
        ),
        Instrument(
            id=InstrumentId(Symbol("WS30"), _DARWINEX),
            raw_symbol="WS30",
            asset_class=AssetClass.INDEX,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="USD",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="Dow Jones 30 CFD (Darwinex WS30)",
            data_source="mt5",
            activation=_TICK_STARTS["WS30"],
            info=_info("WS30"),
        ),
        Instrument(
            id=InstrumentId(Symbol("UK100"), _DARWINEX),
            raw_symbol="UK100",
            asset_class=AssetClass.INDEX,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="GBP",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="FTSE 100 CFD (Darwinex UK100)",
            data_source="mt5",
            activation=_TICK_STARTS["UK100"],
            info=_info("UK100"),
        ),
        # ── Commodities ──────────────────────────────────────────────────────
        Instrument(
            id=InstrumentId(Symbol("XTIUSD"), _DARWINEX),
            raw_symbol="XTIUSD",
            asset_class=AssetClass.COMMODITY,
            instrument_class=InstrumentClass.CFD,
            price_precision=2,
            price_increment=0.01,
            quote_currency="USD",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="WTI Crude Oil CFD (Darwinex XTIUSD)",
            data_source="mt5",
            activation=_TICK_STARTS["XTIUSD"],
            info=_info("XTIUSD"),
        ),
        Instrument(
            id=InstrumentId(Symbol("XNGUSD"), _DARWINEX),
            raw_symbol="XNGUSD",
            asset_class=AssetClass.COMMODITY,
            instrument_class=InstrumentClass.CFD,
            price_precision=3,
            price_increment=0.001,
            quote_currency="USD",
            multiplier=1.0,
            margin_init=0.20,
            venue_name="Darwinex",
            description="Natural Gas CFD (Darwinex XNGUSD)",
            data_source="mt5",
            activation=_TICK_STARTS["XNGUSD"],
            info=_info("XNGUSD"),
        ),
        # ── FX pairs ─────────────────────────────────────────────────────────
        Instrument(
            id=InstrumentId(Symbol("EURUSD"), _DARWINEX),
            raw_symbol="EURUSD",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="USD",
            multiplier=1.0,
            venue_name="Darwinex",
            description="EUR/USD spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["EURUSD"],
            info=_info("EURUSD"),
        ),
        Instrument(
            id=InstrumentId(Symbol("GBPUSD"), _DARWINEX),
            raw_symbol="GBPUSD",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="USD",
            multiplier=1.0,
            venue_name="Darwinex",
            description="GBP/USD spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["GBPUSD"],
            info=_info("GBPUSD"),
        ),
        Instrument(
            id=InstrumentId(Symbol("USDJPY"), _DARWINEX),
            raw_symbol="USDJPY",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=3,
            price_increment=0.001,
            quote_currency="JPY",
            multiplier=1.0,
            venue_name="Darwinex",
            description="USD/JPY spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["USDJPY"],
            info=_info("USDJPY"),
        ),
        Instrument(
            id=InstrumentId(Symbol("AUDUSD"), _DARWINEX),
            raw_symbol="AUDUSD",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="USD",
            multiplier=1.0,
            venue_name="Darwinex",
            description="AUD/USD spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["AUDUSD"],
            info=_info("AUDUSD"),
        ),
        Instrument(
            id=InstrumentId(Symbol("USDCHF"), _DARWINEX),
            raw_symbol="USDCHF",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="CHF",
            multiplier=1.0,
            venue_name="Darwinex",
            description="USD/CHF spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["USDCHF"],
            info=_info("USDCHF"),
        ),
        Instrument(
            id=InstrumentId(Symbol("AUDNZD"), _DARWINEX),
            raw_symbol="AUDNZD",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="NZD",
            multiplier=1.0,
            venue_name="Darwinex",
            description="AUD/NZD spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["AUDNZD"],
            info=_info("AUDNZD"),
        ),
        Instrument(
            id=InstrumentId(Symbol("EURCHF"), _DARWINEX),
            raw_symbol="EURCHF",
            asset_class=AssetClass.FX,
            instrument_class=InstrumentClass.SPOT,
            price_precision=5,
            price_increment=0.00001,
            quote_currency="CHF",
            multiplier=1.0,
            venue_name="Darwinex",
            description="EUR/CHF spot (Darwinex)",
            data_source="mt5",
            activation=_TICK_STARTS["EURCHF"],
            info=_info("EURCHF"),
        ),
    ]


def main() -> None:
    catalog = load_catalog()
    instruments = build_mt5_instruments()
    catalog.add_many(instruments)
    catalog.save()
    print(f"Seeded {len(instruments)} MT5 instruments into catalog.")
    mt5_rows = catalog.filter(data_source="mt5")
    print(f"Total MT5 instruments in catalog: {len(mt5_rows)}")
    for inst in sorted(mt5_rows, key=lambda i: str(i.id)):
        print(f"  {str(inst.id):<18} {inst.asset_class.value:<12} {inst.raw_symbol}")


if __name__ == "__main__":
    main()
