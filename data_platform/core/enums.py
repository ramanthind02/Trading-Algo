"""Nautilus-aligned classification enums for the data-core model.

Names and members mirror NautilusTrader's model enums so the mapping is 1:1
when we migrate. We only include members the repo actually needs today;
adding more later is non-breaking.
"""
from __future__ import annotations

from enum import Enum


class AssetClass(Enum):
    """Broad asset class. Mirrors nautilus_trader.model.enums.AssetClass."""

    FX = "FX"
    EQUITY = "EQUITY"
    COMMODITY = "COMMODITY"
    DEBT = "DEBT"
    INDEX = "INDEX"
    CRYPTOCURRENCY = "CRYPTOCURRENCY"
    ALTERNATIVE = "ALTERNATIVE"


class InstrumentClass(Enum):
    """Instrument class. Mirrors nautilus_trader.model.enums.InstrumentClass."""

    SPOT = "SPOT"
    FUTURE = "FUTURE"
    FUTURES_SPREAD = "FUTURES_SPREAD"
    FORWARD = "FORWARD"
    CFD = "CFD"
    OPTION = "OPTION"
    WARRANT = "WARRANT"
    INDEX = "INDEX"


class PriceType(Enum):
    """Price basis of a bar. Mirrors nautilus_trader.model.enums.PriceType."""

    BID = "BID"
    ASK = "ASK"
    MID = "MID"
    LAST = "LAST"


class BarAggregation(Enum):
    """Bar aggregation method (subset). Mirrors NautilusTrader naming."""

    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"


class AggregationSource(Enum):
    """Where the bar was aggregated. Mirrors NautilusTrader."""

    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


class PriceAdjustment(Enum):
    """Price-adjustment convention for a stored series.

    Not a NautilusTrader enum — this is repo metadata describing how a stored
    bar series was adjusted, so consumers (σ estimation, P&L) pick the right
    series. See docs/library/Data/futures_backtesting_data_guide.md.
    """

    NONE = "NONE"                 # raw traded prices
    TOTAL_RETURN = "TOTAL_RETURN"  # splits + special + ordinary dividends
    CAPITAL = "CAPITAL"           # splits + reconstructions only
    BACK_ADJUSTED = "BACK_ADJUSTED"  # additive continuous-future (futures CCB)
    RATIO = "RATIO"               # proportional continuous-future (%-return / σ reference)


# ── MIC-style venue codes (Norgate exchange name -> MIC) ──────────────────
# Used by adapters to construct InstrumentId venues consistently.
# "OOTC" is a private MIC-style placeholder for Norgate's "OTC" bucket and the
# fallback for any unmapped exchange (no dots/dashes -> a valid Venue code).
FALLBACK_MIC: str = "OOTC"

NORGATE_EXCHANGE_TO_MIC: dict[str, str] = {
    "CME": "XCME",
    "CBOT": "XCBT",
    "NYMEX": "XNYM",
    "COMEX": "XCEC",
    "Nasdaq": "XNAS",
    "NYSE": "XNYS",
    "NYSE American": "XASE",
    "NYSE Arca": "ARCX",
    "Cboe BZX": "BATS",
    "IEX": "IEXG",
    "OTC": "OOTC",
}


def venue_mic_for_exchange(exchange_name: str | None) -> str:
    """Resolve a Norgate exchange name to a MIC, falling back to OOTC."""
    if not exchange_name:
        return FALLBACK_MIC
    return NORGATE_EXCHANGE_TO_MIC.get(exchange_name, FALLBACK_MIC)
