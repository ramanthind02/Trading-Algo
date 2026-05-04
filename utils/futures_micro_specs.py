"""Canonical listed micro-futures economics for research (continuous) tickers.

Single source of truth for dollar-per-index-point multipliers used by:

- ``portfolio_research`` futures simulation (``FuturesInstrumentSpec.multiplier``)
- ``scripts.enigma_live_forecast`` micro contract counts
- ``execution.PositionSizer.from_listed_micro`` (optional live sizing)

Prices are always the **research / continuous** index quote (e.g. ES index-style
level); ``micro_dollars_per_point`` is USD P&L per 1.00 point move for **one**
micro contract (MES/MNQ/MGC/MYM/M2K).

Illustrative margins are for the research futures sim only; refresh against
exchange sheets when relying on margin/leverage diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping


@dataclass(frozen=True)
class ListedMicroFuturesSpec:
    """CME/CBOT/COMEX-style micro vs mini pair for one research ticker."""

    micro_symbol: str
    mini_symbol: str
    micro_dollars_per_point: float
    mini_dollars_per_point: float
    exchange: str
    illustrative_margin_long_usd: float
    illustrative_margin_short_usd: float

    def __post_init__(self) -> None:
        if self.micro_dollars_per_point <= 0 or self.mini_dollars_per_point <= 0:
            raise ValueError("Dollars-per-point values must be positive")
        if self.illustrative_margin_long_usd < 0 or self.illustrative_margin_short_usd < 0:
            raise ValueError("Margin placeholders must be non-negative")


_CANONICAL: Final[Mapping[str, ListedMicroFuturesSpec]] = {
    "ES": ListedMicroFuturesSpec(
        micro_symbol="MES",
        mini_symbol="ES",
        micro_dollars_per_point=5.0,
        mini_dollars_per_point=50.0,
        exchange="CME",
        illustrative_margin_long_usd=2_413.0,
        illustrative_margin_short_usd=2_265.0,
    ),
    "NQ": ListedMicroFuturesSpec(
        micro_symbol="MNQ",
        mini_symbol="NQ",
        micro_dollars_per_point=2.0,
        mini_dollars_per_point=20.0,
        exchange="CME",
        illustrative_margin_long_usd=3_653.0,
        illustrative_margin_short_usd=3_576.0,
    ),
    "YM": ListedMicroFuturesSpec(
        micro_symbol="MYM",
        mini_symbol="YM",
        micro_dollars_per_point=0.5,
        mini_dollars_per_point=5.0,
        exchange="CBOT",
        illustrative_margin_long_usd=0.0,
        illustrative_margin_short_usd=0.0,
    ),
    "RTY": ListedMicroFuturesSpec(
        micro_symbol="M2K",
        mini_symbol="RTY",
        micro_dollars_per_point=5.0,
        mini_dollars_per_point=50.0,
        exchange="CME",
        illustrative_margin_long_usd=0.0,
        illustrative_margin_short_usd=0.0,
    ),
    "GC": ListedMicroFuturesSpec(
        micro_symbol="MGC",
        mini_symbol="GC",
        micro_dollars_per_point=10.0,
        mini_dollars_per_point=100.0,
        exchange="COMEX",
        illustrative_margin_long_usd=2_817.0,
        illustrative_margin_short_usd=2_817.0,
    ),
    # TLT research leg: sized against ZN (no micro); same dollars/point for micro+mini slots.
    "TLT": ListedMicroFuturesSpec(
        micro_symbol="ZN",
        mini_symbol="ZN",
        micro_dollars_per_point=1_000.0,
        mini_dollars_per_point=1_000.0,
        exchange="CBOT",
        illustrative_margin_long_usd=0.0,
        illustrative_margin_short_usd=0.0,
    ),
}


def canonical_listed_micro_futures() -> Mapping[str, ListedMicroFuturesSpec]:
    """Return the immutable canonical table (research ticker → spec)."""
    return _CANONICAL


def listed_micro_futures_row(research_ticker: str) -> ListedMicroFuturesSpec | None:
    """Lookup by research ticker (``ES``, ``NQ``, …); ``None`` if unknown."""
    return _CANONICAL.get(research_ticker)


def micro_contract_fractional_and_whole(
    *,
    futures_index_price: float,
    position_fraction: float,
    capital_usd: float,
    micro_dollars_per_point: float,
) -> tuple[float, int]:
    """Core sizing: target notional / (price × micro $/point), then banker's round.

    Matches ``portfolio_research.futures_sim`` and Enigma prop rounding
    (``int(round(...))``) for the default path.

    Returns
    -------
    (contracts_fractional, contracts_whole)
    """
    if futures_index_price <= 0 or micro_dollars_per_point <= 0:
        return 0.0, 0
    notional_per_contract = futures_index_price * micro_dollars_per_point
    target_dollars = position_fraction * capital_usd
    contracts_fractional = (
        target_dollars / notional_per_contract if notional_per_contract > 0 else 0.0
    )
    contracts_whole = int(round(contracts_fractional))
    return contracts_fractional, contracts_whole
