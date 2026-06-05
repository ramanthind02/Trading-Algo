"""Fixed-date roll rules for futures continuous contracts.

Each ticker has a RollRule defining when the continuous contract rolls
from the expiring front-month to the next contract. These rules
capture the fixed-date fallback schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from lib.core.enums import Ticker


@dataclass(frozen=True)
class RollRule:
    ticker: Ticker
    rollover_offset: int
    reference_point: Literal["expiration", "month_end"]
    description: str


ROLL_RULES: Dict[Ticker, RollRule] = {
    # Equity Indices: roll ~5 trading days before quarterly expiration
    Ticker.ES: RollRule(Ticker.ES, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.NQ: RollRule(Ticker.NQ, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.YM: RollRule(Ticker.YM, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.RTY: RollRule(Ticker.RTY, -5, "expiration", "5 days before quarterly expiration"),

    # Energy: roll ~3 trading days before expiration
    Ticker.CL: RollRule(Ticker.CL, -3, "expiration", "3 days before monthly expiration"),
    Ticker.HO: RollRule(Ticker.HO, -3, "expiration", "3 days before monthly expiration"),

    # Metals: roll ~2 days from month end
    Ticker.GC: RollRule(Ticker.GC, 2, "month_end", "2 days from month end"),
    Ticker.HG: RollRule(Ticker.HG, 2, "month_end", "2 days from month end"),
    Ticker.SI: RollRule(Ticker.SI, 2, "month_end", "2 days from month end"),
    Ticker.PL: RollRule(Ticker.PL, 2, "month_end", "2 days from month end"),

    # FX Currencies: roll ~2 trading days before quarterly expiration
    Ticker.EU: RollRule(Ticker.EU, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.JY: RollRule(Ticker.JY, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.BP: RollRule(Ticker.BP, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.CD: RollRule(Ticker.CD, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.SF: RollRule(Ticker.SF, -2, "expiration", "2 days before quarterly expiration"),

    # Agricultural: roll ~8 days from prior month-end
    Ticker.C: RollRule(Ticker.C, 8, "month_end", "8 days from prior month end"),
    Ticker.S: RollRule(Ticker.S, 8, "month_end", "8 days from prior month end"),
    Ticker.W: RollRule(Ticker.W, 8, "month_end", "8 days from prior month end"),
    Ticker.GF: RollRule(Ticker.GF, 8, "month_end", "8 days from prior month end"),

    # Fixed Income: roll ~3 days from month end
    Ticker.TY: RollRule(Ticker.TY, 3, "month_end", "3 days from month end"),
    Ticker.FV: RollRule(Ticker.FV, 3, "month_end", "3 days from month end"),
    Ticker.US: RollRule(Ticker.US, 3, "month_end", "3 days from month end"),
    Ticker.TU: RollRule(Ticker.TU, 3, "month_end", "3 days from month end"),
    Ticker.TLT: RollRule(Ticker.TLT, 3, "month_end", "3 days from month end"),
}


def get_roll_rule(ticker: Ticker) -> RollRule:
    """Look up the roll rule for a given ticker.

    Parameters
    ----------
    ticker : Ticker
        The futures ticker to look up.

    Returns
    -------
    RollRule
        The roll rule for the ticker.

    Raises
    ------
    ValueError
        If the ticker has no defined roll rule.
    """
    if ticker not in ROLL_RULES:
        raise ValueError(f"No roll rule defined for {ticker}")
    return ROLL_RULES[ticker]


def get_all_roll_rules() -> Dict[Ticker, RollRule]:
    """Return a copy of all roll rules keyed by Ticker.

    Returns
    -------
    Dict[Ticker, RollRule]
        Mapping from every Ticker to its RollRule.
    """
    return dict(ROLL_RULES)
