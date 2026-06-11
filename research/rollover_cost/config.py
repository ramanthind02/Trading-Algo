"""Static config for the rollover-cost study.

Symbol specs are snapshotted from the live Darwinex terminal (research/rollover_cost/spec_probe.py,
2026-06-05). Swap is quoted in POINTS/lot/day; the per-event bps cost is recomputed from the
event's own mid price so it is price-accurate (POINTS swap is a fixed point amount, so its bps
value drifts with price).

Rollover is at 00:00 BROKER TIME (= 17:00 ET). All windows below are broker wall-clock
(the timestamps stored in data/mt5_data are broker time mislabelled UTC).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    point: float            # price increment (== tick_size on Darwinex)
    tick_size: float
    contract_size: float
    swap_long_pts: float    # POINTS mode: points/lot/day. MARGIN_CCY mode: USD/lot/day.
    swap_short_pts: float
    triple_weekday: int     # MT5 swap_rollover3days: weekday charged 3x (0=Sun..6=Sat)
    swap_mode: str = "POINTS"  # MT5 swap_mode: "POINTS" (1) or "MARGIN_CCY" (3)

    def swap_bps(self, mid: float, *, is_long: bool) -> float:
        """Signed daily swap as bps of notional at price `mid` (negative = you pay).

        POINTS mode: the swap is quoted in price points/lot → bps = pts·point/mid.
        MARGIN_CCY mode (e.g. XTIUSD): the swap is quoted in the margin currency
        (USD) per lot → bps = swap_usd / (contract_size·mid). Probed from the live
        Darwinex terminal (research/rollover_cost/spec_probe.py).
        """
        val = self.swap_long_pts if is_long else self.swap_short_pts
        if self.swap_mode == "MARGIN_CCY":
            return val / (self.contract_size * mid) * 1e4
        return val * self.point / mid * 1e4


# canonical -> Darwinex symbol: ES->SP500, NQ->NDX, GC->XAUUSD, SI->XAGUSD, CL->XTIUSD.
# Probed live 2026-06-07: SP500/NDX/XAUUSD/XAGUSD are POINTS-mode; XTIUSD is MARGIN_CCY.
SPECS: dict[str, SymbolSpec] = {
    "SP500":  SymbolSpec("SP500",  0.1,   0.1,   10.0,   -10.85,   4.59, 5),
    "NDX":    SymbolSpec("NDX",    0.1,   0.1,   10.0,   -45.51,  18.79, 5),
    "XAUUSD": SymbolSpec("XAUUSD", 0.01,  0.01,  100.0,  -63.60,  38.90, 3),
    "XAGUSD": SymbolSpec("XAGUSD", 0.001, 0.001, 5000.0,  -8.90,   7.20, 3),
    "XTIUSD": SymbolSpec("XTIUSD", 0.01,  0.01,  1000.0,  42.20, -145.30, 3, "MARGIN_CCY"),
}

SYMBOLS: tuple[str, ...] = ("SP500", "NDX", "XAUUSD", "XAGUSD", "XTIUSD")

# Window geometry, in MINUTES relative to the 00:00-broker rollover.
EXIT_WINDOW_MIN = 60     # [rollover-60, rollover)
DEADZONE_MIN = 60        # [rollover, rollover+60) — no quotes
ENTRY_WINDOW_MIN = 60    # [rollover+60, rollover+120)
FETCH_PAD_MIN = 5        # extra padding either side when fetching from MT5

# Candidate exit offsets (minutes BEFORE rollover) to evaluate.
EXIT_OFFSETS_MIN = (60, 45, 30, 20, 15, 10, 7, 5, 3, 2, 1)
# Reference arrival mid is taken at this offset (the 'decision' moment).
ARRIVAL_OFFSET_MIN = 15

# Entry: limit offsets in TICKS below the open mid (for a buy; symmetric for a sell).
ENTRY_LIMIT_TICKS = (0, 1, 2, 3, 5, 8, 12, 20)
# Entry: time offsets (minutes after open) at which we snapshot spread + mid path.
ENTRY_SNAPSHOT_MIN = (1, 2, 5, 10, 15, 30, 45, 59)
# Fill horizons (minutes after open) to evaluate limit fills / chase deadlines.
FILL_HORIZONS_MIN = (1, 2, 5, 10, 15, 30, 60)
