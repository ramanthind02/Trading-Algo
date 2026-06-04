"""Calculate cumulative back-adjustment factors from detected roll events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from .roll_detector import RollEvent


@dataclass(frozen=True)
class AdjustmentFactor:
    roll_date: datetime
    gap_points: float
    cumulative_adjustment: float


def calculate_adjustments(
    roll_events: List[RollEvent],
) -> List[AdjustmentFactor]:
    """Compute cumulative adjustment factors from chronological roll events.

    Each adjustment's cumulative_adjustment = sum of gap_points for all
    rolls AFTER this one. Most recent roll has cumulative_adjustment = 0.
    """
    if not roll_events:
        return []

    n = len(roll_events)
    gaps = [e.gap_points for e in roll_events]

    suffix_sum = 0.0
    cumulative = [0.0] * n
    for i in range(n - 2, -1, -1):
        suffix_sum += gaps[i + 1]
        cumulative[i] = suffix_sum

    return [
        AdjustmentFactor(
            roll_date=roll_events[i].roll_date,
            gap_points=gaps[i],
            cumulative_adjustment=cumulative[i],
        )
        for i in range(n)
    ]
