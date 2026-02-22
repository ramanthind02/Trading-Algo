"""Apply arithmetic back-adjustment to OHLC price data."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data_cleaning.back_adjustment.gap_calculator import AdjustmentFactor

OHLC_COLS = ("open", "high", "low", "close")


def apply_back_adjustment(
    df: pd.DataFrame,
    adjustments: List[AdjustmentFactor],
) -> pd.DataFrame:
    """Apply arithmetic back-adjustment to OHLC columns.

    - Pre-first-roll rows: adjusted by sum of ALL gaps
    - Between rolls i and i+1: adjusted by adjustments[i].cumulative_adjustment
    - On/after most recent roll: adjustment = 0
    """
    result = df.copy()

    if not adjustments:
        return result

    dt = pd.to_datetime(result["datetime"])
    adj_values = np.zeros(len(result), dtype=np.float64)

    total_gaps = sum(a.gap_points for a in adjustments)
    first_roll = adjustments[0].roll_date
    adj_values[dt < first_roll] = total_gaps

    for i in range(len(adjustments)):
        roll_dt = adjustments[i].roll_date
        next_dt = adjustments[i + 1].roll_date if i + 1 < len(adjustments) else None

        if next_dt is not None:
            mask = (dt >= roll_dt) & (dt < next_dt)
        else:
            mask = dt >= roll_dt

        adj_values[mask] = adjustments[i].cumulative_adjustment

    for col in OHLC_COLS:
        if col in result.columns:
            result[col] = result[col].astype(np.float64) + adj_values

    return result


def validate_adjusted_data(
    original: pd.DataFrame,
    adjusted: pd.DataFrame,
    adjustments: List[AdjustmentFactor],
) -> bool:
    """Validate back-adjusted data integrity. Raises ValueError on failure."""
    if len(original) != len(adjusted):
        raise ValueError(
            f"Row count mismatch: original={len(original)}, adjusted={len(adjusted)}"
        )

    for col in OHLC_COLS:
        if col in adjusted.columns:
            min_val = adjusted[col].min()
            if min_val < 0:
                raise ValueError(
                    f"Adjusted '{col}' contains negative prices (min={min_val:.4f})"
                )

    if "volume" in original.columns and "volume" in adjusted.columns:
        if not original["volume"].equals(adjusted["volume"]):
            raise ValueError("Volume column was modified during adjustment")

    for col in ("datetime", "timestamp"):
        if col in original.columns and col in adjusted.columns:
            if not original[col].equals(adjusted[col]):
                raise ValueError(f"'{col}' column was modified during adjustment")

    return True
