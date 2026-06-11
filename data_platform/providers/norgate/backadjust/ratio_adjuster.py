"""Proportional (RATIO) continuous-future back-adjustment.

The live pipeline stores an ADDITIVE continuous future (Norgate ``_CCB``):
each historical segment has a constant POINTS offset added so segments join at
rolls. That preserves absolute point moves but distorts %-returns (it inflates
old price levels, deflating old %-returns) and can push prices NEGATIVE (crude
Apr-2020), which makes log/%-returns undefined.

A RATIO (proportional) back-adjustment instead MULTIPLIES each historical
segment by the product of the LATER rolls' ``new_close/old_close`` ratios. This
preserves %-returns exactly within each contract, never introduces a negative,
and ties out to the unadjusted price on the most-recent segment.

There is no in-repo roll-event detection on the live data path — the canonical
``data/ohlc_data/{T}/D_{T}.parquet`` is Norgate's additive ``_CCB`` fetched
whole. We therefore recover the exact roll dates and per-roll
``old_close``/``new_close`` from the stored additive + unadjusted pair (the same
approach validated in ``scripts/ratio_backadjust_prototype.py``): the additive
offset ``adj - unadj`` is piecewise-constant and steps by ``new - old`` at each
roll, and the unadjusted close at the roll boundary gives both contracts'
closing prices.

This module is the pure-function core. ``ratio_driver.py`` is the I/O shell that
emits ``D_{T}_ratio.parquet`` (+ ``W_``/``M_``) alongside the existing files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A genuine roll gap is a material fraction of price; the stored OHLC is float32,
# so the additive offset (adj - unadj) carries ~1e-4 absolute quantization noise
# that must NOT be mistaken for a roll. Threshold the offset STEP relative to the
# price level: a real roll moves the offset by >= REL_ROLL_FLOOR of price.
REL_ROLL_FLOOR: float = 5e-3  # 0.5% of price
ABS_ROLL_FLOOR: float = 1e-3  # absolute floor for very small prices

OHLC_COLS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class RatioRoll:
    """A single roll event recovered from the additive/unadjusted pair."""

    date: pd.Timestamp
    old_close: float
    new_close: float

    @property
    def ratio(self) -> float:
        return self.new_close / self.old_close if self.old_close != 0.0 else 1.0


def recover_rolls(
    adj_close: pd.Series,
    unadj_close: pd.Series,
    *,
    rel_floor: float = REL_ROLL_FLOOR,
    abs_floor: float = ABS_ROLL_FLOOR,
) -> list[RatioRoll]:
    """Recover roll events from the additive offset series.

    ``offset = adj - unadj`` is piecewise-constant and steps at each roll by
    ``new_close - old_close``. At a roll date ``d``, ``unadj[d-]`` is the last
    bar of the expiring contract (``old_close``) and ``unadj[d]`` is the first
    bar of the new front month (``new_close``). Both are read straight off the
    unadjusted series.

    A real roll is detected when the offset step exceeds ``rel_floor`` of the
    prior bar's price level (float32 storage noise is ~1e-4 absolute, i.e. well
    below ``rel_floor``).
    """
    merged = (
        pd.concat([adj_close.rename("a"), unadj_close.rename("u")], axis=1)
        .dropna()
        .sort_index()
    )
    if merged.empty:
        return []
    offset = merged["a"] - merged["u"]
    step = offset.diff().abs().to_numpy()
    unadj_arr = merged["u"].to_numpy()
    idx = merged.index
    thresh = rel_floor * np.abs(np.roll(unadj_arr, 1))
    roll_mask = step > np.maximum(thresh, abs_floor)
    roll_positions = np.where(roll_mask)[0]
    return [
        RatioRoll(
            date=idx[i],
            old_close=float(unadj_arr[i - 1]),
            new_close=float(unadj_arr[i]),
        )
        for i in roll_positions
        if i != 0
    ]


def build_ratio_factor(dates: pd.DatetimeIndex, rolls: list[RatioRoll]) -> np.ndarray:
    """Per-bar multiplicative factor: product of ``ratio`` for every later roll.

    The most-recent segment (on/after the last roll) keeps the true unadjusted
    price (cumulative factor 1.0). For a bar at date ``d`` the factor is the
    product of ``new/old`` for every roll strictly AFTER ``d``.
    """
    factor = np.ones(len(dates), dtype=np.float64)
    for roll in rolls:
        factor[dates < roll.date] *= roll.ratio
    return factor


def apply_ratio_adjustment(
    unadj_ohlc: pd.DataFrame,
    rolls: list[RatioRoll],
) -> pd.DataFrame:
    """Multiplicatively back-adjust every OHLC column of an unadjusted frame.

    ``unadj_ohlc`` is indexed by date (the canonical schema). Volume is passed
    through unchanged. Returns a new frame; the input is not mutated.
    """
    result = unadj_ohlc.copy()
    if not rolls:
        return result
    sorted_rolls = sorted(rolls, key=lambda r: r.date)
    dates = pd.DatetimeIndex(result.index)
    factor = build_ratio_factor(dates, sorted_rolls)
    for col in OHLC_COLS:
        if col in result.columns:
            result[col] = (result[col].astype(np.float64) * factor)
    return result


def build_ratio_close(
    adj_close: pd.Series,
    unadj_close: pd.Series,
) -> pd.Series:
    """Convenience: recover rolls then return the ratio-adjusted close series."""
    rolls = recover_rolls(adj_close, unadj_close)
    factor = build_ratio_factor(pd.DatetimeIndex(unadj_close.index), sorted(rolls, key=lambda r: r.date))
    return (unadj_close.astype(np.float64) * factor).rename(unadj_close.name)
