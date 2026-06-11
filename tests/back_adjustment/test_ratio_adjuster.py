"""Unit tests for the proportional (RATIO) continuous-future back-adjustment.

Pure-function tests on synthetic data — no repo data dependency. The data-backed
regression checks (real ratio store, EWSD repoint, prop loader) live in
``tests/back_adjustment/test_ratio_repoint_regression.py`` and skip when the
ratio parquet store is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.providers.norgate.backadjust.ratio_adjuster import (
    RatioRoll,
    apply_ratio_adjustment,
    build_ratio_close,
    build_ratio_factor,
    recover_rolls,
)


def _synthetic_pair() -> tuple[pd.Series, pd.Series]:
    """Build an unadjusted series with two known rolls and its additive partner.

    Unadjusted segments:
        seg A: bars 0..9   around 100  (old_close at roll1 = unadj[9])
        seg B: bars 10..19 around 50   (new_close at roll1 = unadj[10])
        seg C: bars 20..29 around 200  (new_close at roll2 = unadj[20])
    The additive partner is unadj + a piecewise-constant offset that steps at
    each roll by (new_close - old_close), so recover_rolls must find both rolls.
    """
    dates = pd.bdate_range("2020-01-01", periods=30)
    unadj = pd.Series(
        np.concatenate([
            np.full(10, 100.0) + np.arange(10) * 0.1,
            np.full(10, 50.0) + np.arange(10) * 0.1,
            np.full(10, 200.0) + np.arange(10) * 0.1,
        ]),
        index=dates,
        name="close",
    )
    # additive offset: 0 on the last segment, steps backward by the roll gaps
    old1, new1 = unadj.iloc[9], unadj.iloc[10]
    old2, new2 = unadj.iloc[19], unadj.iloc[20]
    gap1 = new1 - old1   # roll 1 gap
    gap2 = new2 - old2   # roll 2 gap
    offset = np.concatenate([
        np.full(10, gap1 + gap2),   # seg A: both later gaps
        np.full(10, gap2),          # seg B: only the last gap
        np.full(10, 0.0),           # seg C: anchor
    ])
    adj = (unadj + offset).rename("close")
    return adj, unadj


def test_recover_rolls_finds_both_rolls() -> None:
    adj, unadj = _synthetic_pair()
    rolls = recover_rolls(adj, unadj)
    assert len(rolls) == 2
    assert rolls[0].date == unadj.index[10]
    assert rolls[1].date == unadj.index[20]
    assert np.isclose(rolls[0].old_close, unadj.iloc[9])
    assert np.isclose(rolls[0].new_close, unadj.iloc[10])


def test_ratio_factor_is_one_on_last_segment() -> None:
    adj, unadj = _synthetic_pair()
    rolls = sorted(recover_rolls(adj, unadj), key=lambda r: r.date)
    factor = build_ratio_factor(pd.DatetimeIndex(unadj.index), rolls)
    # last segment (bars 20..29) keeps the true unadjusted price
    assert np.allclose(factor[20:], 1.0)
    # earlier segments are scaled by the product of later ratios
    assert factor[0] != 1.0 and factor[10] != 1.0


def test_ratio_preserves_within_segment_pct_returns() -> None:
    adj, unadj = _synthetic_pair()
    ratio = build_ratio_close(adj, unadj)
    # within segment A (bars 0..9) the ratio close is a constant multiple of the
    # unadjusted close, so %-returns are identical
    u_ret = unadj.iloc[:10].pct_change().dropna()
    r_ret = ratio.iloc[:10].pct_change().dropna()
    assert np.allclose(u_ret.to_numpy(), r_ret.to_numpy())


def test_ratio_never_negative_on_positive_source() -> None:
    adj, unadj = _synthetic_pair()
    ratio = build_ratio_close(adj, unadj)
    assert (ratio > 0).all()


def test_apply_ratio_adjustment_passes_volume_through() -> None:
    adj, unadj = _synthetic_pair()
    rolls = recover_rolls(adj, unadj)
    frame = pd.DataFrame({
        "open": unadj.to_numpy(),
        "high": unadj.to_numpy() + 1.0,
        "low": unadj.to_numpy() - 1.0,
        "close": unadj.to_numpy(),
        "volume": np.arange(len(unadj)),
    }, index=unadj.index)
    out = apply_ratio_adjustment(frame, rolls)
    # volume untouched, OHLC scaled, input not mutated
    assert (out["volume"].to_numpy() == np.arange(len(unadj))).all()
    assert out["close"].iloc[0] != frame["close"].iloc[0]
    assert frame["close"].iloc[0] == unadj.iloc[0]


def test_no_rolls_returns_source_unchanged() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    unadj = pd.Series([10.0, 11.0, 12.0, 11.5, 12.5], index=dates, name="close")
    out = build_ratio_close(unadj.rename("close"), unadj)
    assert np.allclose(out.to_numpy(), unadj.to_numpy())
