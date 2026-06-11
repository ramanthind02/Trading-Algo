"""Unit tests for IB daily append filter + junction ratio alignment."""
from __future__ import annotations

import pandas as pd

from cache.runtime.ib_candle_ratio_align import prepare_ib_rows_for_central_cache_append


def _row(
    dt: str,
    o: float,
    h: float,
    l: float,
    c: float,
) -> dict[str, object]:
    return {
        "datetime": pd.Timestamp(dt),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": 1,
        "ticker": "ES",
        "timeframe": "D",
    }


def test_empty_incoming() -> None:
    r = prepare_ib_rows_for_central_cache_append(
        pd.DataFrame([_row("2024-01-01", 1, 1, 1, 1)]),
        pd.DataFrame(),
        apply_junction_ratio=True,
    )
    assert r.skip_reason == "empty_incoming"
    assert r.rows_kept == 0


def test_no_existing_cache_keeps_all_without_ratio() -> None:
    inc = pd.DataFrame([_row("2024-01-02", 10.0, 11.0, 9.0, 10.5)])
    r = prepare_ib_rows_for_central_cache_append(pd.DataFrame(), inc, apply_junction_ratio=True)
    assert not r.applied_ratio
    assert r.ratio is None
    assert r.rows_kept == 1
    assert abs(float(r.candles_df["close"].iloc[0]) - 10.5) < 1e-9


def test_append_only_then_ratio_scales_first_close_to_anchor() -> None:
    exist = pd.DataFrame(
        [
            _row("2024-01-01", 100.0, 101.0, 99.0, 100.0),
            _row("2024-01-05", 102.0, 103.0, 101.0, 102.0),
        ]
    )
    inc = pd.DataFrame(
        [
            _row("2024-01-03", 50.0, 51.0, 49.0, 50.0),
            _row("2024-01-06", 54.0, 55.0, 53.0, 54.0),
        ]
    )
    r = prepare_ib_rows_for_central_cache_append(exist, inc, apply_junction_ratio=True)
    assert r.rows_in == 2
    assert r.rows_kept == 1
    assert r.applied_ratio
    assert r.ratio is not None
    assert abs(r.ratio - (102.0 / 54.0)) < 1e-9
    assert abs(float(r.candles_df["close"].iloc[0]) - 102.0) < 1e-6
    assert abs(float(r.candles_df["open"].iloc[0]) - 54.0 * r.ratio) < 1e-6


def test_append_only_stk_skips_ratio() -> None:
    exist = pd.DataFrame([_row("2024-01-01", 10.0, 11.0, 9.0, 10.0)])
    inc = pd.DataFrame(
        [
            _row("2024-01-01", 99.0, 100.0, 98.0, 99.5),
            _row("2024-01-02", 100.0, 101.0, 99.0, 100.5),
        ]
    )
    r = prepare_ib_rows_for_central_cache_append(exist, inc, apply_junction_ratio=False)
    assert r.rows_kept == 1
    assert not r.applied_ratio
    assert abs(float(r.candles_df["close"].iloc[0]) - 100.5) < 1e-9


def test_all_sessions_on_or_before_cache_max_skipped() -> None:
    exist = pd.DataFrame([_row("2024-01-10", 1.0, 2.0, 0.5, 1.5)])
    inc = pd.DataFrame([_row("2024-01-09", 5.0, 6.0, 4.0, 5.5)])
    r = prepare_ib_rows_for_central_cache_append(exist, inc, apply_junction_ratio=True)
    assert r.skip_reason == "no_new_sessions"
    assert r.rows_kept == 0
