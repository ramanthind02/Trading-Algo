"""Integration smoke test for the demand-driven MT5 tick cache (`ensure_ticks`).

Proves the cold fetch populates the cache and a warm re-read returns an identical
row count. Cache-backed by default; if the window is not already cached it needs a
live MT5 terminal, so it skips cleanly when neither cache nor terminal is available
(per the repo integration-test policy).

Was `scripts/_ensure_ticks_smoketest.py` (a loose __main__ probe).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_platform.providers.mt5.tick_cache import ensure_ticks, is_cached

SYM = "NDX"
# A small 2-hour window on a recent weekday afternoon (UTC) — fast to fetch.
START = datetime(2026, 6, 3, 19, 0, tzinfo=timezone.utc)
END = datetime(2026, 6, 3, 21, 0, tzinfo=timezone.utc)


def test_ensure_ticks_warm_read_matches_cold() -> None:
    if not is_cached(SYM, START, END):
        mt5 = pytest.importorskip("MetaTrader5")
        if not mt5.initialize():
            pytest.skip("ensure_ticks window not cached and no MT5 terminal available")

    df_cold = ensure_ticks(SYM, START, END)
    assert is_cached(SYM, START, END), "cold fetch must populate the cache"

    df_warm = ensure_ticks(SYM, START, END)
    assert len(df_cold) == len(df_warm), "warm read must match cold read row count"
