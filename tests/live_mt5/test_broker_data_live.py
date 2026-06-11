"""Gated live smoke test for the MT5 daily fetch utility.

The live signal no longer fetches per-broker ``copy_rates`` (signals come from the
shared Darwinex scrape — see ``deployment/live/broker_data.refresh_signal_daily``),
but ``scripts.mt5_data_fetch.fetch_mt5_daily_candles`` is still used by the
Darwinex scraper / legacy cfd_prop pipeline. This validates that the live MT5
terminal returns clean daily bars via that utility. Read-only — no orders. Gated
by the shared FTMO-demo harness in this directory's conftest.
"""
from __future__ import annotations

import pytest

from data_platform.providers.mt5 import brokers
from scripts.mt5_data_fetch import fetch_mt5_daily_candles


@pytest.mark.parametrize("canonical", ["ES", "NQ", "GC"])
def test_ftmo_copy_rates_returns_clean_daily(live, canonical: str) -> None:
    native = brokers.resolve("ftmo", canonical)
    daily = fetch_mt5_daily_candles(live.mt5, canonical, native, 2000)
    assert not daily.empty, f"FTMO returned no daily bars for {native}"
    assert (daily["close"] > 0).all(), "broker daily closes must be positive"
    assert list(daily.columns)[:6] == ["datetime", "open", "high", "low", "close", "volume"]
    assert daily["datetime"].is_monotonic_increasing
