"""Unit tests for the pure helpers of deployment.live.forecast_engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from deployment.live.forecast_engine import (
    TickerWarmup,
    VaultForecastEngine,
    WarmupStatus,
    _ticker_str,
)
from lib.core.enums import Ticker


def test_ticker_str_handles_enum_and_string() -> None:
    assert _ticker_str(Ticker.ES) == "ES"
    assert _ticker_str("NQ") == "NQ"


def test_warmup_status_ready_requires_all_tickers() -> None:
    ws = WarmupStatus(
        min_bars=500,
        per_ticker=(
            TickerWarmup("ES", 600, True),
            TickerWarmup("GC", 400, False),
        ),
    )
    assert not ws.ready
    assert ws.not_ready() == ("GC",)

    ws_ok = WarmupStatus(
        min_bars=500,
        per_ticker=(TickerWarmup("ES", 600, True), TickerWarmup("GC", 700, True)),
    )
    assert ws_ok.ready
    assert ws_ok.not_ready() == ()


def test_warmup_status_empty_is_not_ready() -> None:
    assert not WarmupStatus(min_bars=500, per_ticker=()).ready


def _engine_with_fixed_coverage(start: str, end: str) -> VaultForecastEngine:
    """A forecast engine whose cache coverage window is pinned (no cache reads)."""
    eng = VaultForecastEngine()
    eng._required_tickers = ("ES",)  # type: ignore[attr-defined]
    cov = (pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime())
    eng._coverage_window = lambda: cov  # type: ignore[assignment,method-assign]
    return eng


def test_build_query_without_as_of_uses_full_coverage_end() -> None:
    eng = _engine_with_fixed_coverage("2020-01-01", "2026-01-01")
    q = eng._build_query()
    assert pd.Timestamp(q.end) == pd.Timestamp("2026-01-01")


def test_build_query_caps_end_at_as_of() -> None:
    """The causality bound: as_of earlier than coverage end tightens the window."""
    eng = _engine_with_fixed_coverage("2020-01-01", "2026-01-01")
    q = eng._build_query(as_of=datetime(2024, 6, 1))
    assert pd.Timestamp(q.end) == pd.Timestamp("2024-06-01")


def test_build_query_as_of_after_coverage_is_a_noop() -> None:
    """LIVE case: wall clock >= cache end -> bound never tightens below coverage."""
    eng = _engine_with_fixed_coverage("2020-01-01", "2026-01-01")
    q = eng._build_query(as_of=datetime(2030, 1, 1))
    assert pd.Timestamp(q.end) == pd.Timestamp("2026-01-01")


def test_build_query_as_of_is_tz_aware_safe() -> None:
    """A tz-aware as_of (Nautilus clock.utc_now()) compares cleanly vs naive end."""
    eng = _engine_with_fixed_coverage("2020-01-01", "2026-01-01")
    q = eng._build_query(as_of=datetime(2024, 6, 1, tzinfo=timezone.utc))
    assert pd.Timestamp(q.end) == pd.Timestamp("2024-06-01")


def test_build_query_end_is_monotone_in_as_of() -> None:
    eng = _engine_with_fixed_coverage("2020-01-01", "2026-01-01")
    e1 = pd.Timestamp(eng._build_query(as_of=datetime(2023, 1, 1)).end)
    e2 = pd.Timestamp(eng._build_query(as_of=datetime(2024, 1, 1)).end)
    assert e1 < e2


def test_instrument_returns_pivots_and_diffs() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])
    daily = pd.DataFrame(
        {
            "ticker": ["ES"] * 3 + ["GC"] * 3,
            "datetime": list(dates) * 2,
            "close": [100.0, 101.0, 102.0, 50.0, 49.0, 49.5],
        }
    )
    returns = VaultForecastEngine._instrument_returns(daily, ("ES", "GC"))
    assert list(returns.columns) == ["ES", "GC"]
    assert len(returns) == 2  # 3 closes -> 2 pct-changes
    assert returns.iloc[0]["ES"] == pytest.approx(0.01)  # 101/100 - 1
