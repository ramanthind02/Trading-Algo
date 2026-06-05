"""WP-3 realism experiment test — spreads vs market orders (NDX 2026).

Exercises :mod:`research.portfolio.pnl.experiments.spread_vs_market` end to end on
the NDX 2026 MT5 intraday fixture and asserts the execution economics are
*directionally sane*:

* ``MARKET_ON_OPEN`` crosses the spread → every entry is a **TAKER** fill that
  **pays** the spread (liquidity-signed half-spread ``<= 0`` vs mid).
* ``LIMIT_AT_TOUCH`` rests at the near touch → fills are **MAKER** and **capture**
  the spread (liquidity-signed half-spread ``>= 0``), i.e. ``>=`` the MARKET value.

This is the spread-vs-market acceptance check from WP-3
``docs/refactor/nautilus/03_backtest_validation_lane.md``: a ``LIMIT_AT_TOUCH``
policy shows measurable, signed spread capture vs ``MARKET_ON_OPEN``.

The test runs a deliberately **small 2-session window** (2026-02-24..26 — the
first dates where both M1 bars and bid/ask ticks exist) into a throwaway catalog
so it stays fast. Skips cleanly when ``data/mt5_data/NDX`` is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_platform.nautilus.ingest import mt5_data_root

_SYMBOL = "NDX"

pytestmark = pytest.mark.skipif(
    not (mt5_data_root() / _SYMBOL).exists(),
    reason=f"data/mt5_data/{_SYMBOL} not present",
)

# A two-session window where BOTH M1 bars and bid/ask ticks exist (bars start
# 2026-02-23; ticks start 2026-01-02). Small on purpose — the spread economics
# are a per-fill property, so two sessions are enough to assert direction.
_START = pd.Timestamp("2026-02-24", tz="UTC")
_END = pd.Timestamp("2026-02-26", tz="UTC")


def _run_window(tmp_path):
    """Run the experiment over the small window into a throwaway catalog root."""
    from research.portfolio.pnl.experiments.spread_vs_market import run_experiment

    return run_experiment(
        start=_START,
        end=_END,
        fraction=1.0,
        catalog_root=tmp_path / "_catalogs",
    )


def _row(df: pd.DataFrame, policy: str) -> pd.Series:
    sub = df[df["policy"] == policy]
    assert len(sub) == 1, f"expected exactly one {policy!r} row, got {len(sub)}"
    return sub.iloc[0]


def test_spread_vs_market_experiment_runs_and_is_economically_sane(tmp_path) -> None:
    df = _run_window(tmp_path)

    # The experiment produced a row per policy with the expected schema.
    assert not df.empty
    for col in (
        "policy",
        "n_sessions",
        "n_entry_fills",
        "maker_fills",
        "taker_fills",
        "avg_signed_spread_px",
        "total_log_return",
    ):
        assert col in df.columns, f"missing column {col!r}"

    market = _row(df, "MARKET_ON_OPEN")
    at_touch = _row(df, "LIMIT_AT_TOUCH")

    # Both policies actually filled their entries (otherwise the comparison is
    # vacuous) — the window is chosen so the touch is reached.
    assert int(market["n_entry_fills"]) > 0
    assert int(at_touch["n_entry_fills"]) > 0

    # MARKET crosses the spread → TAKER fills only, paying the spread (<= 0).
    assert int(market["taker_fills"]) == int(market["n_entry_fills"])
    assert int(market["maker_fills"]) == 0
    assert float(market["avg_signed_spread_px"]) <= 0.0

    # LIMIT_AT_TOUCH rests passively → MAKER fills, capturing the spread (>= 0).
    assert int(at_touch["maker_fills"]) == int(at_touch["n_entry_fills"])
    assert int(at_touch["taker_fills"]) == 0
    assert float(at_touch["avg_signed_spread_px"]) >= 0.0

    # The core directional claim: LIMIT_AT_TOUCH captures at least as much spread
    # as MARKET pays — maker captures, taker pays (>= 0 >= market).
    assert float(at_touch["avg_signed_spread_px"]) >= float(
        market["avg_signed_spread_px"]
    )
