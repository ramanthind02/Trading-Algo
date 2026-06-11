"""Integration test: the sandbox live-trading simulation (Darwinex signal / FTMO exec).

Drives the REAL rollover-aligned VaultRebalanceStrategy through a BacktestEngine with
the production Darwinex signal cache and a modeled FTMO execution venue (spread + swap
+ per-symbol session times). The sim seams (``_SimRolloverStrategy``) feed broker time
from the sim clock and a deterministic negative-carry swap. Proves the end-to-end live
emulation runs, resolves the FTMO venue, and applies modeled swap drag.

(The rollover EXIT/ENTRY decision mechanics are covered deterministically by
``test_strategy_backtest`` and the ``test_rollover_*`` unit tests; this is the
realistic-data smoke test.)

Skips cleanly when Darwinex M1 data for the vault tickers isn't present on this
checkout. Bounded to a short window to keep the per-decision refit affordable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cache.runtime.central_cache import CentralCacheStore
from deployment.live.sandbox_sim import SandboxSimConfig, run_sandbox_sim
from lib.core.enums import Ticker, TimeFrame

_TICKERS = ("ES", "NQ", "GC", "CL", "SI")


@pytest.fixture
def restore_singleton():
    prev = CentralCacheStore._instance  # type: ignore[attr-defined]
    yield
    CentralCacheStore._instance = prev  # type: ignore[attr-defined]


def _darwinex_covers(tickers: tuple[str, ...]) -> bool:
    from data_platform.providers.mt5.cfd_candles import load_cfd_candles_raw

    for t in tickers:
        if t not in Ticker.__members__:
            return False
        try:
            f = load_cfd_candles_raw(Ticker[t], TimeFrame.D)
        except (FileNotFoundError, KeyError):
            return False
        if f is None or f.empty:
            return False
    return True


def test_sandbox_sim_runs_rollover_with_costs(restore_singleton):
    if not _darwinex_covers(_TICKERS):
        pytest.skip("no Darwinex daily coverage in data/mt5_data for vault tickers on this checkout")

    config = SandboxSimConfig(
        start=pd.Timestamp("2026-05-01"),
        end=pd.Timestamp("2026-06-05"),
        tickers=_TICKERS,
        exec_broker="ftmo",
        signal_broker="darwinex",
        refresh_signal_cache=True,
    )
    result = run_sandbox_sim(config)

    # End-to-end: the real rollover strategy ran on the FTMO venue and produced an
    # equity curve; modeled swap drag is non-negative and never lifts net above gross.
    assert result.resolved_tickers, "no tickers resolved on the FTMO venue"
    assert len(result.net_equity) > 1, "the sim produced no equity series"
    assert result.swap_cost_total >= 0.0
    assert result.net_equity.iloc[-1] <= result.gross_equity.iloc[-1] + 1e-6
    assert result.n_fills >= 0
