"""Deterministic end-to-end test of the rollover-aligned VaultRebalanceStrategy
in a Nautilus ``BacktestEngine`` (NETTING ``MT5`` venue).

The *real* strategy runs over a synthetic quote stream spanning a full broker day.
The sim clock is mapped to broker wall-clock via the ``_now_broker`` seam (offset 0
→ sim UTC IS broker time), the forecast is injected (ES=+0.5) via
``_evaluate_targets``, and a **negative-carry** swap via ``_fetch_swap``. It
validates the whole live cycle in one run:

* **ENTRY** at the symbol's reopen (01:05 broker) → demand quote → ``net_rebalance``
  → market → sandbox fill → net long ~10 lots;
* **EXIT** at T-15 before the 00:00 rollover (23:45 broker) → carry-aware
  ``decide_exit_leg``: the swap-negative held leg is flattened → net 0.

(The positive-carry HOLD branch and the swap/spread maths are covered by the unit
tests ``test_rollover_market`` / ``test_rollover_schedule`` and the overlay's own
tests; only ONE ``BacktestEngine`` is created per process — multiple engines in one
pytest process stall on shared Nautilus core state.)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from deployment.live.config_mt5_sandbox import vendored_adapter_path
from deployment.live.forecast_engine import ForecastResult, TickerWarmup, WarmupStatus
from deployment.live.runtime.rollover_market import SwapInfo
from deployment.live.runtime.rollover_schedule import SymbolSession
from deployment.live.vault_strategy import VaultRebalanceConfig, VaultRebalanceStrategy

_ADAPTER = str(vendored_adapter_path())
if _ADAPTER not in sys.path:
    sys.path.insert(0, _ADAPTER)

_SESSIONS = {"US500.cash": SymbolSession(open_h=1, open_m=5, close_h=23, close_m=49)}


class _FakeEngine:
    required_tickers = ("ES",)

    def load(self) -> "_FakeEngine":
        return self


class _RolloverStrategy(VaultRebalanceStrategy):
    """Strategy with injected forecast, broker clock (= sim UTC), swap, sessions."""

    def _build_engine(self):  # type: ignore[override]
        return _FakeEngine()

    def _evaluate_targets(self) -> ForecastResult:  # type: ignore[override]
        return ForecastResult(
            targets={"ES": 0.5},
            forecast_scores={"ES": 1.0},
            as_of=datetime(2026, 6, 10, tzinfo=timezone.utc),
            warmup=WarmupStatus(min_bars=0, per_ticker=(TickerWarmup("ES", 999, True),)),
            ready=True,
        )

    def _now_broker(self):  # type: ignore[override]
        return self.clock.utc_now().replace(tzinfo=None)  # sim UTC IS broker wall-clock here

    def _fetch_swap(self, native: str):  # type: ignore[override]
        # swap_long -200 pts → negative carry when long → overlay candidate.
        return SwapInfo(native, -200.0, +1.0, 0.01, 1.0, 1, 5)

    def _load_sessions(self):  # type: ignore[override]
        return dict(_SESSIONS)


def _instrument():
    from mt5connect.parsing import parse_symbol_info  # type: ignore import-not-found

    return parse_symbol_info(SimpleNamespace(
        name="US500.cash", digits=2, volume_step=0.01, volume_min=0.01, volume_max=100.0,
        trade_contract_size=1.0, margin_initial=0.0, margin_maintenance=0.0,
        currency_base="USD", currency_profit="USD",
    ))


def _quotes(instrument, start: datetime, minutes: int):
    from nautilus_trader.model.data import QuoteTick

    out = []
    for i in range(minutes):
        ts = int((start + timedelta(minutes=i)).timestamp() * 1e9)
        out.append(QuoteTick(
            instrument_id=instrument.id,
            bid_price=instrument.make_price(4999.5), ask_price=instrument.make_price(5000.5),
            bid_size=instrument.make_qty(10), ask_size=instrument.make_qty(10),
            ts_event=ts, ts_init=ts,
        ))
    return out


def test_rollover_full_cycle_enters_then_flattens_negative_carry():
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money

    instrument = _instrument()
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id="BACKTEST-001", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=Venue("MT5"), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     base_currency=USD, starting_balances=[Money(100_000, USD)])
    engine.add_instrument(instrument)

    # One broker day: 00:50 (Wed 2026-06-10) → next-day 00:05, 1-minute quotes.
    # reopen 01:05 → ENTER; 23:45 (T-15 before 00:00) → EXIT.
    start = datetime(2026, 6, 10, 0, 50, tzinfo=timezone.utc)
    engine.add_data(_quotes(instrument, start, minutes=23 * 60 + 20))

    config = VaultRebalanceConfig(
        broker="ftmo", venue="MT5", tickers=("ES",), warmup_min_bars=0,
        poll_interval_secs=60, quote_window_timeout_secs=90, sizing_basis_usd=0.0,
        startup_grace_secs=0, min_rebalance_lots=0.01, min_rebalance_notional_usd=50.0,
    )
    engine.add_strategy(_RolloverStrategy(config=config))
    engine.run()

    positions = engine.cache.positions(instrument_id=instrument.id)
    assert positions, "a position should have been opened at the reopen"
    # entered long ~10 at reopen, then the negative-carry exit flattened it before rollover
    net = float(engine.portfolio.net_position(instrument.id))
    assert net == pytest.approx(0.0, abs=0.001), f"negative-carry leg should end flat, got {net}"
    # the opened position reached ~10 lots long at its peak
    peak = max(float(p.peak_qty) for p in positions)
    assert peak == pytest.approx(10.0, abs=0.5), f"entry should have been ~10 lots, peak={peak}"

    engine.dispose()
