r"""Benchmark the Nautilus BacktestEngine throughput on the NDX intraday catalog.

Mirrors research.portfolio.pnl.nautilus_engine._run_backtest (same venue / fill
model / strategy) but instruments the timing so we can separate:
    - data LOAD  (catalog read)         one-time-ish per run
    - INGEST     (raw MT5 -> catalog)   one-time (persists)
    - engine.RUN (the actual backtest)  the repeatable compute cost

and report events/sec for the bar-driven path (1-min) vs the quote-driven path
(ticks, capped budgets to extrapolate to the full ~54M tick set).

Run:
    .\.venv\Scripts\python.exe scripts\dev\bench_nautilus_backtest.py
    ... --tick-budgets 1000000,4000000
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.ingest import _resolve_instrument, ingest_mt5_intraday
from data_platform.nautilus.instruments import to_nautilus_instrument
from execution.position_sizer import ContractSpec, PositionSizer, RoundingMethod
from research.portfolio.pnl.nautilus_engine import (
    CrossAfterPolicy,
    ExecutionPolicy,
    ExecutionWindowPolicy,
    TargetRebalanceStrategy,
    _default_fill_model,
    _session_close_ns,
    _session_open_ns,
)

TICKER = "NDX"
CAPITAL = 1_000_000.0


def _targets_constant_long(bars) -> dict[date, float]:
    dates = {pd.Timestamp(b.ts_event, tz="UTC").date() for b in bars}
    return {d: 1.0 for d in dates}


def run_case(label: str, *, execution_policy: ExecutionPolicy, catalog, max_ticks):
    needs_quotes = execution_policy in (
        ExecutionPolicy.LIMIT_AT_TOUCH, ExecutionPolicy.LIMIT_IMPROVE,
    )
    dp_inst = _resolve_instrument(TICKER)
    nt_inst = to_nautilus_instrument(dp_inst)
    bar_type = BarType.from_str(f"{nt_inst.id}-1-MINUTE-LAST-EXTERNAL")

    # --- ensure data present (ingest into this catalog if empty) ---
    t = time.perf_counter()
    bars = catalog.bars(bar_types=[str(bar_type)])
    quotes = catalog.quote_ticks(instrument_ids=[str(nt_inst.id)]) if needs_quotes else []
    t_load0 = time.perf_counter() - t

    t_ingest = 0.0
    if not bars or (needs_quotes and not quotes):
        t = time.perf_counter()
        ingest_mt5_intraday(TICKER, catalog, max_ticks=max_ticks if needs_quotes else 0)
        t_ingest = time.perf_counter() - t
        t = time.perf_counter()
        bars = catalog.bars(bar_types=[str(bar_type)])
        quotes = catalog.quote_ticks(instrument_ids=[str(nt_inst.id)]) if needs_quotes else []
        t_load0 += time.perf_counter() - t

    n_bars, n_quotes = len(bars), len(quotes)
    targets = _targets_constant_long(bars)

    # --- build engine (mirror _run_backtest) ---
    t = time.perf_counter()
    venue = nt_inst.id.venue
    currency = nt_inst.quote_currency
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id="BENCH-001", logging=LoggingConfig(bypass_logging=True)))
    engine.add_venue(venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     base_currency=currency, starting_balances=[Money(CAPITAL, currency)],
                     fill_model=_default_fill_model(42))
    engine.add_instrument(nt_inst)
    engine.add_data(bars)
    if needs_quotes and quotes:
        engine.add_data(quotes)
    sizer = PositionSizer(capital=CAPITAL, contract_specs={TICKER: ContractSpec(
        ticker=TICKER, price=float(bars[0].close), multiplier=float(nt_inst.multiplier),
        fx_rate=1.0, min_tick=float(nt_inst.price_increment))}, rounding_method=RoundingMethod.ROUND)
    strategy = TargetRebalanceStrategy(
        instrument=nt_inst, bar_type=bar_type, targets_by_date=targets, sizer=sizer, ticker=TICKER,
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE, execution_policy=execution_policy,
        session_close_ns=_session_close_ns(bars), session_open_ns=_session_open_ns(bars),
        improve_ticks=1, cross_after=CrossAfterPolicy(), subscribe_quotes=needs_quotes)
    engine.add_strategy(strategy)
    t_build = time.perf_counter() - t

    # --- run ---
    t = time.perf_counter()
    engine.run()
    t_run = time.perf_counter() - t
    n_orders = len(engine.cache.orders())
    n_fills = sum(len(o.events) for o in engine.cache.orders())  # rough
    engine.dispose()

    events = n_bars + n_quotes
    eps = events / t_run if t_run else float("nan")
    print(f"\n### {label}")
    print(f"  bars={n_bars:,}  quotes={n_quotes:,}  events={events:,}  orders={n_orders:,}")
    print(f"  load={t_load0:6.2f}s  ingest={t_ingest:7.2f}s  build={t_build:6.2f}s  RUN={t_run:6.2f}s")
    print(f"  >>> throughput = {eps:,.0f} events/sec  ({events:,} events / {t_run:.2f}s run)")
    return {"label": label, "bars": n_bars, "quotes": n_quotes, "events": events,
            "t_run": t_run, "eps": eps, "t_ingest": t_ingest, "t_load": t_load0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-budgets", default="1000000,4000000",
                    help="comma-separated quote-tick caps for the LIMIT runs")
    args = ap.parse_args()
    budgets = [int(x) for x in args.tick_budgets.split(",") if x.strip()]

    rows = []
    # Case 1: bars-only (MARKET) against the default catalog (bars already present).
    print("=" * 70)
    print("  NAUTILUS BACKTEST THROUGHPUT BENCHMARK — NDX intraday")
    print("=" * 70)
    rows.append(run_case("1-min bars only (MARKET_ON_OPEN)",
                         execution_policy=ExecutionPolicy.MARKET_ON_OPEN,
                         catalog=get_catalog(None), max_ticks=0))

    # Case 2..n: bars + capped quote ticks (LIMIT) — fresh temp catalog per budget
    # so each tick volume is isolated (avoids the default catalog caching quotes).
    for budget in budgets:
        with tempfile.TemporaryDirectory(prefix="nx_bench_") as td:
            cat = get_catalog(Path(td) / "catalog")
            rows.append(run_case(
                f"1-min bars + {budget:,} quote ticks (LIMIT_AT_TOUCH)",
                execution_policy=ExecutionPolicy.LIMIT_AT_TOUCH, catalog=cat, max_ticks=budget))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'case':<48}{'events':>12}{'run(s)':>9}{'evt/s':>12}")
    for r in rows:
        print(f"  {r['label']:<48}{r['events']:>12,}{r['t_run']:>9.2f}{r['eps']:>12,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
