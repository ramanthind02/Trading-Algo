import shutil
import pandas as pd
from pathlib import Path
from research.portfolio.pnl.experiments.spread_vs_market import (
    build_sliced_catalog, constant_long_targets, _SYMBOL)
from research.portfolio.pnl.nautilus_engine import (
    NautilusPnLEngine, ExecutionPolicy, ExecutionWindowPolicy, CrossAfterPolicy)

start = pd.Timestamp("2026-02-23", tz="UTC")
end = pd.Timestamp("2026-03-06", tz="UTC")  # ~2 weeks
root = Path("research/portfolio/pnl/experiments/results/_dbg")
shutil.rmtree(root, ignore_errors=True)

for label, pol, ticks in [
    ("MARKET", ExecutionPolicy.MARKET_ON_OPEN, 0),
    ("AT_TOUCH", ExecutionPolicy.LIMIT_AT_TOUCH, 0),
    ("IMPROVE1", ExecutionPolicy.LIMIT_IMPROVE, 1),
]:
    cat = str(root / label)
    bars_df, _ = build_sliced_catalog(_SYMBOL, start, end, cat)
    targets = constant_long_targets(bars_df, 1.0)
    eng = NautilusPnLEngine(
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE,
        execution_policy=pol, improve_ticks=ticks,
        cross_after=CrossAfterPolicy(session_fraction=1.0),
        measure_spread=True, catalog_path=cat, max_ticks=None)
    res = eng.run_with_diagnostics(targets, pd.DataFrame())
    ent = [d for d in res.fill_diagnostics if d.is_entry]
    mk = sum(1 for d in ent if d.liquidity_side=="MAKER")
    tk = sum(1 for d in ent if d.liquidity_side=="TAKER")
    avg_ls = sum(d.liquidity_signed_spread for d in ent)/len(ent) if ent else 0.0
    print(f"{label:9s} sessions={len(targets)} fills={len(ent)} rej={res.entry_rejects} MAKER={mk} TAKER={tk} avg_liqsigned={avg_ls:+.4f} ret={res.returns.sum():+.5f}")
