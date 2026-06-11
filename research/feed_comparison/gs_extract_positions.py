r"""Extract the prop-firm portfolio's per-ticker daily position_fraction over the
OOS Test window — WITH SILVER (SI) in the universe so the silver sleeves load —
plus the pipeline's own (futures-executed) strategy returns and the daily futures
candles. Outputs CSVs to research/feed_comparison/_gs_out/ for the gold/silver feed study.

This is the SI-inclusive sibling of portfolio_extract_positions.py. The
base config discovery excludes the SI sleeves unless SI is in the ticker list, so
we discover ensemble dirs for the full universe explicitly.

Run (cold cache → populates feature cache for ES/NQ/GC/CL/SI on first run; the
harmless CacheManager `_resolved_source_revision` warnings appear but the daily
portfolio still produces positions):
    .\.venv\Scripts\python.exe -m research.feed_comparison.gs_extract_positions
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from research.portfolio.config import (  # noqa: E402
    load_config,
    _discover_ensemble_dirs,
    filter_ensemble_dirs_for_portfolio_tickers,
)
from lib.core.vault_paths import vault_discovery_dirnames_for_profile  # noqa: E402
from research.portfolio.pipelines.portfolio_test import (  # noqa: E402
    run_single_phase_for_prop_firm,
)
from lib.core.enums import Ticker, TimeFrame  # noqa: E402

OUT = Path(__file__).resolve().parent / "_gs_out"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    base = load_config()
    # Full prop universe INCLUDING SI so the silver_mr / silver_trend sleeves load.
    universe = [Ticker.ES, Ticker.NQ, Ticker.GC, Ticker.CL, Ticker.SI]

    ens_all = _discover_ensemble_dirs(
        allowed_timeframes=(TimeFrame.D, TimeFrame.M, TimeFrame.W),
        vault_discovery_dirnames=vault_discovery_dirnames_for_profile("prop"),
    )
    ens = filter_ensemble_dirs_for_portfolio_tickers(ens_all, universe)

    cfg = replace(
        base,
        tickers=universe,
        ensemble_dirs=ens,
        target_volatility=0.20,
        max_position_pct=2.5,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
        ensemble_vault_refit=False,
        futures_sim=replace(base.futures_sim, enabled=False),
        prop_firm_report=replace(base.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(base.feature_vault_correlation, enabled=False),
    )
    print(f"Tickers   : {[t.name for t in cfg.tickers]}")
    print(f"Ensembles : {len(cfg.ensemble_dirs)}")
    for name in sorted(cfg.ensemble_dirs):
        print(f"    {name} -> {cfg.ensemble_dirs[name]}")
    print(f"Test win  : {cfg.test_window.start.date()} -> {cfg.test_window.end.date()}")

    res = run_single_phase_for_prop_firm(
        cfg, "test", emit_tearsheets=False, run_purpose="metrics_only"
    )

    pos = res.combined_positions.copy()
    pos.to_csv(OUT / "positions_test.csv", index=False)
    print(f"\npositions_test.csv: {len(pos)} rows, "
          f"tickers={sorted(pos['ticker'].astype(str).unique())}, "
          f"{pd.to_datetime(pos['datetime']).min().date()}..{pd.to_datetime(pos['datetime']).max().date()}")

    sr = res.combined_strategy_returns.rename("strategy_return_futures")
    sr.to_csv(OUT / "strategy_returns_futures.csv")
    print(f"strategy_returns_futures.csv: {len(sr)} rows")

    res.daily_test_candles.to_csv(OUT / "daily_test_candles.csv", index=False)
    print(f"daily_test_candles.csv: {len(res.daily_test_candles)} rows, "
          f"cols={list(res.daily_test_candles.columns)}")


if __name__ == "__main__":
    main()
