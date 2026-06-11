r"""Extract the prop-firm portfolio's per-ticker daily position_fraction over the
OOS Test window, plus the pipeline's own (futures-executed) strategy returns and
the daily futures candles. Outputs CSVs for the feed-substitution P&L study.

Run (cold cache → populates feature cache for ES/NQ/GC on first run):
    .\.venv\Scripts\python.exe -m research.feed_comparison.portfolio_extract_positions
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
    filter_ensemble_dirs_for_portfolio_tickers,
)
from research.portfolio.pipelines.portfolio_test import (  # noqa: E402
    run_single_phase_for_prop_firm,
)
from lib.core.enums import Ticker  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Build the prop-equivalent config manually: load_prop_firm_portfolio_research_config()
    # is currently broken (replace(oos, test_end=...) — that window has no test_end field).
    base = load_config()
    prop_tickers = [Ticker.ES, Ticker.NQ, Ticker.GC]
    cfg = replace(
        base,
        tickers=prop_tickers,
        ensemble_dirs=filter_ensemble_dirs_for_portfolio_tickers(
            base.ensemble_dirs, prop_tickers
        ),
        target_volatility=0.20,
        max_position_pct=2.5,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
        ensemble_vault_refit=False,  # frozen vault fits (prop live behavior, faster)
        futures_sim=replace(base.futures_sim, enabled=False),
        prop_firm_report=replace(base.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(base.feature_vault_correlation, enabled=False),
    )
    print(f"Tickers   : {[t.name for t in cfg.tickers]}")
    print(f"Ensembles : {len(cfg.ensemble_dirs)}")
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
