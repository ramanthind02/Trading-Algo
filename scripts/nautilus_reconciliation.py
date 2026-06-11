r"""Frictionless reconciliation: vectorized vs Nautilus INTRADAY_OPEN_TO_CLOSE.

Both lanes should capture the same open→close return on each day (flat overnight).
If they don't match, there is a bug in the Nautilus P&L calculation.

Runs on a SINGLE instrument (NDX by default) to isolate per-instrument behaviour
from multi-ticker aggregation effects.

Run:  .\.venv\Scripts\python.exe scripts\nautilus_reconciliation.py [--ticker NDX]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import research.feed_overlay.feed_overlay_experiment as fox
from dataclasses import replace
from datetime import datetime

from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.portfolio_impl.portfolio_tester import aggregate_intraday_returns_to_daily
from lib.core import research_feed
from research.portfolio.pnl import make_pnl_engine
from research.portfolio.pnl.nautilus_engine import ExecutionWindowPolicy
from research.portfolio.pipelines.portfolio_test import (
    _enable_cache,
    _evaluate_phase,
    _group_ensembles_by_timeframe,
    run_portfolio_research_cache_preflight,
)
from tests.parity._metrics import headline_metrics, normalize_returns_series


def _build_cfg():
    # Use the full universe so baseline returns (ES benchmark) don't fail.
    # We filter positions to the target ticker after _evaluate_phase.
    fox.TRAIN_START = datetime(2008, 1, 1)
    cfg = fox.load_config()
    ensemble_dirs = fox.filter_ensemble_dirs_for_portfolio_tickers(
        fox._discover_ensemble_dirs(
            allowed_timeframes=(fox.TimeFrame.D, fox.TimeFrame.M, fox.TimeFrame.W),
            vault_discovery_dirnames=fox.vault_discovery_dirnames_for_profile("prop"),
        ),
        list(fox.UNIVERSE),
    )
    wlk = {k: v for k, v in cfg.weight_layer_kwargs.items() if k != "hierarchy_spec"}
    cfg = replace(
        cfg,
        tickers=list(fox.UNIVERSE),
        ensemble_dirs=ensemble_dirs,
        weight_layer_kwargs=wlk,
        train_window=replace(cfg.train_window, start=fox.TRAIN_START),
        start=fox.TRAIN_START,
        data_feed="cfd",
        realistic_phases=(),
        export_per_timeframe_tearsheets=False,
        export_per_ensemble_tearsheets=False,
        prop_firm_report=replace(cfg.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(cfg.feature_vault_correlation, enabled=False),
    )
    return fox.with_rebuilt_weight_layer(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NQ")
    args = ap.parse_args()

    ticker = args.ticker
    print(f"\nReconciliation: vectorized vs Nautilus frictionless — ticker={ticker}")
    print("=" * 70)

    cfg = _build_cfg()
    research_feed.set_research_feed("cfd")
    run_portfolio_research_cache_preflight(cfg)

    named = [
        (n, _enable_cache(load_ensemble_from_vault(
            p, refit=cfg.ensemble_vault_refit, target_volatility=cfg.target_volatility,
            exclude_feature_stems_by_ensemble=getattr(cfg, "exclude_feature_stems_by_ensemble", None)), True))
        for n, p in cfg.ensemble_dirs.items()
    ]
    grouped = _group_ensembles_by_timeframe(named)
    tfs = sorted(grouped.keys())

    res, _ = _evaluate_phase(
        phase_title="Test", output_dir_name="test",
        fit_start=pd.Timestamp(cfg.train_window.start),
        fit_end=pd.Timestamp(cfg.validation_window.end),
        test_start=pd.Timestamp(cfg.test_window.start),
        test_end=pd.Timestamp(cfg.test_window.end),
        config=cfg, grouped_ensembles=grouped, unique_timeframes=tfs,
        emit_tearsheets=False, run_purpose="metrics_only", collect_strategy_returns=False,
    )
    all_pos = res.combined_positions
    candles = res.daily_test_candles
    # Filter to the single ticker of interest
    pos = all_pos[all_pos["ticker"].map(str) == ticker].copy()
    print(f"positions: {len(pos)} rows, ticker={ticker}, "
          f"{pd.to_datetime(pos['datetime']).min().date()}..{pd.to_datetime(pos['datetime']).max().date()}")

    # ── vectorized ────────────────────────────────────────────────────────────
    vec_engine = make_pnl_engine("vectorized")
    r_vec = aggregate_intraday_returns_to_daily(
        vec_engine.returns_from_positions(pos, candles)
    )
    r_vec = r_vec[r_vec != 0]  # drop flat days

    # ── Nautilus frictionless INTRADAY_OPEN_TO_CLOSE ──────────────────────────
    nt_engine = make_pnl_engine(
        "nautilus",
        multi_ticker=False,   # single instrument
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE,
        measure_spread=False,  # no spread → frictionless
    )
    r_nt = aggregate_intraday_returns_to_daily(
        nt_engine.returns_from_positions(pos, candles)
    )
    r_nt = r_nt[r_nt != 0]

    # ── alignment ─────────────────────────────────────────────────────────────
    common = r_vec.index.intersection(r_nt.index)
    v = r_vec.loc[common]
    n = r_nt.loc[common]
    print(f"\nCommon days: {len(common)}  (vec={len(r_vec)}, nautilus={len(r_nt)})")

    corr = float(v.corr(n))
    vol_ratio = float(n.std() / v.std())
    ret_ratio = float(n.sum() / v.sum())
    mean_abs_diff = float((v - n).abs().mean())

    print(f"\n{'Metric':30s}  {'vectorized':>12s}  {'nautilus':>12s}")
    print(f"{'Daily std (ann %)':30s}  {v.std() * np.sqrt(252) * 100:>12.3f}  {n.std() * np.sqrt(252) * 100:>12.3f}")
    print(f"{'Total return':30s}  {np.exp(v.sum()) - 1:>12.4f}  {np.exp(n.sum()) - 1:>12.4f}")
    m_v = headline_metrics(normalize_returns_series(v))
    m_n = headline_metrics(normalize_returns_series(n))
    print(f"{'Sharpe':30s}  {m_v['sharpe']:>12.3f}  {m_n['sharpe']:>12.3f}")
    print(f"\n{'Correlation':30s}  {corr:>12.4f}")
    print(f"{'Vol ratio (nt/vec)':30s}  {vol_ratio:>12.4f}")
    print(f"{'Return ratio (nt/vec)':30s}  {ret_ratio:>12.4f}")
    print(f"{'Mean |daily diff|':30s}  {mean_abs_diff:>12.6f}")

    status = "PASS" if corr > 0.99 and abs(vol_ratio - 1.0) < 0.02 else "FAIL"
    print(f"\n{status}: lanes diverge (corr={corr:.4f}, vol_ratio={vol_ratio:.4f})")

    # Day-by-day table: first 20 common days
    compare = pd.DataFrame({"vec": v, "nt": n, "diff": v - n, "ratio": n / v})
    print("\nFirst 20 common days (non-zero returns):")
    print(compare.head(20).to_string(float_format=lambda x: f"{x:.6f}"))

    # Largest divergence days
    print("\nTop 10 largest |diff| days:")
    top10 = compare.reindex(compare["diff"].abs().sort_values(ascending=False).index).head(10)
    print(top10.to_string(float_format=lambda x: f"{x:.6f}"))

    # Distribution of ratio
    valid = compare["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\nRatio (nt/vec) stats: mean={valid.mean():.4f} median={valid.median():.4f} "
          f"p25={valid.quantile(0.25):.4f} p75={valid.quantile(0.75):.4f}")

    # Magnitude check: what dollar return does each lane imply?
    capital = 1_000_000.0
    print(f"\nImplied dollar P&L:")
    print(f"  vectorized : ${(np.exp(v.sum()) - 1) * capital:,.0f}")
    print(f"  nautilus   : ${(np.exp(n.sum()) - 1) * capital:,.0f}")

    # Check if the positions frame has reasonable values
    pos_fracs = pos["position_fraction"].abs()
    print(f"\nPosition fraction (NQ) stats: mean={pos_fracs.mean():.4f} max={pos_fracs.max():.4f} "
          f"non-zero={( pos_fracs > 0).sum()}/{len(pos_fracs)}")


if __name__ == "__main__":
    main()
