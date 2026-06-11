r"""Isolate the (additive-series + rollover-overlay) effect with MATCHED windows.

Train window forced to 2008+ for BOTH feeds so history length is not a confound.
Runs the portfolio over train / validation / test for three models and reports
composite Sharpe (validation+test, and train+validation+test = concatenated daily
strategy returns):

  futures_frictionless : Norgate futures (additive back-adj) + vectorized      [OLD]
  cfd_frictionless     : Darwinex CFD (faithful %)        + vectorized          (isolates the feed)
  cfd_realistic        : Darwinex CFD                     + realistic Nautilus  [NEW: + rollover overlay]

Run:  .\.venv\Scripts\python.exe -m research.feed_overlay.feed_overlay_experiment
      .\.venv\Scripts\python.exe -m research.feed_overlay.feed_overlay_experiment --models futures_frictionless cfd_frictionless
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.portfolio.config import (
    load_config,
    _discover_ensemble_dirs,
    filter_ensemble_dirs_for_portfolio_tickers,
    vault_discovery_dirnames_for_profile,
    with_rebuilt_weight_layer,
)
from research.portfolio.pipelines.portfolio_test import (
    _enable_cache,
    _evaluate_phase,
    _group_ensembles_by_timeframe,
    run_portfolio_research_cache_preflight,
)
from ensemble.vault_manager import load_ensemble_from_vault
from lib.core import research_feed
from lib.core.enums import Ticker, TimeFrame
from tests.parity._metrics import headline_metrics, normalize_returns_series

OUT = _ROOT / "tests" / "parity" / "snapshots" / "feed_overlay_experiment"
TRAIN_START = datetime(2008, 1, 1)
# Portfolio universe for this comparison: ES/NQ/GC/CL + SI (silver). Adding SI
# pulls in the silver_mr/silver_trend sleeves (skipped in the default config).
UNIVERSE = (Ticker.ES, Ticker.NQ, Ticker.GC, Ticker.CL, Ticker.SI)
_PHASES = ("train", "validation", "test")
_MODELS = {
    "futures_frictionless": ("futures", False),
    "cfd_frictionless": ("cfd", False),
    "cfd_realistic": ("cfd", True),
}


def _phase_windows(cfg, phase: str):
    ts = pd.Timestamp
    if phase == "train":
        return ts(cfg.train_window.start), ts(cfg.train_window.end), ts(cfg.train_window.start), ts(cfg.train_window.end)
    if phase == "validation":
        return ts(cfg.train_window.start), ts(cfg.train_window.end), ts(cfg.validation_window.start), ts(cfg.validation_window.end)
    return ts(cfg.train_window.start), ts(cfg.validation_window.end), ts(cfg.test_window.start), ts(cfg.test_window.end)


def run_model(feed: str, realistic: bool) -> dict[str, pd.Series]:
    cfg = load_config()
    # Rebuild the universe to include SI (the default config filtered it out), so
    # the silver sleeves load. Re-discover + re-filter ensemble_dirs for the new set.
    ensemble_dirs = filter_ensemble_dirs_for_portfolio_tickers(
        _discover_ensemble_dirs(
            allowed_timeframes=(TimeFrame.D, TimeFrame.M, TimeFrame.W),
            vault_discovery_dirnames=vault_discovery_dirnames_for_profile("prop"),
        ),
        list(UNIVERSE),
    )
    # Drop the stale ES/NQ/GC/CL hierarchy_spec load_config baked in — otherwise
    # rebuild_weight_layer_kwargs early-returns (it skips rebuild when a spec is
    # already present) and the silver streams stay undeclared.
    wlk = {k: v for k, v in cfg.weight_layer_kwargs.items() if k != "hierarchy_spec"}
    cfg = replace(
        cfg,
        tickers=list(UNIVERSE),
        ensemble_dirs=ensemble_dirs,
        weight_layer_kwargs=wlk,
        train_window=replace(cfg.train_window, start=TRAIN_START),
        start=TRAIN_START,
        data_feed=feed,
        realistic_phases=_PHASES if realistic else (),
        export_per_timeframe_tearsheets=False,
        export_per_ensemble_tearsheets=False,
        prop_firm_report=replace(cfg.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(cfg.feature_vault_correlation, enabled=False),
    )
    # Rebuild the weight-layer hierarchy spec for the NEW universe so the silver
    # streams (silver_mr / silver_trend groups) are declared — else hierarchy_equal's
    # strict stream-coverage check rejects the produced SI streams.
    cfg = with_rebuilt_weight_layer(cfg)
    research_feed.set_research_feed(feed)
    run_portfolio_research_cache_preflight(cfg)
    named = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=getattr(cfg, "ensemble_vault_refit", True),
                    target_volatility=cfg.target_volatility,
                    exclude_feature_stems_by_ensemble=getattr(cfg, "exclude_feature_stems_by_ensemble", None),
                ),
                True,
            ),
        )
        for name, path in cfg.ensemble_dirs.items()
    ]
    grouped = _group_ensembles_by_timeframe(named)
    tfs = sorted(grouped.keys())
    rets: dict[str, pd.Series] = {}
    for phase in _PHASES:
        fs, fe, tsd, ted = _phase_windows(cfg, phase)
        res, _wl = _evaluate_phase(
            phase_title=phase.capitalize(), output_dir_name=phase,
            fit_start=fs, fit_end=fe, test_start=tsd, test_end=ted,
            config=cfg, grouped_ensembles=grouped, unique_timeframes=tfs,
            emit_tearsheets=False, run_purpose="metrics_only", collect_strategy_returns=False,
        )
        rets[phase] = normalize_returns_series(res.combined_strategy_returns)
    return rets


def composites(rets: dict[str, pd.Series]) -> dict[str, dict]:
    vt = pd.concat([rets["validation"], rets["test"]]).sort_index()
    tvt = pd.concat([rets["train"], rets["validation"], rets["test"]]).sort_index()
    return {
        "train": headline_metrics(rets["train"]),
        "validation": headline_metrics(rets["validation"]),
        "test": headline_metrics(rets["test"]),
        "validation+test": headline_metrics(vt),
        "train+validation+test": headline_metrics(tvt),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=list(_MODELS), choices=list(_MODELS))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"FEED + OVERLAY EXPERIMENT — matched train start {TRAIN_START.date()} for all models")
    print("=" * 78)
    results: dict[str, dict] = {}
    for label in args.models:
        feed, realistic = _MODELS[label]
        t0 = time.time()
        print(f"\n--- {label} (feed={feed}, realistic={realistic}) ---")
        comp = composites(run_model(feed, realistic))
        results[label] = comp
        print(f"  done in {(time.time()-t0)/60:.1f} min")
        for win in ("validation+test", "train+validation+test"):
            m = comp[win]
            print(f"    {win:24s} Sharpe={m['sharpe']:.3f}  ret={m['total_return']:+.4f}  maxDD={m['max_drawdown']:.4f}  n={int(m['n_obs'])}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    # Comparison table
    print("\n" + "=" * 78)
    print(f"{'model':24s} {'val+test Sharpe':>16s} {'train+val+test Sharpe':>22s}")
    print("-" * 78)
    for label in args.models:
        c = results[label]
        print(f"{label:24s} {c['validation+test']['sharpe']:>16.3f} {c['train+validation+test']['sharpe']:>22.3f}")
    print(f"\nWrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
