r"""Capture pre-migration baselines (Norgate futures feed + vectorized P&L).

This snapshots the **OLD model** that the CFD + Nautilus migration is checked
against (the similarity gate). For the default portfolio-research config it runs
the requested phases and records, per phase:

  * combined GlobalPortfolio daily strategy + baseline returns + headline metrics
  * per-vault-ensemble daily strategy returns (wide frame) + per-ensemble metrics

Outputs go under ``tests/parity/snapshots/baseline_futures_vectorized/``. Nothing
in the working tree is mutated except that snapshot dir (artifacts are redirected
to a temp ``output_root``).

Run (repo root):
    .\.venv\Scripts\python.exe scripts\capture_baselines.py                # validation + test
    .\.venv\Scripts\python.exe scripts\capture_baselines.py --phases test  # test only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.portfolio.config import load_config
from research.portfolio.pipelines.portfolio_test import (
    _enable_cache,
    _evaluate_phase,
    _group_ensembles_by_timeframe,
    run_portfolio_research_cache_preflight,
)
from ensemble.vault_manager import load_ensemble_from_vault
from lib.core import research_feed
from tests.parity._metrics import headline_metrics, normalize_returns_series

_OUT_BY_FEED = {
    "futures": _ROOT / "tests" / "parity" / "snapshots" / "baseline_futures_vectorized",
    "cfd": _ROOT / "tests" / "parity" / "snapshots" / "cfd_vectorized",
}
OUT: Path = _OUT_BY_FEED["futures"]  # set per --feed in main()

# Per-phase (fit_window, test_window) selectors over config windows. Mirrors
# research.portfolio.pipelines.portfolio_test.run_single_phase_for_prop_firm.
_PHASES = ("train", "validation", "test")


def _phase_windows(config, phase: str):
    train_s = pd.Timestamp(config.train_window.start)
    train_e = pd.Timestamp(config.train_window.end)
    val_s = pd.Timestamp(config.validation_window.start)
    val_e = pd.Timestamp(config.validation_window.end)
    test_s = pd.Timestamp(config.test_window.start)
    test_e = pd.Timestamp(config.test_window.end)
    if phase == "train":
        return train_s, train_e, train_s, train_e
    if phase == "validation":
        return train_s, train_e, val_s, val_e
    if phase == "test":
        return train_s, val_e, test_s, test_e
    raise ValueError(f"unknown phase {phase!r}")


def _save_combined(phase: str, result) -> dict:
    strat = normalize_returns_series(result.combined_strategy_returns, name="strategy_return")
    base = normalize_returns_series(result.combined_baseline_returns, name="baseline_return")
    strat.to_frame().to_parquet(OUT / f"portfolio__{phase}__combined_strategy_returns.parquet")
    base.to_frame().to_parquet(OUT / f"portfolio__{phase}__combined_baseline_returns.parquet")
    metrics = {f"strategy.{k}": v for k, v in headline_metrics(strat).items()}
    metrics.update({f"baseline.{k}": v for k, v in headline_metrics(base).items()})
    (OUT / f"portfolio__{phase}__combined_metrics.json").write_text(
        json.dumps({k: metrics[k] for k in sorted(metrics)}, indent=2), encoding="utf-8"
    )
    return metrics


def _save_per_ensemble(phase: str, result) -> dict:
    frame = result.strategy_returns_by_ensemble
    if frame is None or frame.empty:
        return {}
    frame = frame.sort_index()
    frame.to_parquet(OUT / f"portfolio__{phase}__ensemble_returns.parquet")
    per: dict[str, dict] = {}
    for col in frame.columns:
        per[str(col)] = headline_metrics(frame[col].dropna())
    (OUT / f"portfolio__{phase}__ensemble_metrics.json").write_text(
        json.dumps(per, indent=2, sort_keys=True), encoding="utf-8"
    )
    return per


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phases", nargs="*", default=["validation", "test"], choices=_PHASES)
    p.add_argument("--feed", choices=["futures", "cfd"], default="futures",
                   help="Data feed to capture under (default: futures = the OLD-model baseline).")
    args = p.parse_args()

    global OUT
    OUT = _OUT_BY_FEED[args.feed]
    research_feed.set_research_feed(args.feed)

    np.random.seed(1234)
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_config()
    tmp_root = Path(tempfile.mkdtemp(prefix="baseline_futures_vectorized_"))
    config = replace(
        config,
        output_root=tmp_root,
        # Baselines/feed-comparison are vectorized-vs-vectorized (the feed change is
        # the variable under test); the realistic Nautilus lane is a separate layer.
        realistic_phases=(),
        export_per_timeframe_tearsheets=False,
        export_per_ensemble_tearsheets=False,
        prop_firm_report=replace(config.prop_firm_report, enabled=False),
        feature_vault_correlation=replace(config.feature_vault_correlation, enabled=False),
    )

    print("=" * 64)
    print(f"BASELINE CAPTURE — {args.feed} feed + vectorized P&L")
    print(f"tickers     : {[t.name for t in config.tickers]}")
    print(f"pnl_engine  : {getattr(config, 'pnl_engine', 'vectorized')}")
    print(f"phases      : {args.phases}")
    print(f"snapshot dir: {OUT}")
    print("=" * 64)

    run_portfolio_research_cache_preflight(config)

    named_ensembles = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=getattr(config, "ensemble_vault_refit", True),
                    target_volatility=config.target_volatility,
                    exclude_feature_stems_by_ensemble=getattr(
                        config, "exclude_feature_stems_by_ensemble", None
                    ),
                ),
                True,
            ),
        )
        for name, path in config.ensemble_dirs.items()
    ]
    grouped = _group_ensembles_by_timeframe(named_ensembles)
    unique_timeframes = sorted(grouped.keys())
    print(f"Loaded {len(named_ensembles)} ensemble(s); timeframes={[t.name for t in unique_timeframes]}")

    manifest = {
        "feed": args.feed,
        "pnl_engine": getattr(config, "pnl_engine", "vectorized"),
        "tickers": [t.name for t in config.tickers],
        "phases": {},
    }

    for phase in args.phases:
        fit_s, fit_e, test_s, test_e = _phase_windows(config, phase)
        print(f"\n--- {phase}: fit {fit_s.date()}->{fit_e.date()} | score {test_s.date()}->{test_e.date()} ---")
        result, _wl = _evaluate_phase(
            phase_title=phase.capitalize(),
            output_dir_name=phase,
            fit_start=fit_s,
            fit_end=fit_e,
            test_start=test_s,
            test_end=test_e,
            config=config,
            grouped_ensembles=grouped,
            unique_timeframes=unique_timeframes,
            emit_tearsheets=False,
            run_purpose="metrics_only",
            collect_strategy_returns=True,
        )
        combined = _save_combined(phase, result)
        per_ens = _save_per_ensemble(phase, result)
        manifest["phases"][phase] = {
            "combined": combined,
            "ensembles": {k: v.get("sharpe", float("nan")) for k, v in per_ens.items()},
        }
        print(f"  combined strategy Sharpe={combined.get('strategy.sharpe'):.3f} "
              f"total_return={combined.get('strategy.total_return'):.4f}")
        for ens, m in sorted(per_ens.items()):
            print(f"    {ens:42s} Sharpe={m.get('sharpe', float('nan')):.3f}")

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDone. Baseline snapshots written to {OUT}")


if __name__ == "__main__":
    main()
