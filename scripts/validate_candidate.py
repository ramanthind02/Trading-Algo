r"""Final-validation lane for a promotion candidate — run ONCE before promoting.

Drives the REAL live ``VaultRebalanceStrategy`` through a Nautilus
``BacktestEngine`` over historical data, generating the signal on-the-fly via the
real ``VaultForecastEngine`` bounded to the simulation clock. Signal AND execution
share one causal, event-driven clock, so the run is lookahead-free by construction
(no precomputed positions, no hand-applied shift) — the strongest validation we
have, and a research<->live consistency check (it runs the exact live code).

This is the EXPENSIVE lane (refits per decision + a one-time central-cache
preflight). Use it on the handful of candidates you actually promote; the fast
vectorized lane stays the workhorse for research sweeps.

Usage (Windows, repo root):
  .\.venv\Scripts\python.exe scripts\validate_candidate.py ^
      --vault-root vault --start 2024-01-01 --end 2025-01-01 ^
      --tickers ES NQ GC CL SI
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.validation import ValidationConfig, ValidationResult, run_validation_backtest

# Final-validation artifact root (repo-relative).
_FINAL_VALIDATION_DIR = _ROOT / "feature_research" / "shared_results" / "final_validation"

# Registry DB path override (None → canonical data/registry.db).  Tests monkeypatch this.
_REGISTRY_DB_PATH: Path | None = None


def _persist_run(
    result: ValidationResult,
    config: ValidationConfig,
    run_id: str,
    run_dir: Path,
    *,
    db_path: Path | None = None,
) -> None:
    """Write artifacts to run_dir and register a final_validation row in the registry.

    Best-effort: never raises; failures are printed to stderr.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        (run_dir / "metrics.json").write_text(
            json.dumps(result.metrics, indent=2), encoding="utf-8"
        )
        # equity.parquet
        eq_df = result.equity_curve.to_frame("equity")
        eq_df.index.name = "datetime"
        eq_df.reset_index().to_parquet(run_dir / "equity.parquet", index=False)
        # fills.parquet + positions.parquet (may be empty DataFrames)
        if not result.fills_report.empty:
            result.fills_report.to_parquet(run_dir / "fills.parquet", index=False)
        if not result.positions_report.empty:
            result.positions_report.to_parquet(run_dir / "positions.parquet", index=False)

    except Exception as exc:
        print(f"[validate_candidate] artifact write failed: {exc}", file=sys.stderr)

    # Registry row (best-effort).
    try:
        from data_platform.registry import db as _db, writer as _writer

        actual_db = db_path or _REGISTRY_DB_PATH or _db.registry_path()
        now = datetime.now(timezone.utc).isoformat()

        # repo-relative path for the run dir
        try:
            reports_dir = run_dir.resolve().relative_to(_ROOT.resolve()).as_posix()
        except ValueError:
            reports_dir = run_dir.as_posix()

        run_record = _writer.RunRecord(
            run_id=run_id,
            kind="final_validation",
            status="completed",
            spec_id=None,
            reports_dir=reports_dir,
            headline_metrics_json=json.dumps(result.metrics),
            created_at=now,
            started_at=now,
            finished_at=now,
        )
        conn = _db.connect(actual_db)
        try:
            with _db.transaction(conn):
                _writer.upsert_run(conn, run_record)
        finally:
            conn.close()
    except Exception as exc:
        print(f"[validate_candidate] registry write failed: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault-root", default="vault")
    ap.add_argument("--start", required=True, help="UTC window start (e.g. 2024-01-01)")
    ap.add_argument("--end", required=True, help="UTC window end (exclusive)")
    ap.add_argument("--tickers", nargs="+", default=["ES", "NQ", "GC", "CL", "SI"])
    ap.add_argument("--starting-balance", type=float, default=1_000_000.0)
    ap.add_argument("--quote-stride-min", type=int, default=5)
    ap.add_argument("--broker", default=None, help="symbol mapping (default: Darwinex)")
    ap.add_argument(
        "--no-populate-cache",
        action="store_true",
        help="skip the central-cache preflight (use when it is already populated, e.g. live env)",
    )
    args = ap.parse_args()

    kwargs = dict(
        start=pd.Timestamp(args.start, tz="UTC"),
        end=pd.Timestamp(args.end, tz="UTC"),
        vault_root=args.vault_root,
        tickers=tuple(args.tickers),
        starting_balance=args.starting_balance,
        quote_stride_min=args.quote_stride_min,
        populate_central_cache=not args.no_populate_cache,
    )
    if args.broker:
        kwargs["broker"] = args.broker

    print(f"\nFinal-validation lane (lookahead-free, real live strategy) — vault={args.vault_root}")
    print(f"window {args.start}..{args.end}  tickers={args.tickers}")
    print("=" * 72)
    config = ValidationConfig(**kwargs)
    res = run_validation_backtest(config)

    m = res.metrics
    print(f"\nResolved tickers : {', '.join(res.resolved_tickers)}")
    print(f"Decisions (fills): {res.n_decisions}")
    print(f"Equity samples   : {len(res.equity_curve)}")
    print(f"\n{'Sharpe':16s} {m['sharpe']:.3f}")
    print(f"{'Ann return %':16s} {m['ann_return_pct']:.2f}")
    print(f"{'Ann vol %':16s} {m['ann_vol_pct']:.2f}")
    print(f"{'Total return':16s} {m['total_return']:+.4f}")
    print(f"{'Obs (days)':16s} {int(m['n_obs'])}")
    if res.n_decisions == 0:
        print(
            "\nNOTE: no fills — likely warmup not satisfied (central cache lacks enough "
            "history for a required ticker). Widen the window or populate the cache."
        )

    # Persist artifacts + registry row (best-effort; stdout output above is unchanged).
    run_id = str(uuid4())
    run_dir = _FINAL_VALIDATION_DIR / run_id
    _persist_run(res, config, run_id, run_dir)
    print(f"\nArtifacts written to: {run_dir}")
    print(f"Registry run_id     : {run_id}")


if __name__ == "__main__":
    main()
