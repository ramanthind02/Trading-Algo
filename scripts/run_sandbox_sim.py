r"""Operator entrypoint: sandbox live-trading simulation (Darwinex signal, FTMO exec).

Runs the REAL live strategy through a Nautilus BacktestEngine over a recent window,
emulating live trading with markets closed: signals from the Darwinex cache, orders
on the FTMO venue with modeled spread + swap, the 16:05-ET decision clock, and the
per-symbol market-hours gate — all the same code that runs live.

Usage (PowerShell, repo root)::

    .\.venv\Scripts\python.exe -m scripts.run_sandbox_sim --days 20
    .\.venv\Scripts\python.exe -m scripts.run_sandbox_sim --start 2026-05-01 --end 2026-06-05 --exec-broker ftmo
    .\.venv\Scripts\python.exe -m scripts.run_sandbox_sim --days 20 --no-refresh   # reuse existing signal cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from lib.core.repo_bootstrap import ensure_project_root_on_path
except ImportError:  # pragma: no cover
    def ensure_project_root_on_path() -> None:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

ensure_project_root_on_path()

import pandas as pd  # noqa: E402

from deployment.live.sandbox_sim import SandboxSimConfig, run_sandbox_sim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Sandbox live-trading simulation (Darwinex signal / FTMO exec).")
    p.add_argument("--days", type=int, default=20, help="window length in calendar days (ignored if --start given)")
    p.add_argument("--start", default=None, help="UTC start (YYYY-MM-DD); overrides --days")
    p.add_argument("--end", default=None, help="UTC end (YYYY-MM-DD); defaults to today")
    p.add_argument("--vault-root", default="vault")
    p.add_argument("--exec-broker", default="ftmo", help="execution venue broker (symbols, costs)")
    p.add_argument("--signal-broker", default="darwinex", help="signal source-of-truth broker")
    p.add_argument("--tickers", default="ES,NQ,GC,CL,SI", help="comma-separated canonical tickers")
    p.add_argument("--no-refresh", action="store_true", help="reuse the existing Darwinex signal cache")
    args = p.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=args.days)

    config = SandboxSimConfig(
        start=start,
        end=end,
        vault_root=args.vault_root,
        signal_broker=args.signal_broker,
        exec_broker=args.exec_broker,
        tickers=tuple(t.strip() for t in args.tickers.split(",") if t.strip()),
        refresh_signal_cache=not args.no_refresh,
    )

    print(f"Sandbox sim: {start.date()}..{end.date()} | signal={config.signal_broker} "
          f"exec={config.exec_broker} tickers={list(config.tickers)}")
    print("Building Darwinex signal cache..." if config.refresh_signal_cache else "Reusing signal cache.")

    result = run_sandbox_sim(config)

    print("\n=== Sandbox simulation result ===")
    print(f"  resolved tickers : {list(result.resolved_tickers)}")
    print(f"  fills            : {result.n_fills}")
    if result.deferred_closed:
        print(f"  never traded     : {list(result.deferred_closed)} (market closed during every window)")
    print(f"  swap cost (total): ${result.swap_cost_total:,.2f}")
    if not result.gross_equity.empty:
        print(f"  gross equity     : {result.gross_equity.iloc[0]:,.0f} -> {result.gross_equity.iloc[-1]:,.0f}")
        print(f"  net equity       : {result.net_equity.iloc[0]:,.0f} -> {result.net_equity.iloc[-1]:,.0f} (after swap)")
    m = result.metrics
    print(f"  sharpe={m.get('sharpe'):.2f} ann_return={m.get('ann_return_pct'):.1f}% "
          f"ann_vol={m.get('ann_vol_pct'):.1f}% n_obs={int(m.get('n_obs', 0))}")


if __name__ == "__main__":
    main()
