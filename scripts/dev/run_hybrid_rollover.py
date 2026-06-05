r"""Run the HYBRID rollover lane through the existing WP-3 NautilusPnLEngine.

No bespoke strategy/harness — this just (1) curates a windowed-tick catalog
(1-min bars all day + quote ticks only around the rollover) via
``ingest_mt5_intraday_windowed`` and (2) runs ``NautilusPnLEngine`` with the new
``ROLLOVER_FLATTEN_REENTER`` window policy + ``LIMIT_AT_TOUCH`` execution, reusing
the engine's validated passive-limit anchoring and fill diagnostics.

    .\.venv\Scripts\python.exe scripts\dev\run_hybrid_rollover.py --rollover-utc 21:00 --half-width 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.ingest import ingest_mt5_intraday_windowed
from research.portfolio.pnl.nautilus_engine import (
    CrossAfterPolicy,
    ExecutionPolicy,
    ExecutionWindowPolicy,
    NautilusPnLEngine,
)

TICKER = "NDX"
FULL_TICKS = 54_204_267


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollover-utc", default="21:00", help="rollover time-of-day HH:MM (UTC)")
    ap.add_argument("--half-width", type=int, default=20, help="window half-width (minutes)")
    args = ap.parse_args()
    hh, mm = (int(x) for x in args.rollover_utc.split(":"))
    center, center_min = dt.time(hh, mm), hh * 60 + mm

    print("=" * 70)
    print(f"  HYBRID ROLLOVER LANE (NautilusPnLEngine) — NDX  "
          f"(rollover {args.rollover_utc} UTC ± {args.half_width}m)")
    print("=" * 70)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="nx_hybrid_") as td:
        cat_dir = Path(td) / "catalog"
        catalog = get_catalog(cat_dir)

        # (1) windowed-tick ingest: full bars + quotes only in the rollover window
        t = time.perf_counter()
        res = ingest_mt5_intraday_windowed(
            TICKER, catalog, rollover=center, half_width_minutes=args.half_width, tz="UTC")
        t_ingest = time.perf_counter() - t
        print(f"\n  ingest: bars={res.bars_written:,}  window_quotes={res.quotes_written:,}  "
              f"({t_ingest:.1f}s)  = {100*res.quotes_written/FULL_TICKS:.2f}% of all ticks "
              f"({FULL_TICKS/max(res.quotes_written,1):.0f}x fewer)")

        # (2) constant-long daily target; engine acts only on dates that have bars
        positions_df = pd.DataFrame({
            "ticker": TICKER,
            "datetime": pd.date_range("2026-01-01", "2026-07-01", freq="D"),
            "position_fraction": 1.0,
        })

        engine = NautilusPnLEngine(
            window_policy=ExecutionWindowPolicy.ROLLOVER_FLATTEN_REENTER,
            execution_policy=ExecutionPolicy.LIMIT_AT_TOUCH,
            cross_after=CrossAfterPolicy(session_fraction=1.0),  # pure passive (no cross)
            catalog_path=str(cat_dir),
            rollover_minute=center_min,
            rollover_half_width_min=args.half_width,
        )

        t = time.perf_counter()
        result = engine.run_with_diagnostics(positions_df, pd.DataFrame())
        t_run = time.perf_counter() - t

        fd = result.fill_diagnostics
        maker = sum(1 for d in fd if d.liquidity_side == "MAKER")
        taker = sum(1 for d in fd if d.liquidity_side == "TAKER")
        signed = sum(d.liquidity_signed_spread for d in fd)
        n_pos = 0 if result.positions_report is None else len(result.positions_report)

        print(f"\n  run_with_diagnostics: {t_run:.2f}s")
        print(f"  return bars: {len(result.returns):,}  positions(round-trips): {n_pos:,}  "
              f"entry_rejects(missed passive fills): {result.entry_rejects}")
        print(f"  FILLS: total={len(fd)}  maker={maker}  taker={taker}")
        print(f"  net liquidity-signed half-spread = {signed:,.2f} price-units "
              f"(>0 = spread captured by resting passive)")
        if not result.returns.empty:
            r = result.returns
            print(f"  returns: mean={r.mean():.2e}  std={r.std():.2e}  "
                  f"cumulative={r.sum():.4f}  (finite={bool(r.notna().all())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
