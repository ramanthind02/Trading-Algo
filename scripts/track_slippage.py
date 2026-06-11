r"""CLI: capture + summarise live execution slippage from MT5 deal history.

Reads magic-tagged fills for a broker over a window, computes per-fill slippage vs the
live touch (ask for a buy, bid for a sell) and vs mid, appends to
``data/broker_cache/<broker>/slippage/slippage.csv`` (idempotent by deal ticket), and
prints the new rows + a summary.

Run when the vault node is **STOPPED** (or pass ``--allow-live`` to override). A live
node holds the same MT5 terminal and heavy tick queries here can stall it — the guard
refuses by default. Deal history is persistent, so capturing after you stop the node
loses nothing.

Examples (PowerShell, repo root)::

    .\.venv\Scripts\python.exe scripts\track_slippage.py --broker ftmo --days 1 --summary
    .\.venv\Scripts\python.exe scripts\track_slippage.py --broker darwinex --since "2026-06-08 23:30"
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from lib.core.repo_bootstrap import ensure_project_root_on_path
except ImportError:  # pragma: no cover
    def ensure_project_root_on_path() -> None:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

ensure_project_root_on_path()

from lib.core.runtime_bootstrap import bootstrap_runtime  # noqa: E402

bootstrap_runtime()

from deployment.live.monitoring import live_state, slippage  # noqa: E402

# host_utc - 4h ≈ broker EEST wall-clock (the data-client offset; host clock is unreliable)
_BROKER_OFFSET_H = 4


def _broker_now_approx() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=_BROKER_OFFSET_H)


def _node_live(broker: str) -> float | None:
    """Snapshot age in seconds if a node is publishing for ``broker``, else None."""
    snap = live_state.read_snapshot(live_state.snapshot_path(live_state.live_state_dir(broker)))
    if not snap:
        return None
    try:
        ts = datetime.fromisoformat(snap["ts"])
        ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _fmt(s: dict) -> str:
    if not s or s.get("n", 0) == 0:
        return "n=0"
    return f"n={s['n']:>3}  mean={s['mean']:>7}  median={s['median']:>7}  min={s['min']:>7}  max={s['max']:>7}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture + summarise live execution slippage.")
    ap.add_argument("--broker", required=True, help="ftmo | darwinex | fundednext")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--since", help='broker wall-clock start "YYYY-MM-DD HH:MM" (default: --days)')
    g.add_argument("--days", type=float, default=1.0, help="look back this many days (default 1)")
    ap.add_argument("--summary", action="store_true", help="print aggregate stats after capture")
    ap.add_argument("--allow-live", action="store_true",
                    help="run even if a node is live (risks stalling it — avoid during a rollover window)")
    args = ap.parse_args()

    age = _node_live(args.broker)
    if age is not None and age < 60 and not args.allow_live:
        sys.exit(
            f"ABORT: a live node is publishing for {args.broker!r} (snapshot {age:.0f}s old). "
            "Heavy MT5 reads can STALL it. Stop the node first, or pass --allow-live to override."
        )

    if args.since:
        since = datetime.fromisoformat(args.since)
    else:
        since = _broker_now_approx() - timedelta(days=args.days)
    print(f"Reading {args.broker} fills since broker {since:%Y-%m-%d %H:%M} ...")

    rows = slippage.read_fills(args.broker, since)
    added = slippage.append_rows(args.broker, rows)
    print(f"fills found={len(rows)}  new appended={added}  -> {slippage.slippage_csv_path(args.broker)}")
    print(f"\n  {'broker_time':19} {'sym':9} {'side':4} {'vol':>6} {'fill_px':>11} {'spr_bps':>8} {'slip_touch':>10} {'slip_mid':>9} {'entry':>5}")
    for r in rows:
        st = "--" if r.slip_vs_touch_bps is None else f"{r.slip_vs_touch_bps:.2f}"
        sm = "--" if r.slip_vs_mid_bps is None else f"{r.slip_vs_mid_bps:.2f}"
        sp = "--" if r.spread_bps is None else f"{r.spread_bps:.2f}"
        bt = r.broker_time.replace("T", " ")[5:19]
        print(f"  {bt:19} {r.symbol:9} {r.side:4} {r.volume:6.2f} {r.fill_px:11.5f} {sp:>8} {st:>10} {sm:>9} {r.entry:>5}")

    if args.summary:
        s = slippage.summarize(slippage.load_rows(args.broker))
        print(f"\n=== {args.broker} slippage summary (all-time, from CSV) ===")
        print("OVERALL:")
        for m in ("spread_bps", "slip_vs_touch_bps", "slip_vs_mid_bps"):
            print(f"  {m:18} {_fmt(s['overall'][m])}")
        print("BY TICKER (slip_vs_mid_bps):")
        for c, st in s["by_canonical"].items():
            print(f"  {c:5} {_fmt(st['slip_vs_mid_bps'])}")


if __name__ == "__main__":
    main()
