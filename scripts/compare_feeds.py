r"""Similarity gate: compare two captured feed snapshots (e.g. futures vs CFD).

Reads the per-phase combined + per-ensemble snapshots written by
``scripts/capture_baselines.py`` for two feeds and reports, per phase:

  * combined portfolio metric deltas (Sharpe / total return / max drawdown / Calmar)
  * combined daily-return-series correlation on the overlapping dates
  * per-ensemble Sharpe side-by-side + delta

This is the "double check the results are still similar" check for the
futures->CFD migration. CFD is a deliberate numbers change (different price feed
+ no additive back-adjustment distortion), so this is a *similarity* report, not a
byte-identical gate (that one is tests/parity, pinned to futures).

Run:
    .\.venv\Scripts\python.exe scripts\compare_feeds.py                       # futures vs cfd
    .\.venv\Scripts\python.exe scripts\compare_feeds.py --a futures --b cfd --phases test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SNAP = _ROOT / "tests" / "parity" / "snapshots"
_DIR_BY_FEED = {
    "futures": _SNAP / "baseline_futures_vectorized",
    "cfd": _SNAP / "cfd_vectorized",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _returns(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_parquet(path)
    s = df[df.columns[0]]
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def _phase_report(a_dir: Path, b_dir: Path, a: str, b: str, phase: str) -> list[str]:
    lines: list[str] = [f"\n## Phase: {phase}\n"]
    am = _load_json(a_dir / f"portfolio__{phase}__combined_metrics.json")
    bm = _load_json(b_dir / f"portfolio__{phase}__combined_metrics.json")
    if am and bm:
        lines.append(f"| metric | {a} | {b} | delta |")
        lines.append("|---|---:|---:|---:|")
        for k in ("strategy.sharpe", "strategy.total_return", "strategy.max_drawdown",
                  "strategy.calmar", "strategy.sortino"):
            av, bv = am.get(k, float("nan")), bm.get(k, float("nan"))
            lines.append(f"| {k} | {av:.4f} | {bv:.4f} | {bv - av:+.4f} |")

    ar = _returns(a_dir / f"portfolio__{phase}__combined_strategy_returns.parquet")
    br = _returns(b_dir / f"portfolio__{phase}__combined_strategy_returns.parquet")
    idx = ar.index.intersection(br.index)
    if len(idx) > 2:
        corr = float(ar.reindex(idx).corr(br.reindex(idx)))
        lines.append(
            f"\n**Combined daily-return correlation ({a} vs {b})**: "
            f"`{corr:.4f}` over {len(idx)} overlapping days "
            f"({a} n={len(ar)}, {b} n={len(br)}).\n"
        )

    ae = _load_json(a_dir / f"portfolio__{phase}__ensemble_metrics.json")
    be = _load_json(b_dir / f"portfolio__{phase}__ensemble_metrics.json")
    if ae and be:
        lines.append(f"| ensemble | {a} Sharpe | {b} Sharpe | delta |")
        lines.append("|---|---:|---:|---:|")
        for ens in sorted(set(ae) | set(be)):
            av = ae.get(ens, {}).get("sharpe", float("nan"))
            bv = be.get(ens, {}).get("sharpe", float("nan"))
            lines.append(f"| {ens} | {av:.3f} | {bv:.3f} | {bv - av:+.3f} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="futures", choices=list(_DIR_BY_FEED))
    p.add_argument("--b", default="cfd", choices=list(_DIR_BY_FEED))
    p.add_argument("--phases", nargs="*", default=["validation", "test"])
    p.add_argument("--out", type=Path, default=_SNAP / "feed_comparison.md")
    args = p.parse_args()

    a_dir, b_dir = _DIR_BY_FEED[args.a], _DIR_BY_FEED[args.b]
    lines = [f"# Feed comparison: {args.a} vs {args.b}",
             f"\nSnapshots: `{a_dir.name}` vs `{b_dir.name}`. "
             "Similarity gate for the futures->CFD migration (deliberate numbers change).\n"]
    for phase in args.phases:
        if not (a_dir / f"portfolio__{phase}__combined_metrics.json").exists():
            lines.append(f"\n## Phase: {phase}\n_(no {args.a} snapshot)_\n")
            continue
        lines += _phase_report(a_dir, b_dir, args.a, args.b, phase)

    report = "\n".join(lines)
    print(report)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
