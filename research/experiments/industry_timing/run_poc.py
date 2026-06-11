"""Phase-0 frictionless POC: Industry-Timing METHOD on OUR tradeable book.

Reports:
  - lookahead verification (two independent P&L derivations + a no-shift 'cheat' premium)
  - per-instrument standalone metrics (where does the method work?)
  - portfolio metrics (gross) + a timing-attribution benchmark (always-long vol-target)
  - FX as a negative control (long-only trend has no risk-premium premise on FX)

Run: .\.venv\Scripts\python.exe research\experiments\industry_timing\run_poc.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace
from pathlib import Path

from engine import Config, backtest, benchmark_buyhold, metrics

OUT = Path(__file__).resolve().parent / "outputs"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Our live tradeable book that has a long-only risk-premium premise + local data.
CORE = ["ES", "NQ", "YM", "RTY", "GC", "SI", "CL", "TLT"]
# FX: negative control (no positive drift premise for being long any currency).
FX = ["EU", "BP", "JY", "CD", "SF", "AUDNZD"]


def _fmt(m: dict) -> str:
    return (f"Sh={m['sharpe']:+.2f}  ann={m['ann_return']*100:+6.1f}%  "
            f"vol={m['ann_vol']*100:5.1f}%  MDD={m['mdd']*100:6.1f}%  "
            f"Cal={m['calmar']:+.2f}  Sor={m['sortino']:+.2f}  "
            f"hit={m['hit']*100:.0f}%  skew={m['skew']:+.2f}  n={m['n_days']}")


def verify_lookahead(tickers: list[str], cfg: Config) -> None:
    print("\n" + "=" * 90)
    print("LOOKAHEAD VERIFICATION")
    print("=" * 90)
    res = backtest(tickers, cfg)
    w_held = res["weights_held"]
    ret = res["ret"].reindex(w_held.index)
    # Method A: engine vectorized
    a = (w_held * ret).sum(axis=1)
    # Method B: explicit independent loop, w[t-1] . ret[t]
    w = res["weights"].reindex(w_held.index).to_numpy()
    rr = ret.to_numpy()
    b = np.zeros(len(w))
    for t in range(1, len(w)):
        b[t] = float(np.nansum(w[t - 1] * rr[t]))
    b = pd.Series(b, index=w_held.index)
    # row 0 differs by construction: the engine shifts weights BEFORE trimming the
    # warmup, so w_held[0] inherits a pre-trim decision weight while the explicit
    # loop seeds b[0]=0. Exclude that single boundary row.
    diff_all = (a - b).abs()
    diff = diff_all.iloc[1:].max()
    print(f"max |A - B| (excl. row0 trim boundary) over {len(a)-1} days = {diff:.3e}   (must be ~0)")
    print(f"   row0 boundary diff = {diff_all.iloc[0]:.3e} (warmup edge, expected)")
    assert diff < 1e-12, "Lookahead/alignment mismatch between the two derivations!"

    # 'cheat' premium: align weights with SAME-day return (peeks). Should be HIGHER.
    cheat = (res["weights"].reindex(ret.index) * ret).sum(axis=1)
    print(f"Sharpe  (correct, shift+1) : {metrics(a)['sharpe']:+.3f}")
    print(f"Sharpe  (CHEAT, no shift)  : {metrics(cheat)['sharpe']:+.3f}   "
          f"<- lookahead premium = {metrics(cheat)['sharpe'] - metrics(a)['sharpe']:+.3f}")
    print("PASS: two independent derivations agree; reported number is the lookahead-free one.")


def per_instrument(tickers: list[str], cfg: Config, label: str) -> pd.DataFrame:
    # size each standalone asset as it would sit in an 8-asset book (N=8) so
    # leverage/MDD are sane; Sharpe is the leverage-invariant comparison metric.
    cfg1 = replace(cfg, n_assets=8)
    print("\n" + "=" * 90)
    print(f"PER-INSTRUMENT (standalone, sized as 1/8 book) — {label}")
    print("=" * 90)
    rows = {}
    for t in tickers:
        r = backtest([t], cfg1)
        m = r["metrics_gross"]
        rows[t] = m
        print(f"  {t:8s} {_fmt(m)}")
    return pd.DataFrame(rows).T


def main() -> None:
    cfg = Config()  # paper defaults, ratio-adjusted, full history, frictionless
    print("Config:", cfg)

    verify_lookahead(CORE, cfg)

    per_instrument(CORE, cfg, "CORE book")
    per_instrument(FX, cfg, "FX negative control")

    print("\n" + "=" * 90)
    print("PORTFOLIO — CORE book (8 instruments), frictionless")
    print("=" * 90)
    res = backtest(CORE, cfg)
    print(f"  Timed strategy     {_fmt(res['metrics_gross'])}")
    print(f"  avg gross exposure {res['gross_exposure'].mean()*100:.0f}%   "
          f"ann turnover {res['metrics_gross']['ann_turnover']:.1f}x")

    bench = benchmark_buyhold(CORE, cfg)
    print(f"  Always-long VT EW  {_fmt(bench['metrics'])}   "
          f"<- timing attribution (timed minus this = value of the timing rules)")

    # equity-indices-only: the paper's actual long-only-equity premise
    print("\n" + "=" * 90)
    print("PORTFOLIO — EQUITY INDICES ONLY (ES/NQ/YM/RTY) — the paper's true premise")
    print("=" * 90)
    eq = ["ES", "NQ", "YM", "RTY"]
    req = backtest(eq, cfg)
    beq = benchmark_buyhold(eq, cfg)
    print(f"  Timed strategy     {_fmt(req['metrics_gross'])}")
    print(f"  Always-long VT EW  {_fmt(beq['metrics'])}")

    # cost sensitivity (one-way bps on |delta w|)
    print("\n  Cost sensitivity (one-way bps on |Δw|):")
    for bps in (0, 1, 2, 5, 10):
        rc = backtest(CORE, Config(cost_bps=bps))
        print(f"    {bps:2d} bps : {_fmt(rc['metrics_net'])}")

    # FX portfolio (negative control)
    print("\n" + "=" * 90)
    print("PORTFOLIO — FX negative control")
    print("=" * 90)
    rfx = backtest(FX, cfg)
    print(f"  Timed strategy     {_fmt(rfx['metrics_gross'])}")
    bfx = benchmark_buyhold(FX, cfg)
    print(f"  Always-long VT EW  {_fmt(bfx['metrics'])}")

    # save the core equity curve
    out = pd.DataFrame({
        "timed_gross": (1 + res["gross_ret"]).cumprod(),
        "always_long_vt": (1 + bench["ret"]).cumprod(),
    })
    out.to_csv(OUT / "core_equity.csv")
    print(f"\nsaved {OUT / 'core_equity.csv'}")


if __name__ == "__main__":
    main()
