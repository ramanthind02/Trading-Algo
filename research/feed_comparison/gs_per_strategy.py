r"""Tasks 3 & 4 — per-sleeve signal view on the FUTURE vs the CFD, and an
assessment of running signals directly on CFD-native daily data (Option B).

Method (stated explicitly): we stream a daily OHLC frame through the ACTUAL
production node classes (DonchianBreakoutSignal, SmaRegimeSignalNode,
RobustTrendBreakout) configured from the three vault sleeve JSONs. We do this on
each execution feed's daily OHLC, then report:

  * position-agreement %  : fraction of common days where sign(signal) matches
    the FUT_adj reference (what the vault built on)
  * signal-return corr    : corr of the per-sleeve daily strategy return
    (pos(t-1)*r_feed(t)) vs the FUT_adj reference strategy return
  * vectorized backtest    : ann ret / vol / Sharpe of each sleeve on each feed,
    using that feed's own returns (does the strategy 'survive' on CFD-native data?)

For Option B we also quantify the CFD daily-history cleanliness back to 1998/2002
(usable span, gaps, flatline runs).

Run:
    .\.venv\Scripts\python.exe -m research.feed_comparison.gs_per_strategy
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_gs_out"
OUT.mkdir(parents=True, exist_ok=True)

from research.feed_comparison import gs_signals as gs

ANN = np.sqrt(252.0)

# sleeve -> (node kind, future ticker, CFD symbol, ETF letter/sym)
SLEEVES = {
    "gc_breakout/robust_trend_breakout_gc_long":      ("robust_trend", "GC", "XAUUSD", "G", "GLD"),
    "silver_mr/donchian_breakout_si_long_short":      ("donchian",     "SI", "XAGUSD", "S", "SLV"),
    "silver_trend/sma_regime_si_long_short":          ("sma_regime",   "SI", "XAGUSD", "S", "SLV"),
}

# Feeds, each a daily OHLC frame builder for (ticker, cfd, letter, etf)
def feed_ohlc(feed: str, ticker: str, cfd: str, letter: str, etf: str) -> pd.DataFrame:
    if feed == "FUT_adj":
        return gs.fut_ohlc(ticker, adjusted=True)
    if feed == "FUT_real":
        return gs.fut_ohlc(ticker, adjusted=False)
    if feed == "CFD_settle":
        return gs.mt5_daily_ohlc(cfd, snap_stored_hour=20)
    if feed == "CFD_close":
        return gs.mt5_daily_ohlc(cfd, snap_stored_hour=23)
    if feed == "ETF":
        return gs.etf_ohlc(letter, etf, "CAP")
    raise ValueError(feed)


FEEDS = ["FUT_adj", "FUT_real", "CFD_settle", "CFD_close", "ETF"]


def sleeve_signal(feed: str, kind: str, ticker: str, cfd: str, letter: str, etf: str,
                  window: tuple[str, str]) -> tuple[pd.Series, pd.Series]:
    """Return (signal, daily_logret) for a sleeve on a feed, sliced to window."""
    ohlc = feed_ohlc(feed, ticker, cfd, letter, etf)
    ohlc = ohlc.loc[(ohlc.index >= window[0]) & (ohlc.index <= window[1])]
    ohlc = ohlc[~ohlc.index.duplicated(keep="last")].sort_index()
    sig = gs.signal_for(ohlc, kind, ticker)
    ret = gs.daily_logret_from_close(ohlc["close"])
    return sig, ret


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean() * 252.0, r.std() * ANN
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": (mu / sd if sd else np.nan), "n": len(r)}


def per_sleeve_report(window: tuple[str, str]) -> pd.DataFrame:
    print("\n" + "#" * 96)
    print(f"# PER-SLEEVE: signal on FUT vs CFD   window={window}")
    print("#" * 96)
    rows = []
    for path, (kind, ticker, cfd, letter, etf) in SLEEVES.items():
        print("\n" + "=" * 92)
        print(f"### {path}   node={kind} ticker={ticker} cfd={cfd}")
        sigs, rets = {}, {}
        for f in FEEDS:
            try:
                s, r = sleeve_signal(f, kind, ticker, cfd, letter, etf, window)
            except Exception as exc:  # noqa: BLE001
                print(f"  {f}: skipped ({exc})")
                continue
            sigs[f], rets[f] = s, r
        ref_sig = sigs["FUT_adj"]
        # ref strategy return on the *real* futures feed
        ref_strat = (sigs["FUT_adj"].shift(1) * rets["FUT_real"]).dropna()
        print(f"  {'feed':<12}{'agree%':>9}{'strat_corr':>12}{'ann_ret':>9}"
              f"{'ann_vol':>9}{'sharpe':>8}{'n':>7}")
        for f in FEEDS:
            if f not in sigs:
                continue
            sig, ret = sigs[f], rets[f]
            common = ref_sig.dropna().index.intersection(sig.dropna().index)
            agree = (np.sign(ref_sig.reindex(common)) == np.sign(sig.reindex(common))).mean()
            strat = (sig.shift(1) * ret).dropna()
            cc = strat.corr(ref_strat.reindex(strat.index))
            m = perf(strat)
            rows.append({"sleeve": path, "feed": f, "agree_pct": agree,
                         "strat_corr_vs_futadj": cc, **m})
            print(f"  {f:<12}{agree*100:>8.2f}%{cc:>12.4f}{m['ann_ret']*100:>8.2f}%"
                  f"{m['ann_vol']*100:>8.2f}%{m['sharpe']:>8.2f}{m['n']:>7}")
    return pd.DataFrame(rows)


def cfd_history_quality(cfd: str) -> None:
    """Quantify CFD daily-history cleanliness for Option B (full span)."""
    ohlc = gs.mt5_daily_ohlc(cfd, snap_stored_hour=23)
    c = ohlc["close"].dropna()
    idx = c.index
    span_days = (idx.max() - idx.min()).days
    n = len(c)
    # business-day gaps > 4 calendar days (excl weekends ~ >4 => missing run)
    deltas = pd.Series(idx).diff().dt.days.dropna()
    big_gaps = int((deltas > 4).sum())
    max_gap = int(deltas.max()) if len(deltas) else 0
    # flatline runs: consecutive identical close
    flat = (c.diff() == 0)
    flat_runs = int(flat.sum())
    # zero / nan / negative
    bad = int(((c <= 0) | (~np.isfinite(c))).sum())
    ret = np.log(c).diff().dropna()
    jumps = int((ret.abs() > 0.15).sum())  # >15% single-day moves (suspicious)
    print(f"  {cfd}: {idx.min().date()}..{idx.max().date()}  rows={n}  "
          f"span={span_days/365.25:.1f}yr  gaps>4d={big_gaps} (max {max_gap}d)  "
          f"flat-days={flat_runs}  bad(<=0/nan)={bad}  |ret|>15%={jumps}")


def main() -> None:
    # Validate the node-streaming harness reproduces a sane regime on FUT_adj over
    # a long window (sanity: SMA-regime is +/-1 most days, donchian/breakout sparse).
    print("#" * 96)
    print("# Node-streaming sanity (FUT_adj, full history) — signal value distribution")
    print("#" * 96)
    for path, (kind, ticker, cfd, letter, etf) in SLEEVES.items():
        ohlc = gs.fut_ohlc(ticker, adjusted=True)
        sig = gs.signal_for(ohlc, kind, ticker)
        vc = sig.value_counts().to_dict()
        print(f"  {kind:<14} {ticker}: nonzero={int((sig!=0).sum())}/{len(sig)} "
              f"values={ {k: int(v) for k,v in sorted(vc.items())} }")

    # ── Option B: CFD daily-history cleanliness (full span) ───────────────────
    print("\n" + "#" * 96)
    print("# OPTION B — CFD daily-history cleanliness (full MT5 span, stored-23:00 mark)")
    print("#" * 96)
    for cfd in ("XAUUSD", "XAGUSD"):
        cfd_history_quality(cfd)

    # ── Per-sleeve on the OOS test window (matches the portfolio study) ────────
    oos = ("2023-01-01", "2026-05-13")
    df_oos = per_sleeve_report(oos)
    df_oos.to_csv(OUT / "per_sleeve_oos.csv", index=False)

    # ── Option B: same sleeves on a LONG common window where CFD history exists ─
    # gold CFD from 1998, silver CFD from 2002; use 2004+ so all warmups are filled.
    longw = ("2004-01-01", "2026-05-13")
    df_long = per_sleeve_report(longw)
    df_long.to_csv(OUT / "per_sleeve_long.csv", index=False)

    print(f"\nwrote {OUT/'per_sleeve_oos.csv'} and {OUT/'per_sleeve_long.csv'}")


if __name__ == "__main__":
    main()
