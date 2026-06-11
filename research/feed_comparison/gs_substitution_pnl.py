r"""Task 2 — feed-substitution P&L for the gold/silver book.

Takes the portfolio's daily position_fraction (from _gs_extract_positions.py,
scripts/_gs_out/positions_test.csv) and re-prices the GC and SI sleeves on each
execution feed:

  FUT_adj    additive back-adjusted future  (what signals were built on; vol-compressed)
  FUT_real   UNADJUSTED future              (fair futures % baseline)
  CFD_settle Darwinex CFD, stored-20:00 cut (~13:00 ET, aligned to the future settle)
  CFD_close  Darwinex CFD, stored-23:00 cut (16:00 ET, realistic daily mark)
  ETF        GLD / SLV (capital-adjusted)

Strategy P&L(feed, t) = sum_ticker pos(ticker, t-1) * r_feed(ticker, t).
Reports per GC, per SI, combined gold+silver, plus the WHOLE-book Sharpe with
gold/silver executed on CFD@close vs on futures. Validates the FUT reconstruction
against the pipeline's own combined_strategy_returns.

Run:
    .\.venv\Scripts\python.exe -m research.feed_comparison.gs_substitution_pnl
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_gs_out"

from research.feed_comparison import gs_signals as gs

ANN = np.sqrt(252.0)

# Future ticker -> (CFD symbol, ETF letter, ETF symbol)
GS_MAP = {
    "GC": ("XAUUSD", "G", "GLD"),
    "SI": ("XAGUSD", "S", "SLV"),
}
GS_TICKERS = list(GS_MAP)


def feed_close(ticker: str, feed: str) -> pd.Series:
    cfd, letter, etf = GS_MAP[ticker]
    if feed == "FUT_adj":
        return gs.fut_ohlc(ticker, adjusted=True)["close"]
    if feed == "FUT_real":
        return gs.fut_ohlc(ticker, adjusted=False)["close"]
    if feed == "CFD_settle":
        return gs.mt5_daily_ohlc(cfd, snap_stored_hour=20)["close"]
    if feed == "CFD_close":
        return gs.mt5_daily_ohlc(cfd, snap_stored_hour=23)["close"]
    if feed == "ETF":
        return gs.etf_ohlc(letter, etf, "CAP")["close"]
    raise ValueError(feed)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean() * 252.0, r.std() * ANN
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": (mu / sd if sd else np.nan), "n": len(r)}


def book_pnl(wide: pd.DataFrame, tickers: list[str], feed: str) -> pd.Series:
    """Sum_ticker pos(t-1) * logret_feed(t) over the given tickers."""
    cols = [t for t in tickers if t in wide.columns]
    rets = pd.concat(
        [gs.daily_logret_from_close(feed_close(t, feed)).rename(t) for t in cols],
        axis=1,
    )
    rets = rets.reindex(wide.index)
    w = wide[cols].shift(1)
    return (w * rets).sum(axis=1, min_count=1)


FEEDS = ["FUT_adj", "FUT_real", "CFD_settle", "CFD_close", "ETF"]


def report_block(title: str, wide: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 96)
    print(f"### {title}   (tickers={tickers})")
    pnl = {f: book_pnl(wide, tickers, f) for f in FEEDS}
    P = pd.DataFrame(pnl).dropna()
    if P.empty:
        print("  (no overlap)")
        return P
    print(f"  common window: {P.index.min().date()}..{P.index.max().date()}  n={len(P)}")
    base = P["FUT_real"]
    print(f"  {'feed':<12}{'ann_ret':>9}{'ann_vol':>9}{'sharpe':>8}"
          f"{'TE_vs_FUTreal':>15}{'corr_vs_FUTreal':>17}{'cumlog':>9}")
    rows = []
    for f in FEEDS:
        m = perf(P[f])
        te = (P[f] - base).std() * ANN
        cc = P[f].corr(base)
        cum = P[f].sum()
        rows.append({"book": title, "feed": f, **m, "TE_vs_FUTreal": te,
                     "corr_vs_FUTreal": cc, "cum_log": cum})
        print(f"  {f:<12}{m['ann_ret']*100:>8.2f}%{m['ann_vol']*100:>8.2f}%"
              f"{m['sharpe']:>8.2f}{te*100:>14.2f}%{cc:>17.4f}{cum*100:>8.2f}%")
    return pd.DataFrame(rows)


def main() -> None:
    pos = pd.read_csv(OUT / "positions_test.csv")
    pos["ticker"] = pos["ticker"].astype(str).str.replace("Ticker.", "", regex=False)
    pos["date"] = pd.to_datetime(pos["datetime"]).dt.normalize()
    all_tickers = sorted(pos["ticker"].unique())
    wide = pos.pivot_table(index="date", columns="ticker",
                           values="position_fraction", aggfunc="last").sort_index()
    print(f"positions: {wide.index.min().date()}..{wide.index.max().date()} "
          f"tickers={all_tickers} rows={len(wide)}")

    # ── validate FUT reconstruction against pipeline combined returns ──────────
    srf = OUT / "strategy_returns_futures.csv"
    if srf.exists():
        ext = pd.read_csv(srf, index_col=0)
        ext.index = pd.to_datetime(ext.index).normalize()
        ext = ext.iloc[:, 0]
        # whole-book FUT_real reconstruction across all tickers we can price
        priceable = [t for t in all_tickers if t in GS_MAP or t in ("ES", "NQ", "CL")]
        # Reconstruct on FUT_adj since the pipeline executes on back-adjusted futures
        rec = book_pnl_all(wide, all_tickers, "FUT_adj")
        common = rec.dropna().index.intersection(ext.dropna().index)
        c = rec.reindex(common).corr(ext.reindex(common))
        print(f"\n[validation] FUT_adj whole-book reconstruction vs pipeline "
              f"combined_strategy_returns: corr={c:.3f} (n={len(common)})")

    frames = []
    frames.append(report_block("GOLD (GC)", wide, ["GC"]))
    frames.append(report_block("SILVER (SI)", wide, ["SI"]))
    frames.append(report_block("GOLD+SILVER", wide, GS_TICKERS))

    # ── whole-book impact: gold/silver on CFD@close vs futures, rest on futures ─
    print("\n" + "#" * 96)
    print("# WHOLE-BOOK impact: all tickers on FUT_adj, then swap GC+SI to CFD@close")
    print("#" * 96)
    book_futures = book_pnl_all(wide, all_tickers, "FUT_adj").dropna()
    book_cfd = whole_book_with_gs_on_cfd(wide, all_tickers).dropna()
    common = book_futures.index.intersection(book_cfd.index)
    bf, bc = book_futures.reindex(common), book_cfd.reindex(common)
    mf, mc = perf(bf), perf(bc)
    print(f"  window {common.min().date()}..{common.max().date()} n={len(common)}")
    print(f"  {'variant':<28}{'ann_ret':>9}{'ann_vol':>9}{'sharpe':>8}")
    print(f"  {'all futures (FUT_adj)':<28}{mf['ann_ret']*100:>8.2f}%{mf['ann_vol']*100:>8.2f}%{mf['sharpe']:>8.2f}")
    print(f"  {'GC+SI -> CFD@close':<28}{mc['ann_ret']*100:>8.2f}%{mc['ann_vol']*100:>8.2f}%{mc['sharpe']:>8.2f}")
    print(f"  daily corr(all-fut, gs-cfd) = {bf.corr(bc):.4f}   "
          f"TE = {(bc-bf).std()*ANN*100:.2f}%")

    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out.to_csv(OUT / "gs_feed_pnl_summary.csv", index=False)
    print(f"\nwrote {OUT/'gs_feed_pnl_summary.csv'}")


def book_pnl_all(wide: pd.DataFrame, tickers: list[str], feed: str) -> pd.Series:
    """Whole-book P&L; non-gold/silver tickers always priced on their FUT feed
    (FUT_adj or FUT_real depending on `feed`)."""
    out = pd.Series(0.0, index=wide.index)
    contributed = pd.Series(False, index=wide.index)
    for t in tickers:
        s = _ticker_ret(t, feed)
        s = s.reindex(wide.index)
        w = wide[t].shift(1)
        contrib = w * s
        out = out.add(contrib.fillna(0.0))
        contributed = contributed | contrib.notna()
    return out.where(contributed)


def whole_book_with_gs_on_cfd(wide: pd.DataFrame, tickers: list[str]) -> pd.Series:
    out = pd.Series(0.0, index=wide.index)
    contributed = pd.Series(False, index=wide.index)
    for t in tickers:
        feed = "CFD_close" if t in GS_MAP else "FUT_adj"
        s = _ticker_ret(t, feed).reindex(wide.index)
        contrib = wide[t].shift(1) * s
        out = out.add(contrib.fillna(0.0))
        contributed = contributed | contrib.notna()
    return out.where(contributed)


def _ticker_ret(ticker: str, feed: str) -> pd.Series:
    """Daily logret for any ticker. Gold/silver have all feeds; ES/NQ/CL only have
    futures feeds here (we re-price the future via ohlc_data)."""
    if ticker in GS_MAP:
        return gs.daily_logret_from_close(feed_close(ticker, feed))
    # non-GS: FUT_adj / FUT_real from ohlc_data
    adjusted = feed != "FUT_real"
    return gs.daily_logret_from_close(gs.fut_ohlc(ticker, adjusted=adjusted)["close"])


if __name__ == "__main__":
    main()
