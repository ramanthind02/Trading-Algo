r"""Daily data-feed discrepancy study: Norgate futures vs Norgate cash/ETF vs
Darwinex (MT5) CFD/ETF. Read-only; prints tabular results.

Decomposition
-------------
Q1  Instrument/normalization effect (ALL Norgate, identical daily convention):
        back-adjusted FUTURE  vs  CASH index/spot  vs  ETF (capital-adj)
Q2  Broker-feed effect (Darwinex CFD/ETF M1->daily snapshot  vs  true Norgate):
        SP500_CFD vs SPX,  SPY_CFD vs SPY,  XAUUSD vs GC,  GLD_CFD vs GLD, ...

Also: confirm additive back-adjustment, and quantify the back-adjust offset in
the CFD overlap window.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OHLC = REPO / "data" / "ohlc_data"
MT5 = REPO / "data" / "mt5_data"
MS = REPO / "data" / "norgate" / "market_series"
STK = REPO / "data" / "stock_data"

ANN = np.sqrt(252.0)
# MT5 bar timestamps are broker-server time (Darwinex EET/EEST = UTC+2/+3), i.e.
# ET = stored - 7h. Calibrated (research/feed_comparison/feed_tz_calibrate.py): snapshotting the
# last M1 close at stored time-of-day 23:00 == 16:00 ET US cash close maximizes
# correlation to Norgate (SPY 0.995, QQQ 0.998, SP500 0.975).
SNAP_STORED_HHMM = (23, 0)  # 16:00 ET US cash close, in stored (server) tz


# ── loaders → return a daily close Series indexed by tz-naive date ──────────
def s_ohlc(ticker: str, tag: str = "D") -> pd.Series:
    p = OHLC / ticker / f"{tag}_{ticker}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    s = df["close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.rename(f"{ticker}/{tag}")


def s_ohlc_unadj(ticker: str) -> pd.Series:
    p = OHLC / ticker / f"D_{ticker}_unadj.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_parquet(p)["close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.rename(f"{ticker}/unadj")


def s_cash(folder: str, sym: str) -> pd.Series:
    p = MS / folder / f"{sym}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    s = df["close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.rename(f"cash:{sym}")


def s_etf(letter: str, sym: str, adj: str = "CAP") -> pd.Series:
    p = STK / letter / sym / f"D_{adj}_{sym}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    s = df["close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.rename(f"etf:{sym}/{adj}")


def s_mt5_snapshot(sym: str, snap=SNAP_STORED_HHMM) -> pd.Series:
    """Daily close = last M1 close at/before stored-time-of-day `snap` (16:00 ET).

    Bars are kept in their stored (server) tz; we cut at stored 23:00 which is
    16:00 ET year-round, and label the day by the stored calendar date (== ET
    date, since 23:00 EET is 16:00 ET same day).
    """
    base = MT5 / sym / "bars_M1"
    parts = sorted(base.glob("year=*/part.parquet"))
    if not parts:  # fall back to D1 bars
        d1 = MT5 / sym / "bars_D1" / "part.parquet"
        if not d1.exists():
            return pd.Series(dtype=float)
        df = pd.read_parquet(d1, columns=["time", "close"])
        t = pd.to_datetime(df["time"], utc=True)
        out = pd.Series(df["close"].values, index=t.dt.tz_convert(None).dt.normalize())
        return out.rename(f"cfd:{sym}")
    cutoff = snap[0] * 60 + snap[1]
    daily = {}
    for p in parts:
        df = pd.read_parquet(p, columns=["time", "close"])
        t = pd.to_datetime(df["time"], utc=True)  # stored tag = server tz, kept as-is
        mod = t.dt.hour * 60 + t.dt.minute
        keep = (mod <= cutoff).values
        if not keep.any():
            continue
        d = pd.DataFrame({"date": t[keep].dt.tz_convert(None).dt.normalize(),
                          "close": df["close"].values[keep]})
        daily.update(d.groupby("date")["close"].last().to_dict())
    s = pd.Series(daily).sort_index()
    return s.rename(f"cfd:{sym}")


def s_mt5_d1(sym: str) -> pd.Series:
    p = MT5 / sym / "bars_D1" / "part.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p, columns=["time", "close"])
    t = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None).dt.normalize()
    return pd.Series(df["close"].values, index=t).rename(f"cfdD1:{sym}")


def logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def compare_block(title: str, series: list[pd.Series], ref_name: str,
                  window: tuple[str, str] | None = None) -> None:
    print("\n" + "=" * 100)
    print(f"### {title}")
    series = [s for s in series if not s.empty]
    rets = pd.concat([logret(s) for s in series], axis=1).dropna(how="all")
    if window:
        rets = rets.loc[window[0]:window[1]]
    rets = rets.dropna()  # common dates across ALL feeds in block
    if rets.empty or rets.shape[0] < 20:
        print("  (insufficient overlap)")
        return
    print(f"  common window: {rets.index.min().date()} .. {rets.index.max().date()}  "
          f"n={len(rets)} days")
    cols = list(rets.columns)
    # correlation matrix
    corr = rets.corr()
    print("\n  -- daily log-return correlation --")
    print(corr.round(4).to_string())
    # tracking error vs reference + annualized vol + drift
    if ref_name not in cols:
        ref_name = cols[0]
    print(f"\n  -- vs reference [{ref_name}] (annualized) --")
    print(f"  {'feed':<22}{'corr':>8}{'ann_vol':>10}{'track_err':>11}{'ann_drift_vs_ref':>18}")
    rref = rets[ref_name]
    for c in cols:
        te = (rets[c] - rref).std() * ANN
        vol = rets[c].std() * ANN
        drift = (rets[c] - rref).mean() * 252.0
        print(f"  {c:<22}{corr.loc[c, ref_name]:>8.4f}{vol:>10.3f}"
              f"{te:>11.4f}{drift:>+18.4f}")


def main() -> None:
    # ── additive back-adjustment confirmation ──────────────────────────────
    print("=" * 100)
    print("ADJUSTMENT CONVENTION (ES: back-adjusted D vs unadjusted D, by date)")
    adj = s_ohlc("ES", "D")
    una = pd.read_parquet(OHLC / "ES" / "D_ES_unadj.parquet")["close"]
    una.index = pd.to_datetime(una.index).normalize()
    m = pd.concat([adj.rename("adj"), una.rename("unadj")], axis=1).dropna()
    m["diff"] = m["adj"] - m["unadj"]
    m["ratio"] = m["adj"] / m["unadj"]
    # piecewise-constant test: # of days the value JUMPS (changes between rolls)
    djumps = (m["diff"].round(3).diff().abs() > 1e-3).sum()
    rjumps = (m["ratio"].round(6).diff().abs() > 1e-6).sum()
    print(f"  rows={len(m)}  {m.index.min().date()}..{m.index.max().date()}")
    print(f"  DIFF (adj-unadj):  jumps={djumps:>5}  range=[{m['diff'].min():.2f},{m['diff'].max():.2f}]"
          f"   <- few jumps => ADDITIVE/arithmetic back-adjust")
    print(f"  RATIO(adj/unadj):  jumps={rjumps:>5}  range=[{m['ratio'].min():.4f},{m['ratio'].max():.4f}]"
          f"  <- few jumps => proportional/ratio")
    # offset magnitude in the CFD overlap window (2018+)
    recent = m.loc["2018":]
    print(f"  In CFD-overlap window (2018+): mean |adj-unadj| offset = {recent['diff'].abs().mean():.2f} pts "
          f"({(recent['diff'].abs()/recent['unadj']).mean()*100:.2f}% of price)")
    for yr in ("2008", "2012", "2018", "2024"):
        sub = m.loc[yr]
        if len(sub):
            r = sub.iloc[0]
            print(f"    {yr}: adj={r['adj']:.1f} unadj={r['unadj']:.1f} "
                  f"offset={r['diff']:+.1f} ({r['diff']/r['unadj']*100:+.2f}%)")

    # ── Q1: instrument/normalization (Norgate-only, full history) ──────────
    print("\n\n" + "#" * 100)
    print("# Q1 — INSTRUMENT/NORMALIZATION EFFECT (all Norgate, same daily convention)")
    print("#" * 100)
    compare_block(
        "S&P 500: ES_fut_adj | ES_fut_unadj | SPX_cash | SPY_cap | SPY_TR",
        [s_ohlc("ES", "D").rename("ES_fut_adj"),
         s_ohlc_unadj("ES").rename("ES_fut_unadj"),
         s_cash("us_indices", "SPX").rename("SPX_cash"),
         s_etf("S", "SPY", "CAP").rename("SPY_cap"),
         s_etf("S", "SPY", "TR").rename("SPY_TR")],
        "ES_fut_adj",
    )
    compare_block(
        "Nasdaq-100: NQ_fut_adj | NDX_cash | QQQ_cap | QQQ_TR",
        [s_ohlc("NQ", "D").rename("NQ_fut_adj"),
         s_cash("us_indices", "NDX").rename("NDX_cash"),
         s_etf("Q", "QQQ", "CAP").rename("QQQ_cap"),
         s_etf("Q", "QQQ", "TR").rename("QQQ_TR")],
        "NQ_fut_adj",
    )
    compare_block(
        "Gold: GC_fut_adj | GC_spot_cash | GLD_cap | GLD_TR",
        [s_ohlc("GC", "D").rename("GC_fut_adj"),
         s_cash("cash_commodities", "GC").rename("GC_spot"),
         s_etf("G", "GLD", "CAP").rename("GLD_cap"),
         s_etf("G", "GLD", "TR").rename("GLD_TR")],
        "GC_fut_adj",
    )
    compare_block(
        "Crude WTI: CL_fut_adj | WTI_spot_cash | USO_cap",
        [s_ohlc("CL", "D").rename("CL_fut_adj"),
         s_cash("cash_commodities", "WTI").rename("WTI_spot"),
         s_etf("U", "USO", "CAP").rename("USO_cap")],
        "CL_fut_adj",
    )

    # ── Q2: Darwinex broker-feed fidelity (CFD/ETF vs true Norgate) ────────
    print("\n\n" + "#" * 100)
    print("# Q2 — DARWINEX (MT5) FEED FIDELITY  [CFD M1->daily 16:00-NY snapshot vs Norgate]")
    print("#" * 100)
    win = ("2018-01-01", "2026-06-02")
    compare_block(
        "S&P500 feed: SPX_cash | SP500_CFD | SPY_cap | SPY_CFD | ES_fut_adj",
        [s_cash("us_indices", "SPX").rename("SPX_cash"),
         s_mt5_snapshot("SP500").rename("SP500_CFD"),
         s_etf("S", "SPY", "CAP").rename("SPY_cap"),
         s_mt5_snapshot("SPY").rename("SPY_CFD"),
         s_ohlc("ES", "D").rename("ES_fut_adj")],
        "SPX_cash", window=win,
    )
    compare_block(
        "Nasdaq feed: NDX_cash | QQQ_cap | QQQ_CFD | NQ_fut_adj",
        [s_cash("us_indices", "NDX").rename("NDX_cash"),
         s_etf("Q", "QQQ", "CAP").rename("QQQ_cap"),
         s_mt5_snapshot("QQQ").rename("QQQ_CFD"),
         s_ohlc("NQ", "D").rename("NQ_fut_adj")],
        "NDX_cash", window=win,
    )
    compare_block(
        "Gold feed: GC_spot | XAUUSD_CFD | GLD_cap | GLD_CFD | GC_fut_adj",
        [s_cash("cash_commodities", "GC").rename("GC_spot"),
         s_mt5_snapshot("XAUUSD").rename("XAUUSD_CFD"),
         s_etf("G", "GLD", "CAP").rename("GLD_cap"),
         s_mt5_snapshot("GLD").rename("GLD_CFD"),
         s_ohlc("GC", "D").rename("GC_fut_adj")],
        "GC_spot", window=win,
    )
    compare_block(
        "Silver feed: SI_spot | XAGUSD_CFD | SLV_cap | SLV_CFD | SI_fut_adj",
        [s_cash("cash_commodities", "SI").rename("SI_spot"),
         s_mt5_snapshot("XAGUSD").rename("XAGUSD_CFD"),
         s_etf("S", "SLV", "CAP").rename("SLV_cap"),
         s_mt5_snapshot("SLV").rename("SLV_CFD"),
         s_ohlc("SI", "D").rename("SI_fut_adj")],
        "SI_spot", window=win,
    )
    compare_block(
        "Crude feed: WTI_spot | XTIUSD_CFD | CL_fut_adj",
        [s_cash("cash_commodities", "WTI").rename("WTI_spot"),
         s_mt5_snapshot("XTIUSD").rename("XTIUSD_CFD"),
         s_ohlc("CL", "D").rename("CL_fut_adj")],
        "WTI_spot", window=win,
    )


if __name__ == "__main__":
    main()
