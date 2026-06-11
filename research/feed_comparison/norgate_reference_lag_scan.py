r"""Best-lag daily-return correlation scan for ALL Norgate reference families.

Confirms + extends the "time-of-day mismatch" root cause: a same-day daily
return only lines up across two series if both are snapshotted at the SAME
clock-time. Norgate *commodity cash* series (GC/SI/...) are a morning-London
benchmark fix (LBMA), so their same-day return correlation to the future
(13:30 ET settle) and the ETF (16:00 ET close) is capped well below 1.0 and
often improves at lag -1 (the cash fix leads the 16:00 close by one session of
information for the *earlier* fixes). Equity cash (SPX/NDX/DJI) closes 16:00 ET
== SPY/QQQ close, so it is clean at lag 0.

Read-only. Prints tables + writes a CSV/markdown report under
``research/feed_comparison/_out/``. Reuses the loaders from ``feed_compare.py``.

    venv\Scripts\python.exe -m research.feed_comparison.norgate_reference_lag_scan
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_out"

# reuse the verified loaders (s_ohlc, s_ohlc_unadj, s_cash, s_etf, s_mt5_snapshot)
from research.feed_comparison import feed_compare as fc

WIN = ("2018-01-01", "2026-06-04")
LAGS = (-2, -1, 0, 1, 2)


# ── helpers ──────────────────────────────────────────────────────────────────
def logret(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s = s[s > 0]  # guard against additive negatives / zeros for log
    return np.log(s).diff()


def best_lag_corr(ref: pd.Series, target: pd.Series,
                  window=WIN, lags=LAGS) -> dict:
    """Correlate ref's daily log-return against target shifted by each lag.

    lag k means: corr(ref_t, target_{t+k}). A negative best lag means the
    reference LEADS the target (its snapshot is earlier in the day).
    """
    r = logret(ref)
    t = logret(target)
    r = r.loc[window[0]:window[1]]
    t = t.loc[window[0]:window[1]]
    out = {"lag0": np.nan, "best_lag": None, "best_corr": np.nan, "n": 0}
    if r.dropna().empty or t.dropna().empty:
        return out
    per_lag = {}
    for k in lags:
        j = pd.concat([r.rename("r"), t.shift(k).rename("t")], axis=1).dropna()
        if len(j) < 60:
            per_lag[k] = np.nan
            continue
        per_lag[k] = j["r"].corr(j["t"])
    valid = {k: v for k, v in per_lag.items() if not pd.isna(v)}
    if not valid:
        return out
    bl = max(valid, key=lambda k: valid[k])
    j0 = pd.concat([r.rename("r"), t.rename("t")], axis=1).dropna()
    out.update(lag0=per_lag.get(0, np.nan), best_lag=bl,
               best_corr=valid[bl], n=len(j0), per_lag=per_lag)
    return out


def neg_price_flag(s: pd.Series) -> bool:
    return bool((s.dropna() < 0).any())


# ── reference -> {future, etf} analog map ────────────────────────────────────
# cash symbol (folder cash_commodities) : (future ticker | None, (etf_letter, etf_sym) | None, label)
CASH_MAP = [
    ("GC",     "GC", ("G", "GLD"),  "Gold spot"),
    ("SI",     "SI", ("S", "SLV"),  "Silver spot"),
    ("WTI",    "CL", ("U", "USO"),  "WTI crude spot"),
    ("HO",     "HO", None,          "Heating oil spot"),
    ("PL",     "PL", ("P", "PPLT"), "Platinum spot"),
    ("PA",     None, ("P", "PALL"), "Palladium spot"),
    ("CU",     "HG", ("C", "CPER"), "Copper (LME) spot"),
    ("HHNG",   None, ("U", "UNG"),  "Henry Hub natgas spot"),
    ("CO",     None, None,          "Brent crude spot"),
    ("BCOM",   None, ("D", "DBC"),  "Bloomberg Commodity index"),
    ("CRB",    None, None,          "CRB commodity index"),
    ("SPGSCI", None, None,          "S&P GSCI commodity index"),
]

# Equity cash (us_indices) should be CLEAN at lag 0 (16:00 ET close == ETF/future).
US_INDEX_MAP = [
    ("SPX", "ES",  ("S", "SPY"), "S&P 500 cash"),
    ("NDX", "NQ",  ("Q", "QQQ"), "Nasdaq-100 cash"),
    ("DJI", "YM",  ("D", "DIA"), "Dow 30 cash"),
    ("RUT", "RTY", ("I", "IWM"), "Russell 2000 cash"),
]

# FX spot (forex_spot) vs FX futures.  Norgate FX futures are quoted FOREIGN/USD
# while the future ticker convention may invert; correlate on abs() of returns is
# wrong, so we test both the pair and its inverse and keep the better.
FX_MAP = [
    ("EURUSD", "EU", "EUR/USD spot"),
    ("JPYUSD", "JY", "JPY/USD spot"),
    ("GBPUSD", "BP", "GBP/USD spot"),
    ("CADUSD", "CD", "CAD/USD spot"),
    ("CHFUSD", "SF", "CHF/USD spot"),
]


def _fmt_lags(per_lag: dict | None) -> str:
    if not per_lag:
        return ""
    return " ".join(f"{k:+d}={v:.3f}" if not pd.isna(v) else f"{k:+d}=  -  "
                    for k, v in sorted(per_lag.items()))


def run_block(title: str, rows: list[dict]) -> pd.DataFrame:
    print("\n" + "=" * 110)
    print(f"### {title}")
    print("=" * 110)
    hdr = (f"{'cash_sym':<9}{'vs':<10}{'lag0':>8}{'best_lag':>9}{'best_corr':>10}"
           f"{'n':>6}  {'verdict':<22} per-lag")
    print(hdr)
    print("-" * 110)
    for r in rows:
        bl = "" if r["best_lag"] is None else f"{r['best_lag']:+d}"
        print(f"{r['cash_sym']:<9}{r['vs']:<10}{r['lag0']:>8.3f}"
              f"{bl:>9}{r['best_corr']:>10.3f}{r['n']:>6}  "
              f"{r['verdict']:<22} {_fmt_lags(r.get('per_lag'))}")
    return pd.DataFrame(rows)


def verdict(lag0: float, best_lag, best_corr: float, family: str) -> str:
    if pd.isna(lag0):
        return "no-overlap"
    if family == "equity":
        return "CLEAN (lag0>=0.95)" if lag0 >= 0.95 else "UNEXPECTED-mismatch"
    # commodity / fx
    if lag0 >= 0.95:
        return "clock-aligned"
    if best_lag is not None and best_lag != 0 and best_corr - lag0 > 0.02:
        return "TIME-OF-DAY-MISMATCH"
    return "low-corr (mismatch)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # ── commodity cash ──────────────────────────────────────────────────────
    com_rows = []
    for cash_sym, fut, etf, label in CASH_MAP:
        ref = fc.s_cash("cash_commodities", cash_sym)
        if ref.empty:
            continue
        # vs future (adjusted)
        if fut is not None:
            tgt = fc.s_ohlc(fut, "D")
            res = best_lag_corr(ref, tgt)
            v = verdict(res["lag0"], res["best_lag"], res["best_corr"], "commodity")
            row = dict(cash_sym=cash_sym, vs=f"FUT:{fut}", label=label, family="commodity",
                       lag0=res["lag0"], best_lag=res["best_lag"],
                       best_corr=res["best_corr"], n=res["n"],
                       per_lag=res.get("per_lag"), verdict=v,
                       neg_price_target=neg_price_flag(tgt))
            com_rows.append(row); all_rows.append(row)
        # vs ETF (capital-adjusted)
        if etf is not None:
            tgt = fc.s_etf(etf[0], etf[1], "CAP")
            res = best_lag_corr(ref, tgt)
            v = verdict(res["lag0"], res["best_lag"], res["best_corr"], "commodity")
            row = dict(cash_sym=cash_sym, vs=f"ETF:{etf[1]}", label=label, family="commodity",
                       lag0=res["lag0"], best_lag=res["best_lag"],
                       best_corr=res["best_corr"], n=res["n"],
                       per_lag=res.get("per_lag"), verdict=v,
                       neg_price_target=neg_price_flag(tgt))
            com_rows.append(row); all_rows.append(row)
    run_block("COMMODITY CASH (Norgate cash_commodities) vs FUTURE / ETF  [2018+]", com_rows)

    # ── equity cash (control: must be clean) ────────────────────────────────
    eq_rows = []
    for cash_sym, fut, etf, label in US_INDEX_MAP:
        ref = fc.s_cash("us_indices", cash_sym)
        if ref.empty:
            continue
        if fut:
            tgt = fc.s_ohlc(fut, "D")
            res = best_lag_corr(ref, tgt)
            v = verdict(res["lag0"], res["best_lag"], res["best_corr"], "equity")
            row = dict(cash_sym=cash_sym, vs=f"FUT:{fut}", label=label, family="equity",
                       lag0=res["lag0"], best_lag=res["best_lag"], best_corr=res["best_corr"],
                       n=res["n"], per_lag=res.get("per_lag"), verdict=v, neg_price_target=False)
            eq_rows.append(row); all_rows.append(row)
        if etf:
            tgt = fc.s_etf(etf[0], etf[1], "CAP")
            res = best_lag_corr(ref, tgt)
            v = verdict(res["lag0"], res["best_lag"], res["best_corr"], "equity")
            row = dict(cash_sym=cash_sym, vs=f"ETF:{etf[1]}", label=label, family="equity",
                       lag0=res["lag0"], best_lag=res["best_lag"], best_corr=res["best_corr"],
                       n=res["n"], per_lag=res.get("per_lag"), verdict=v, neg_price_target=False)
            eq_rows.append(row); all_rows.append(row)
    run_block("EQUITY CASH (Norgate us_indices) vs FUTURE / ETF  [2018+]  (control: expect CLEAN)", eq_rows)

    # ── FX spot ─────────────────────────────────────────────────────────────
    fx_rows = []
    for cash_sym, fut, label in FX_MAP:
        ref = fc.s_cash("forex_spot", cash_sym)
        if ref.empty or fut is None:
            continue
        tgt = fc.s_ohlc(fut, "D")
        # FX future may be quoted in the same direction (FOREIGN/USD) as *USD pairs;
        # take the better of (ref vs tgt) and (ref vs 1/tgt) to be quote-direction agnostic.
        res = best_lag_corr(ref, tgt)
        res_inv = best_lag_corr(ref, (1.0 / tgt.astype(float)))
        if abs(res_inv["best_corr"]) > abs(res["best_corr"]):
            res = res_inv
            note = "(inv)"
        else:
            note = ""
        v = verdict(res["lag0"], res["best_lag"], res["best_corr"], "commodity")
        row = dict(cash_sym=cash_sym, vs=f"FUT:{fut}{note}", label=label, family="fx",
                   lag0=res["lag0"], best_lag=res["best_lag"], best_corr=res["best_corr"],
                   n=res["n"], per_lag=res.get("per_lag"), verdict=v, neg_price_target=False)
        fx_rows.append(row); all_rows.append(row)
    run_block("FX SPOT (Norgate forex_spot) vs FX FUTURE  [2018+]", fx_rows)

    # ── write report ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    keep = ["family", "cash_sym", "vs", "label", "lag0", "best_lag", "best_corr",
            "n", "verdict", "neg_price_target"]
    df_out = df[keep].copy()
    csv = OUT / "reference_lag_scan.csv"
    df_out.to_csv(csv, index=False)
    print(f"\nWrote {csv}")

    # markdown
    md = OUT / "reference_lag_scan.md"
    with open(md, "w") as f:
        f.write("# Norgate reference-series best-lag correlation scan (2018+)\n\n")
        f.write("`best_lag` is k in corr(ref_t, target_{t+k}); negative => the "
                "reference snapshot LEADS the target (earlier clock-time).\n\n")
        f.write(df_out.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
