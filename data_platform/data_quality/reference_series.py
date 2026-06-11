r"""Reference-series data-quality validator (permanent guardrail).

Purpose
-------
We were silently using mis-timed / mis-adjusted price series as same-day return
references (Norgate *commodity cash* = a morning London/LBMA benchmark fix, not
a 16:00-ET close; additive back-adjusted continuous futures carry NEGATIVE
historical prices that blow up % returns). Both are the same pathology: a series
used for daily %-return math must be snapshotted at the same clock-time as what
it is compared against, and must be on a multiplicative (ratio/unadjusted) price
scale so %-returns are well-defined.

This validator runs three READ-ONLY checks over the registered reference series
and **fails loudly (nonzero exit)** so CI / a scheduled job catches any
regression before research or live trading consumes a bad reference:

  CHECK 1  same-day return alignment
      For every series DECLARED as a same-day return reference (declared_lag=0),
      the best-lag daily-log-return correlation to its canonical partner must be
      >= MIN_CORR (0.95) AND occur at the declared lag. A series whose best lag
      is non-zero, or whose lag-0 corr is below threshold, is a time-of-day
      mismatch and FAILS.

  CHECK 2  no negative prices on a %-returns series
      Any series tagged for %-return math (futures back-adjusted, cash, ETF)
      must have strictly positive prices. Additive back-adjustment can push
      historical levels negative (crude Apr-2020), which makes log/%% returns
      undefined -> FAIL.

  CHECK 3  additive-vs-proportional adjustment sanity
      For each future with both an adjusted (D) and unadjusted (D_unadj) series,
      classify the adjustment as ADDITIVE (adj-unadj piecewise-constant) vs
      RATIO (adj/unadj piecewise-constant). ADDITIVE on a series used as a
      %-return reference is flagged (warning, or FAIL under --strict-additive).

Output
------
A CSV + markdown report under ``data/quality_reports/`` and a console
summary. Exit code is the number of hard failures (0 = all good).

Usage
-----
    venv\Scripts\python.exe -m data_platform.data_quality.reference_series
    venv\Scripts\python.exe -m data_platform.data_quality.reference_series --strict-additive
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OHLC = REPO / "data" / "ohlc_data"
MS = REPO / "data" / "norgate" / "market_series"
STK = REPO / "data" / "stock_data"
OUT = REPO / "data" / "quality_reports"

MIN_CORR = 0.95          # a declared same-day reference must clear this at its declared lag
WINDOW = ("2018-01-01", "2026-12-31")
LAGS = (-2, -1, 0, 1, 2)


# ── loaders (self-contained; do not depend on scratch _feed_* scripts) ───────
def _load_close(path: Path, index_is_col: bool = False) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    s = df["close"].astype(float).copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s


def s_future_adj(ticker: str) -> pd.Series:
    return _load_close(OHLC / ticker / f"D_{ticker}.parquet")


def s_future_unadj(ticker: str) -> pd.Series:
    return _load_close(OHLC / ticker / f"D_{ticker}_unadj.parquet")


def s_future_ratio(ticker: str) -> pd.Series:
    return _load_close(OHLC / ticker / f"D_{ticker}_ratio.parquet")


def s_cash(folder: str, sym: str) -> pd.Series:
    return _load_close(MS / folder / f"{sym}.parquet")


def s_etf(sym: str, adj: str = "CAP") -> pd.Series:
    return _load_close(STK / sym[0] / sym / f"D_{adj}_{sym}.parquet")


# ── core stats ───────────────────────────────────────────────────────────────
def _logret(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s = s.where(s > 0)  # log-undefined for <=0; leaves NaN to be dropped
    return np.log(s).diff()


def best_lag_corr(ref: pd.Series, tgt: pd.Series) -> tuple[float, int, float, int]:
    """Return (corr_at_lag0, best_lag, corr_at_best_lag, n_overlap)."""
    r = _logret(ref).loc[WINDOW[0]:WINDOW[1]]
    t = _logret(tgt).loc[WINDOW[0]:WINDOW[1]]
    per: dict[int, float] = {}
    for k in LAGS:
        j = pd.concat([r.rename("r"), t.shift(k).rename("t")], axis=1).dropna()
        per[k] = j["r"].corr(j["t"]) if len(j) >= 60 else np.nan
    valid = {k: v for k, v in per.items() if not pd.isna(v)}
    if not valid:
        return (np.nan, 0, np.nan, 0)
    bl = max(valid, key=lambda k: valid[k])
    n0 = len(pd.concat([r, t], axis=1).dropna())
    return (per.get(0, np.nan), bl, valid[bl], n0)


def has_negative_price(s: pd.Series) -> bool:
    return bool((s.dropna() < 0).any())


def classify_adjustment(adj: pd.Series, unadj: pd.Series) -> str:
    """ADDITIVE if (adj-unadj) is piecewise-constant; RATIO if (adj/unadj) is."""
    m = pd.concat([adj.rename("a"), unadj.rename("u")], axis=1).dropna()
    if len(m) < 50:
        return "unknown"
    diff_jumps = int((m["a"] - m["u"]).round(3).diff().abs().gt(1e-3).sum())
    ratio = (m["a"] / m["u"]).replace([np.inf, -np.inf], np.nan)
    ratio_jumps = int(ratio.round(6).diff().abs().gt(1e-6).sum())
    return "additive" if diff_jumps <= ratio_jumps else "ratio"


# ── registry of reference series + their canonical same-day partner ──────────
@dataclass(frozen=True)
class RefSpec:
    name: str
    family: str                # equity | commodity | fx
    ref: str                   # how to load (encoded below)
    partner: str               # canonical 16:00-ET partner to align against
    declared_lag: int = 0      # the lag at which this series is USED as a same-day ref
    pct_returns: bool = True   # is this series used for %-return math?


# Loader dispatch by encoded "kind:arg" strings keeps the registry declarative.
def _load(spec: str) -> pd.Series:
    kind, arg = spec.split(":", 1)
    if kind == "futadj":
        return s_future_adj(arg)
    if kind == "futunadj":
        return s_future_unadj(arg)
    if kind == "cashcom":
        return s_cash("cash_commodities", arg)
    if kind == "cashidx":
        return s_cash("us_indices", arg)
    if kind == "fx":
        return s_cash("forex_spot", arg)
    if kind == "etf":
        return s_etf(arg)
    raise ValueError(f"unknown loader kind {kind!r}")


# Each commodity cash series is DECLARED as a same-day (lag-0) return reference —
# which is exactly the (false) assumption this validator exists to catch.
REGISTRY: list[RefSpec] = [
    # equity cash (control — these SHOULD pass)
    RefSpec("SPX_cash",  "equity",    "cashidx:SPX", "etf:SPY"),
    RefSpec("NDX_cash",  "equity",    "cashidx:NDX", "etf:QQQ"),
    RefSpec("DJI_cash",  "equity",    "cashidx:DJI", "etf:DIA"),
    RefSpec("RUT_cash",  "equity",    "cashidx:RUT", "etf:IWM"),
    # commodity cash (expected to FAIL check-1 — mis-timed LBMA/spot fixes)
    RefSpec("GC_cash",   "commodity", "cashcom:GC",  "etf:GLD"),
    RefSpec("SI_cash",   "commodity", "cashcom:SI",  "etf:SLV"),
    RefSpec("WTI_cash",  "commodity", "cashcom:WTI", "etf:USO"),
    RefSpec("PL_cash",   "commodity", "cashcom:PL",  "etf:PPLT"),
    RefSpec("PA_cash",   "commodity", "cashcom:PA",  "etf:PALL"),
    RefSpec("CU_cash",   "commodity", "cashcom:CU",  "etf:CPER"),
    RefSpec("HHNG_cash", "commodity", "cashcom:HHNG", "etf:UNG"),
    # fx spot (mostly clean)
    RefSpec("EURUSD_spot", "fx", "fx:EURUSD", "futadj:EU"),
    RefSpec("JPYUSD_spot", "fx", "fx:JPYUSD", "futadj:JY"),
    RefSpec("GBPUSD_spot", "fx", "fx:GBPUSD", "futadj:BP"),
]

# Futures whose back-adjusted (D) series is consumed for %-returns / σ and must
# therefore not carry negative prices, and whose adjustment convention is audited.
FUTURES_FOR_PCT = ["ES", "NQ", "YM", "RTY", "GC", "SI", "CL", "HG", "PL", "HO",
                   "TY", "FV", "US", "TU", "C", "S", "W", "EU", "JY", "BP", "CD", "SF"]


# ── checks ───────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    check: str
    series: str
    status: str          # PASS | FAIL | WARN | SKIP
    detail: str
    metrics: dict = field(default_factory=dict)


def check_alignment() -> list[Finding]:
    out: list[Finding] = []
    for spec in REGISTRY:
        ref = _load(spec.ref)
        tgt = _load(spec.partner)
        if ref.empty or tgt.empty:
            out.append(Finding("alignment", spec.name, "SKIP",
                               f"missing data (ref or partner {spec.partner})"))
            continue
        lag0, bl, blc, n = best_lag_corr(ref, tgt)
        passed = (not np.isnan(lag0)) and lag0 >= MIN_CORR and bl == spec.declared_lag
        status = "PASS" if passed else "FAIL"
        if bl != spec.declared_lag and not np.isnan(blc):
            why = (f"best lag {bl:+d} != declared {spec.declared_lag:+d} "
                   f"(time-of-day mismatch); lag0_corr={lag0:.3f} best_corr={blc:.3f}")
        else:
            why = f"lag0_corr={lag0:.3f} < {MIN_CORR} (declared lag {spec.declared_lag:+d})"
        detail = (f"vs {spec.partner}: lag0={lag0:.3f} best_lag={bl:+d} "
                  f"best_corr={blc:.3f} n={n}" + ("" if passed else f" -> {why}"))
        out.append(Finding("alignment", spec.name, status, detail,
                           {"lag0": lag0, "best_lag": bl, "best_corr": blc, "n": n,
                            "family": spec.family, "partner": spec.partner}))
    return out


def check_negative_prices() -> list[Finding]:
    out: list[Finding] = []
    # commodity/equity cash + etfs from the registry
    for spec in REGISTRY:
        if not spec.pct_returns:
            continue
        s = _load(spec.ref)
        if s.empty:
            continue
        neg = has_negative_price(s)
        out.append(Finding("neg_price", spec.name, "FAIL" if neg else "PASS",
                           f"min={s.min():.4f}" + (" NEGATIVE" if neg else "")))
    # futures adjusted series used for %-returns
    for t in FUTURES_FOR_PCT:
        s = s_future_adj(t)
        if s.empty:
            continue
        neg = has_negative_price(s)
        out.append(Finding("neg_price", f"{t}_fut_adj", "FAIL" if neg else "PASS",
                           f"min={s.min():.4f}" + (" NEGATIVE (additive blow-up)" if neg else "")))
    return out


def check_adjustment_convention(strict_additive: bool) -> list[Finding]:
    out: list[Finding] = []
    for t in FUTURES_FOR_PCT:
        adj, unadj = s_future_adj(t), s_future_unadj(t)
        if adj.empty or unadj.empty:
            out.append(Finding("adjustment", f"{t}_fut", "SKIP", "missing adj or unadj"))
            continue
        kind = classify_adjustment(adj, unadj)
        if kind == "additive":
            status = "FAIL" if strict_additive else "WARN"
            detail = ("ADDITIVE back-adjustment used for %%-returns: inflates "
                      "historical levels -> deflates %% returns; can go negative. "
                      "Use a RATIO series as the %%-return reference.")
        else:
            status = "PASS"
            detail = f"adjustment classified as {kind}"
        out.append(Finding("adjustment", f"{t}_fut", status, detail, {"kind": kind}))
    return out


# ── scoped check (pipeline preflight) ────────────────────────────────────────
def validate_consumed_series(tickers: list[str]) -> list[Finding]:
    """Validate ONLY the %-return / σ series the portfolio pipeline consumes.

    The repointed architecture feeds the RATIO series (``D_{T}_ratio.parquet``)
    to EWSD σ and the IDM/weight return correlation. This scoped check audits,
    for each *consumed* ticker, that:

      * the ratio series exists (so we are not silently on the additive
        fallback for a traded instrument) — missing -> WARN;
      * the ratio series has no negative prices on a positive source -> FAIL
        (a negative inherited from a genuinely negative source print, e.g. WTI
        Apr-2020, is a WARN, not a method failure);
      * the ratio series is classified RATIO vs additive -> additive is a FAIL
        (the whole point of the repoint is to avoid additive %-return math).

    It deliberately does NOT run the commodity-cash same-day alignment checks
    (those reference series are not consumed by the portfolio pipeline and are
    expected to fail by construction). Returns findings; never exits.
    """
    out: list[Finding] = []
    for t in sorted({str(x) for x in tickers}):
        ratio = s_future_ratio(t)
        unadj = s_future_unadj(t)
        # 1) availability of the consumed ratio series
        if ratio.empty:
            out.append(Finding(
                "consumed_ratio", f"{t}_ratio", "WARN",
                "ratio series missing - consumer is on the unadjusted/additive "
                "fallback (run data_platform.providers.norgate.backadjust.ratio_driver)",
            ))
            continue
        # 2) negative prices (method failure only when the source was positive)
        ratio_neg = has_negative_price(ratio)
        source_neg = has_negative_price(unadj) if not unadj.empty else False
        if ratio_neg and not source_neg:
            out.append(Finding(
                "consumed_ratio", f"{t}_ratio", "FAIL",
                f"ratio min={ratio.min():.4f} NEGATIVE with positive source "
                f"(method failure - breaks log/%%-returns)",
            ))
        elif ratio_neg and source_neg:
            out.append(Finding(
                "consumed_ratio", f"{t}_ratio", "WARN",
                f"ratio min={ratio.min():.4f} <0 inherited from a NEGATIVE source "
                f"print (genuine, e.g. WTI Apr-2020) - not a method failure",
            ))
        else:
            out.append(Finding(
                "consumed_ratio", f"{t}_ratio", "PASS", f"min={ratio.min():.4f}",
            ))
        # 3) adjustment convention of the consumed series must be RATIO, not additive
        if not unadj.empty:
            kind = classify_adjustment(ratio, unadj)
            if kind == "additive":
                out.append(Finding(
                    "consumed_adjustment", f"{t}_ratio", "FAIL",
                    "consumed series classified ADDITIVE - %%-return math is distorted",
                    {"kind": kind},
                ))
            else:
                out.append(Finding(
                    "consumed_adjustment", f"{t}_ratio", "PASS",
                    f"adjustment classified as {kind}", {"kind": kind},
                ))
    return out


# ── report ───────────────────────────────────────────────────────────────────
def write_report(findings: list[Finding]) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"check": f.check, "series": f.series, "status": f.status,
             "detail": f.detail, **f.metrics} for f in findings]
    df = pd.DataFrame(rows)
    csv = OUT / "validate_reference_series.csv"
    df.to_csv(csv, index=False)
    md = OUT / "validate_reference_series.md"
    with open(md, "w") as fh:
        fh.write("# Reference-series data-quality validation\n\n")
        for chk in ("alignment", "neg_price", "adjustment"):
            sub = df[df["check"] == chk]
            if sub.empty:
                continue
            fh.write(f"## {chk}\n\n")
            cols = ["series", "status", "detail"]
            fh.write(sub[cols].to_markdown(index=False))
            fh.write("\n\n")
    return csv, md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate Norgate reference series quality.")
    p.add_argument("--strict-additive", action="store_true",
                   help="Treat ADDITIVE back-adjustment on a %%-return series as a hard FAIL.")
    args = p.parse_args(argv)

    findings: list[Finding] = []
    findings += check_alignment()
    findings += check_negative_prices()
    findings += check_adjustment_convention(args.strict_additive)

    csv, md = write_report(findings)

    # console summary
    fails = [f for f in findings if f.status == "FAIL"]
    warns = [f for f in findings if f.status == "WARN"]
    print("=" * 96)
    print("REFERENCE-SERIES DATA-QUALITY VALIDATION")
    print("=" * 96)
    for chk in ("alignment", "neg_price", "adjustment"):
        sub = [f for f in findings if f.check == chk]
        if not sub:
            continue
        print(f"\n-- {chk} --")
        for f in sub:
            mark = {"PASS": "ok  ", "FAIL": "FAIL", "WARN": "warn", "SKIP": "skip"}[f.status]
            print(f"  [{mark}] {f.series:<14} {f.detail}")
    print("\n" + "-" * 96)
    print(f"SUMMARY: {len(fails)} FAIL, {len(warns)} WARN, "
          f"{sum(1 for f in findings if f.status=='PASS')} PASS, "
          f"{sum(1 for f in findings if f.status=='SKIP')} SKIP")
    print(f"Report: {csv}")
    print(f"        {md}")
    if fails:
        print("\nHARD FAILURES (these series must NOT be used as same-day %%-return references):")
        for f in fails:
            print(f"  - [{f.check}] {f.series}: {f.detail}")
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
