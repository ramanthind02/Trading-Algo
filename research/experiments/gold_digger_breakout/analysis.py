"""
Phase-1 realism + robustness battery for the Gold Digger breakout POC.

Builds on poc.py. Reports:
  1. Gross vs gap-aware-entry (honest fills on session gaps).
  2. Cost sensitivity: net avg_R / Sharpe / PF / total_R vs round-trip cost (price),
     plus the closed-form total-R breakeven cost.
  3. Cost-in-R by range-width quintile (tight-range slippage fragility).
  4. Range-window robustness sweep (rs_h x re_h) + close-hour sweep.
  5. SL-to-breakeven variant sweep (the user's best-reported feature), gross + costed.

Cost model: percent-risk sizing => cost in R = cost_price / range_width. So a fixed
price cost hurts tight-range days far more (they carry larger size). net_R = gross_R
- cost_price/width per trade. total-R breakeven: cost* = sum(gross_R) / sum(1/width).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.gold_digger_breakout.poc import (
    load_m1, prep_days, build_trades, stats, by_year,
)

# Reference round-trip cost levels (price units), spanning recorded-spread-only ->
# spread + generous breakout slippage.
COST_LEVELS = {
    "XAUUSD": [0.0, 0.10, 0.30, 0.50, 0.80, 1.20],   # $ (point=0.01)
    "USDJPY": [0.0, 0.005, 0.010, 0.020, 0.040, 0.060],  # price (point=0.001)
}


def apply_cost(tr: pd.DataFrame, cost_price: float) -> pd.DataFrame:
    t = tr.copy()
    t["R"] = t["R"] - cost_price / t["width"]
    return t


def headline(tr: pd.DataFrame, all_days, cost_price: float = 0.0) -> dict:
    t = apply_cost(tr, cost_price) if cost_price else tr
    s = stats(t, all_days)
    return dict(cost=cost_price, n=s["n_trades"], win=round(s["win_rate"], 3),
                avg_R=round(s["avg_R"], 4), total_R=round(s["total_R"], 1),
                PF=round(s["profit_factor"], 3), Sharpe=round(s["sharpe_daily"], 3),
                maxDD_R=round(s["maxdd_R"], 1))


def breakeven_cost(tr: pd.DataFrame) -> float:
    return tr["R"].sum() / (1.0 / tr["width"]).sum()


def width_buckets(tr: pd.DataFrame, ref_cost: float) -> pd.DataFrame:
    t = tr.copy()
    t["q"] = pd.qcut(t["width"], 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for q, g in t.groupby("q", observed=True):
        gross = g["R"].mean()
        cost_R = (ref_cost / g["width"]).mean()
        rows.append(dict(width_q=int(q), n=len(g),
                         med_width=round(g["width"].median(), 3),
                         gross_avg_R=round(gross, 3),
                         cost_R_at_ref=round(cost_R, 3),
                         net_avg_R=round(gross - cost_R, 3)))
    return pd.DataFrame(rows)


def window_sweep(days, all_days, close_h=18, cost_price=0.0) -> pd.DataFrame:
    rows = []
    for rs in [1, 2, 3, 4]:
        for re in [5, 6, 7, 8]:
            if re <= rs:
                continue
            tr = build_trades(days, rs, re, close_h)
            if len(tr) < 50:
                continue
            h = headline(tr, all_days, cost_price)
            rows.append(dict(rs=rs, re=re, **{k: h[k] for k in
                        ("n", "win", "avg_R", "total_R", "PF", "Sharpe")}))
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


def close_sweep(days, all_days, rs=3, re=6, cost_price=0.0) -> pd.DataFrame:
    rows = []
    for ch in [12, 14, 16, 18, 20, 22]:
        tr = build_trades(days, rs, re, ch)
        h = headline(tr, all_days, cost_price)
        rows.append(dict(close_h=ch, **{k: h[k] for k in
                    ("n", "win", "avg_R", "total_R", "PF", "Sharpe")}))
    return pd.DataFrame(rows)


def be_sweep(days, all_days, rs=3, re=6, close_h=18, cost_price=0.0) -> pd.DataFrame:
    rows = []
    for be in [None, 0.5, 0.75, 1.0, 1.5, 2.0]:
        tr = build_trades(days, rs, re, close_h, be_R=be)
        h = headline(tr, all_days, cost_price)
        rows.append(dict(be_R=("none" if be is None else be),
                    n_be=int((tr["reason"] == "BE").sum()),
                    **{k: h[k] for k in ("win", "avg_R", "total_R", "PF", "Sharpe", "maxDD_R")}))
    return pd.DataFrame(rows)


def run_symbol(sym: str) -> None:
    print(f"\n{'#'*78}\n#  {sym}\n{'#'*78}")
    df = load_m1(sym)
    days = prep_days(df)
    all_days = df["day"].values

    tr = build_trades(days, 3, 6, 18, gap_fill=False)
    tr_gap = build_trades(days, 3, 6, 18, gap_fill=True)

    print("\n--- gross, range 03-06, flat 18 ---")
    print(pd.DataFrame([headline(tr, all_days), headline(tr_gap, all_days)],
                       index=["fill@level", "gap-aware"]).to_string())

    be = breakeven_cost(tr_gap)
    print(f"\ntotal-R breakeven round-trip cost (gap-aware base): {be:.4f} price units")

    print("\n--- cost sensitivity (gap-aware base) ---")
    rows = [headline(tr_gap, all_days, c) for c in COST_LEVELS[sym]]
    print(pd.DataFrame(rows).to_string(index=False))

    ref = COST_LEVELS[sym][2]  # mid cost level
    print(f"\n--- cost-in-R by range-width quintile (ref cost {ref}) ---")
    print(width_buckets(tr_gap, ref).to_string(index=False))

    print(f"\n--- range-window sweep (gross, flat 18) ---")
    print(window_sweep(days, all_days).head(10).to_string(index=False))

    print(f"\n--- close-hour sweep (gross, range 03-06) ---")
    print(close_sweep(days, all_days).to_string(index=False))

    print(f"\n--- SL->BE sweep (gross) ---")
    print(be_sweep(days, all_days).to_string(index=False))
    print(f"\n--- SL->BE sweep (at ref cost {ref}, gap-aware) ---")
    # be_sweep with cost on gap-aware base would need gap trades; rebuild costed
    rows = []
    for beR in [None, 0.5, 0.75, 1.0, 1.5, 2.0]:
        trx = build_trades(days, 3, 6, 18, gap_fill=True, be_R=beR)
        h = headline(trx, all_days, ref)
        rows.append(dict(be_R=("none" if beR is None else beR),
                    n_be=int((trx["reason"] == "BE").sum()),
                    **{k: h[k] for k in ("win", "avg_R", "total_R", "PF", "Sharpe", "maxDD_R")}))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for s in ["XAUUSD", "USDJPY"]:
        run_symbol(s)
