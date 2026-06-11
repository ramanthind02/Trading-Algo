r"""Run the rollover-cost experiments off the per-event feature tables.

Reads research/rollover_cost/outputs/events_{SYM}.parquet and produces:

  A_exit_profile.csv          spread + liquidity vs minutes-to-rollover  -> optimal exit timing
  B_exit_market_vs_limit.csv  exit cost: market at offset vs passive limit (fill prob)
  C_entry_strategies.csv      entry: limit at/through open, fill rate, chase-vs-skip
  D_net_savings.csv           overlay net = swap_saved - round-trip cost, vs hold-through

All costs are in BPS of notional. Convention: a market SELL to exit a long pays
half the spread; a market BUY to enter pays half the spread; a passive limit that
fills captures (rather than pays) that half-spread. Swap is price-accurate per event.

Run:
  .\.venv\Scripts\python.exe -m research.rollover_cost.experiments
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.rollover_cost.config import (
    SPECS, SYMBOLS, EXIT_OFFSETS_MIN, ENTRY_LIMIT_TICKS, ENTRY_SNAPSHOT_MIN,
    FILL_HORIZONS_MIN,
)

OUT = _REPO / "research" / "rollover_cost" / "outputs"


def _load(sym: str) -> pd.DataFrame | None:
    p = OUT / f"events_{sym}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df if len(df) else None


def _bps(price_diff: pd.Series | float, mid: pd.Series | float) -> pd.Series | float:
    return price_diff / mid * 1e4


# ---------------------------------------------------------------------------
# Experiment A — exit spread + liquidity profile vs minutes-to-rollover
# ---------------------------------------------------------------------------

def exp_A(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    mid = df["arrival_mid"]
    rows = []
    for k in EXIT_OFFSETS_MIN:
        sp = df.get(f"exit_spread_t{k}")
        rate = df.get(f"exit_rate_t{k}")
        if sp is None:
            continue
        half_bps = _bps(sp / 2.0, df[f"exit_mid_t{k}"])
        rows.append({
            "symbol": sym,
            "minutes_before_rollover": k,
            "spread_price_median": float(np.nanmedian(sp)),
            "spread_bps_median": float(np.nanmedian(_bps(sp, df[f"exit_mid_t{k}"]))),
            "spread_bps_p90": float(np.nanpercentile(_bps(sp, df[f"exit_mid_t{k}"]).dropna(), 90)) if sp.notna().any() else np.nan,
            "half_spread_bps_median": float(np.nanmedian(half_bps)),
            "tick_rate_per_min_median": float(np.nanmedian(rate)) if rate is not None else np.nan,
            "n_events": int(sp.notna().sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment B — exit cost: market at offset vs passive limit
# ---------------------------------------------------------------------------

def exp_B(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    rows = []
    # Market exit at each offset: cost = half spread at that offset (bps).
    for k in EXIT_OFFSETS_MIN:
        sp = df.get(f"exit_spread_t{k}")
        m = df.get(f"exit_mid_t{k}")
        if sp is None:
            continue
        cost = _bps(sp / 2.0, m)
        rows.append({
            "symbol": sym, "strategy": f"market_exit_T-{k}m",
            "fill_rate": 1.0,
            "cost_bps_median": float(np.nanmedian(cost)),
            "cost_bps_mean": float(np.nanmean(cost)),
            "n": int(cost.notna().sum()),
        })
    # Passive limit at arrival mid (T-15), must fill before rollover else market out at T-1.
    fillmin = df.get("exit_sell_limit_mid_fill_min")
    if fillmin is not None:
        filled = fillmin.notna()
        fr = float(filled.mean())
        # filled => captured ~0 cost (paid mid). not filled => fallback market at T-1.
        sp1 = df.get("exit_spread_t1"); m1 = df.get("exit_mid_t1")
        fallback = _bps(sp1 / 2.0, m1)
        exp_cost = np.where(filled, 0.0, fallback)
        rows.append({
            "symbol": sym, "strategy": "passive_limit_mid@T-15_fallback_mkt@T-1",
            "fill_rate": fr,
            "cost_bps_median": float(np.nanmedian(exp_cost)),
            "cost_bps_mean": float(np.nanmean(exp_cost)),
            "n": int(np.isfinite(exp_cost).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment C — entry strategies: limit at/through open, chase vs skip
# ---------------------------------------------------------------------------

def exp_C(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    spec = SPECS[sym]
    open_mid = df["open_mid"]
    open_spread = df["open_spread"]
    settled = df["entry_settled_mid"]
    rows = []

    # Baseline: market buy at the open (pay half the WIDE reopening spread).
    mkt_cost = _bps(open_spread / 2.0, open_mid)
    rows.append({
        "symbol": sym, "strategy": "market_buy_at_open", "limit_ticks_below": 0,
        "fill_rate_5m": 1.0, "fill_rate_30m": 1.0, "fill_rate_60m": 1.0,
        "cost_vs_open_bps_median": float(np.nanmedian(mkt_cost)),
        "cost_vs_open_bps_mean": float(np.nanmean(mkt_cost)),
        "adverse_sel_vs_settled_bps_median": float(np.nanmedian(_bps(open_mid + open_spread / 2.0 - settled, open_mid))),
        "n": int(mkt_cost.notna().sum()),
    })

    for n in ENTRY_LIMIT_TICKS:
        limit_price = open_mid - n * spec.tick_size
        improvement_bps = _bps(n * spec.tick_size, open_mid)  # captured if filled (paid below open)
        fr = {h: float(df.get(f"entry_buylim_t{n}_filled_{h}m", pd.Series(dtype=float)).mean())
              for h in FILL_HORIZONS_MIN}
        filled60 = df.get(f"entry_buylim_t{n}_filled_60m").astype(bool)
        # Cost vs open if filled = -improvement (price improvement). Adverse selection:
        # among fills, where is settled price vs our fill? (negative = we caught a drop).
        adverse = _bps(limit_price - settled, open_mid)        # >0 => we paid above settled (bad)
        rows.append({
            "symbol": sym, "strategy": f"limit_{n}t_below_open", "limit_ticks_below": n,
            "fill_rate_5m": fr.get(5, np.nan), "fill_rate_30m": fr.get(30, np.nan),
            "fill_rate_60m": fr.get(60, np.nan),
            "improvement_if_filled_bps": float(improvement_bps if np.isscalar(improvement_bps) else np.nanmedian(improvement_bps)),
            "adverse_sel_vs_settled_bps_median_iffilled": float(np.nanmedian(adverse[filled60])) if filled60.any() else np.nan,
            "n": int(len(df)),
        })
    return pd.DataFrame(rows)


def exp_C_policies(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Expected entry cost (bps vs open) for full policies: limit-then-chase vs limit-then-skip.

    Policy 'limit@open_mid - n, chase market @Hm if unfilled':
      cost = filled_by_H ? -improvement : market_cost_at_H (pay ask at H).
    We approximate the chase market cost by the spread snapshot at the nearest horizon.
    Policy 'limit ..., skip if unfilled': cost over FILLED only (+ report skip rate).
    """
    spec = SPECS[sym]
    open_mid = df["open_mid"]
    rows = []
    # map fill horizon to the nearest entry spread snapshot we stored
    snap_for = {1: 1, 2: 2, 5: 5, 10: 10, 15: 15, 30: 30, 60: 59}
    for n in (0, 1, 2, 3, 5):
        improvement_bps = _bps(n * spec.tick_size, open_mid)
        for H in (5, 10, 30):
            col = f"entry_buylim_t{n}_filled_{H}m"
            if col not in df:
                continue
            filled = df[col].astype(bool)
            fr = float(filled.mean())
            # chase cost: market buy at H => pay half spread at snapshot H vs open_mid,
            # PLUS any drift (mid at H - open_mid). Use stored mid/spread snapshots.
            sm = snap_for[H]
            mid_H = df.get(f"entry_mid_p{sm}")
            sp_H = df.get(f"entry_spread_p{sm}")
            chase_cost = _bps((mid_H - open_mid) + sp_H / 2.0, open_mid)
            cost_chase = np.where(filled, -improvement_bps, chase_cost)
            rows.append({
                "symbol": sym, "policy": "limit_then_chase",
                "limit_ticks_below": n, "chase_after_min": H,
                "fill_rate": fr,
                "exp_cost_vs_open_bps_median": float(np.nanmedian(cost_chase)),
                "exp_cost_vs_open_bps_mean": float(np.nanmean(cost_chase)),
                "skip_rate_if_skip_policy": float(1 - fr),
                "cost_if_filled_only_bps": float(-improvement_bps if np.isscalar(improvement_bps) else np.nanmedian(-improvement_bps)),
                "n": int(len(df)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment D — net savings vs hold-through-rollover
# ---------------------------------------------------------------------------

def exp_D(df: pd.DataFrame, sym: str, best_exit_offset: int = 7,
          entry_chase_min: int = 5) -> pd.DataFrame:
    """Net overlay benefit (bps/night) = swap_saved - round_trip_transaction_cost.

    Reports, per LONG position (the negative-carry leg), for:
      - naive: market exit @ best_exit_offset + market buy at open
      - tuned: market exit @ best_exit_offset + limit@open_mid then chase @entry_chase_min
    against the hold-through baseline (pay the long swap, zero transaction cost).
    """
    spec = SPECS[sym]
    open_mid = df["open_mid"]
    is_triple = df["is_triple"].astype(bool)

    # swap saved per night (price-accurate, abs value of the long swap; 3x on triple days)
    swap_long_bps = df.apply(lambda r: spec.swap_bps(r["open_mid"], is_long=True), axis=1).abs()
    swap_saved = np.where(is_triple, 3.0, 1.0) * swap_long_bps

    # exit cost (market at best_exit_offset)
    k = best_exit_offset if best_exit_offset in EXIT_OFFSETS_MIN else min(EXIT_OFFSETS_MIN, key=lambda x: abs(x - best_exit_offset))
    exit_cost = _bps(df[f"exit_spread_t{k}"] / 2.0, df[f"exit_mid_t{k}"])

    # entry naive: market buy at open
    entry_naive = _bps(df["open_spread"] / 2.0, open_mid)

    # entry tuned: limit @ open_mid (n=0), chase market @ entry_chase_min if unfilled
    snap_for = {5: 5, 10: 10, 30: 30}
    filled = df[f"entry_buylim_t0_filled_{entry_chase_min}m"].astype(bool)
    sm = snap_for.get(entry_chase_min, 5)
    chase_cost = _bps((df[f"entry_mid_p{sm}"] - open_mid) + df[f"entry_spread_p{sm}"] / 2.0, open_mid)
    entry_tuned = np.where(filled, 0.0, chase_cost)

    def summarize(name, rt_cost):
        net = swap_saved - rt_cost
        return {
            "symbol": sym, "policy": name,
            "swap_saved_bps_median": float(np.nanmedian(swap_saved)),
            "exit_cost_bps_median": float(np.nanmedian(exit_cost)),
            "entry_cost_bps_median": float(np.nanmedian(rt_cost - exit_cost)),
            "roundtrip_cost_bps_median": float(np.nanmedian(rt_cost)),
            "net_benefit_bps_median": float(np.nanmedian(net)),
            "net_benefit_bps_mean": float(np.nanmean(net)),
            "pct_nights_net_positive": float((net > 0).mean() * 100),
            "annualized_net_pct_252n": float(np.nanmedian(net) * 252 / 1e4 * 100),
            "n": int(len(df)),
        }

    rows = [
        summarize(f"naive_mkt_exit@T-{k}_mkt_entry@open", exit_cost + entry_naive),
        summarize(f"tuned_mkt_exit@T-{k}_limit@open_chase@{entry_chase_min}m", exit_cost + entry_tuned),
    ]
    # Reference: long buy&hold pays swap every night (the drag the overlay removes)
    rows.append({
        "symbol": sym, "policy": "REFERENCE_long_buy_hold_pays_swap",
        "swap_saved_bps_median": 0.0,
        "exit_cost_bps_median": 0.0, "entry_cost_bps_median": 0.0,
        "roundtrip_cost_bps_median": 0.0,
        "net_benefit_bps_median": float(-np.nanmedian(swap_saved)),
        "net_benefit_bps_mean": float(-np.nanmean(swap_saved)),
        "pct_nights_net_positive": 0.0,
        "annualized_net_pct_252n": float(-np.nanmedian(swap_long_bps) * 252 / 1e4 * 100),
        "n": int(len(df)),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recommendation — pick optimal exit offset + best entry policy per symbol
# ---------------------------------------------------------------------------

def recommend(df: pd.DataFrame, sym: str, a: pd.DataFrame, cp: pd.DataFrame,
              b: pd.DataFrame, min_fill_rate: float = 0.75) -> dict:
    spec = SPECS[sym]
    # Optimal exit: among offsets in [2,15] minutes, the one with the lowest median
    # half-spread (keeps flat-time short while paying the tightest spread).
    cand = a[(a.symbol == sym) & (a.minutes_before_rollover.between(2, 15))]
    best_mkt_exit = cand.loc[cand["half_spread_bps_median"].idxmin()]
    mkt_exit_offset = int(best_mkt_exit["minutes_before_rollover"])
    mkt_exit_cost = float(best_mkt_exit["half_spread_bps_median"])
    # Passive-limit exit alternative (post at mid @ T-15, market fallback @ T-1).
    # Use the MEAN cost (includes the unfilled-fallback tail) for a fair comparison.
    pl = b[(b.symbol == sym) & (b.strategy.str.startswith("passive"))]
    pl_cost = float(pl["cost_bps_mean"].iloc[0]) if len(pl) else float("inf")
    pl_fill = float(pl["fill_rate"].iloc[0]) if len(pl) else 0.0
    if pl_cost <= mkt_exit_cost:
        exit_desc = f"passive limit @ mid T-15 (fill {pl_fill:.0%}, mkt fallback)"
        exit_cost_bps = pl_cost
    else:
        exit_desc = f"market @ T-{mkt_exit_offset}m"
        exit_cost_bps = mkt_exit_cost

    # Best entry policy: lowest expected cost among policies that keep fill_rate >= threshold.
    cps = cp[(cp.symbol == sym) & (cp.fill_rate >= min_fill_rate)]
    if len(cps) == 0:
        cps = cp[cp.symbol == sym]
    best_entry = cps.loc[cps["exp_cost_vs_open_bps_median"].idxmin()]
    entry_cost_bps = float(best_entry["exp_cost_vs_open_bps_median"])

    is_triple = df["is_triple"].astype(bool)
    swap_long_bps = df.apply(lambda r: spec.swap_bps(r["open_mid"], is_long=True), axis=1).abs()
    swap_1d = float(np.nanmedian(swap_long_bps))                       # single-night swap (bps)
    swap_saved = float(np.nanmedian(np.where(is_triple, 3.0, 1.0) * swap_long_bps))
    roundtrip = exit_cost_bps + entry_cost_bps
    net = swap_saved - roundtrip

    # Fallback decision rule: only cross the spread (market order) when the half-spread
    # you'd pay is cheaper than the swap you'd save that night.
    half = lambda sp, m: float(np.nanmedian((sp / 2.0) / m * 1e4))
    exit_t1_half = half(df["exit_spread_t1"], df["exit_mid_t1"])       # cost of market-out at T-1
    entry_p15_half = half(df["entry_spread_p15"], df["entry_mid_p15"]) # decayed reopen half-spread
    exit_fallback = "market-out" if exit_t1_half < swap_1d else "eat-swap"
    entry_fallback = "chase" if entry_p15_half < swap_1d else "skip"

    # Conservative net: eat-swap exit fallback (miss => net 0) and ZERO entry credit
    # (assume the limit-below-open improvement is fully offset by adverse selection).
    net_conservative = pl_fill * (swap_saved - exit_cost_bps)

    return {
        "symbol": sym,
        "exit": exit_desc,
        "exit_cost_bps": round(exit_cost_bps, 3),
        "exit_fallback_on_miss": exit_fallback,
        "entry": f"limit {int(best_entry['limit_ticks_below'])}t below open, chase @ {int(best_entry['chase_after_min'])}m",
        "entry_fill_rate": round(float(best_entry["fill_rate"]), 3),
        "entry_cost_bps": round(entry_cost_bps, 3),
        "entry_fallback_on_miss": entry_fallback,
        "swap_1night_bps": round(swap_1d, 3),
        "swap_saved_bps_median": round(swap_saved, 3),
        "roundtrip_cost_bps": round(roundtrip, 3),
        "net_expected_bps_per_night": round(net, 3),
        "net_conservative_bps_per_night": round(net_conservative, 3),
        "net_expected_pct_yr": round(net * 252 / 1e4 * 100, 2),
        "net_conservative_pct_yr": round(net_conservative * 252 / 1e4 * 100, 2),
        "buyhold_swap_drag_pct_yr": round(-swap_1d * 252 / 1e4 * 100, 2),
        "n_events": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    A, B, C, Cp, D = [], [], [], [], []
    for sym in SYMBOLS:
        df = _load(sym)
        if df is None:
            print(f"[skip] {sym}: no events yet"); continue
        print(f"[{sym}] {len(df)} events  ({df['rollover'].dt.date.min()} .. {df['rollover'].dt.date.max()})")
        A.append(exp_A(df, sym))
        B.append(exp_B(df, sym))
        C.append(exp_C(df, sym))
        Cp.append(exp_C_policies(df, sym))
        D.append(exp_D(df, sym))

    if not A:
        print("No event tables found — run build_events first."); return

    dfa = pd.concat(A, ignore_index=True); dfa.to_csv(OUT / "A_exit_profile.csv", index=False)
    dfb = pd.concat(B, ignore_index=True); dfb.to_csv(OUT / "B_exit_market_vs_limit.csv", index=False)
    dfc = pd.concat(C, ignore_index=True); dfc.to_csv(OUT / "C_entry_strategies.csv", index=False)
    dfcp = pd.concat(Cp, ignore_index=True); dfcp.to_csv(OUT / "C_entry_policies.csv", index=False)
    dfd = pd.concat(D, ignore_index=True); dfd.to_csv(OUT / "D_net_savings.csv", index=False)

    # Recommendation summary (the headline deliverable).
    recs = []
    for sym in SYMBOLS:
        df = _load(sym)
        if df is None:
            continue
        recs.append(recommend(df, sym, dfa, dfcp, dfb))
    dfr = pd.DataFrame(recs)
    dfr.to_csv(OUT / "RECOMMENDATION.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
    print("\n=== D: NET SAVINGS (bps/night, LONG leg) ===")
    print(dfd[["symbol", "policy", "swap_saved_bps_median", "roundtrip_cost_bps_median",
               "net_benefit_bps_median", "pct_nights_net_positive", "annualized_net_pct_252n"]].to_string(index=False))
    print("\n=== RECOMMENDED POLICY PER SYMBOL ===")
    print(dfr.to_string(index=False))
    print("\nWrote CSVs to", OUT)


if __name__ == "__main__":
    main()
