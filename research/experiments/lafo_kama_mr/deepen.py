"""Phase-0 deepening: map the long-only deep-dip plateau, per-year regime breakdown,
and the make-or-break adversarial test — is it intraday TIMING or just buy-the-dip beta?

The random-entry null: on exactly the days the real strategy trades, place a long
entry at a RANDOM eligible RTH bar (same revert/stop/eod exit logic). If the real
dip-timed Sharpe sits well above the null distribution, the KAMA-deviation ENTRY adds
value beyond "be long intraday on a high-vol day". Also benchmark vs always-long RTH.

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.deepen
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E

SYM, Y0, Y1 = "NDX", 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"
TD = E.TRADING_DAYS


# --------------------------------------------------------------------------- #
def daily_series(trades: pd.DataFrame, session_days: np.ndarray, col="ret") -> pd.Series:
    grid = pd.Series(0.0, index=pd.Index(session_days, name="day"))
    if not trades.empty:
        dr = trades.groupby("day")[col].sum()
        grid.loc[dr.index] = dr.values
    return grid


def sharpe(arr: np.ndarray) -> float:
    return float(arr.mean() / arr.std() * np.sqrt(TD)) if arr.std() > 0 else 0.0


def per_year(trades: pd.DataFrame, session_days: np.ndarray) -> pd.DataFrame:
    s = daily_series(trades, session_days)
    df = s.to_frame("ret")
    df["year"] = pd.DatetimeIndex(df.index).year
    rows = []
    for y, g in df.groupby("year"):
        r = g["ret"].to_numpy()
        eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
        n_tr = int((trades["day"].dt.year == y).sum()) if not trades.empty else 0
        rows.append({"year": y, "sharpe": round(sharpe(r), 2),
                     "ret%": round(100 * r.sum(), 1), "maxDD%": round(100 * dd, 1),
                     "trades": n_tr})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Random-entry null (long-only; same exit machinery as engine.simulate)
# --------------------------------------------------------------------------- #
def _long_exit(arrays, ej, sl, p, n, days):
    o, h, lo, c, kama_v, day = arrays
    j = ej
    held = 0
    while True:
        last_of_day = (j == n - 1) or (day[j + 1] != day[j])
        if lo[j] <= sl:
            return sl, "sl", j, held
        if p.exit_mode == "revert" and c[j] >= kama_v[j]:
            return c[j], "revert", j, held
        if p.exit_mode == "close" and last_of_day:
            return c[j], "eod", j, held
        if last_of_day:
            return c[j], "eod", j, held
        j += 1
        held += 1


def random_entry_null(bars: pd.DataFrame, real_trades: pd.DataFrame, p: Params_like,
                      n_seeds: int = 200) -> dict:
    """Sharpe distribution if, on the SAME days, we entered long at a random RTH bar."""
    o = bars["open"].to_numpy(); h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy(); c = bars["close"].to_numpy()
    kama_v = bars["kama"].to_numpy(); atr = bars["atr"].to_numpy()
    day = bars["day"].to_numpy()
    arrays = (o, h, lo, c, kama_v, day)
    n = len(bars)
    session_days = np.sort(bars["day"].unique())
    # map day -> list of eligible bar indices (have a next-same-day bar, finite atr, warmup)
    warmup = p.kama_period + 10
    by_day: dict = {}
    for i in range(warmup, n - 1):
        if day[i + 1] == day[i] and np.isfinite(atr[i]) and atr[i] > 0:
            by_day.setdefault(day[i], []).append(i)
    trade_days = real_trades["day"].unique() if not real_trades.empty else []
    nulls = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        rets = {}
        for d in trade_days:
            elig = by_day.get(d)
            if not elig:
                continue
            i = int(rng.choice(elig))
            ej = i + 1
            entry = o[ej]
            sl = entry - p.stop_atr_mult * atr[i]
            exit_px, _, _, _ = _long_exit(arrays, ej, sl, p, n, day)
            rets[d] = rets.get(d, 0.0) + (exit_px - entry) / entry
        grid = pd.Series(0.0, index=pd.Index(session_days))
        for d, r in rets.items():
            grid.loc[d] = r
        nulls.append(sharpe(grid.to_numpy()))
    nulls = np.array(nulls)
    return {"null_mean": round(float(nulls.mean()), 3),
            "null_p95": round(float(np.percentile(nulls, 95)), 3),
            "null_max": round(float(nulls.max()), 3),
            "null_std": round(float(nulls.std()), 3)}


class Params_like:  # typing alias for clarity
    pass


# --------------------------------------------------------------------------- #
def buyhold_rth(bars: pd.DataFrame) -> dict:
    """Always-long-during-RTH benchmark: daily open->close return, flat overnight."""
    g = bars.groupby("day").agg(o=("open", "first"), c=("close", "last"))
    r = (g["c"] / g["o"] - 1.0).to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"sharpe": round(sharpe(r), 2), "ann%": round(100 * r.mean() * TD, 2),
            "maxDD%": round(100 * dd, 1), "tot%": round(100 * r.sum(), 1)}


def corr_to_bh(trades: pd.DataFrame, bars: pd.DataFrame, session_days: np.ndarray) -> float:
    s = daily_series(trades, session_days)
    g = bars.groupby("day").agg(o=("open", "first"), c=("close", "last"))
    bh = (g["c"] / g["o"] - 1.0).reindex(session_days).fillna(0.0)
    if s.std() == 0:
        return 0.0
    return round(float(np.corrcoef(s.to_numpy(), bh.to_numpy())[0, 1]), 3)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

    # ---- A. extended long-only plateau: threshold x stop x exit x tf ----------
    print("=" * 100)
    print("A. LONG-ONLY plateau — threshold x stop_mult x exit x tf  (FRICTIONLESS)")
    print("=" * 100)
    rows = []
    for tf in ("M5", "M15"):
        for thr in (0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03):
            for sm in (2.5, 3.5, 4.5):
                for ex in ("revert", "close"):
                    p = E.Params(tf=tf, session="rth", entry_mode="rel", threshold=thr,
                                 exit_mode=ex, stop_atr_mult=sm, longs=True, shorts=False)
                    m, _ = E.run(SYM, Y0, Y1, p, f"{tf} thr={thr} sm={sm} {ex}")
                    m.update(dict(tf=tf, thr=thr, sm=sm, exit=ex)); rows.append(m)
    A = pd.DataFrame(rows)
    A.to_csv(f"{OUT}/deepen_A_long_plateau.csv", index=False)
    cols = ["tf", "thr", "sm", "exit", "sharpe", "ann_ret%", "maxDD%", "trades", "tr/yr",
            "win%", "avgR", "PF", "avg_hold"]
    print(A.sort_values("sharpe", ascending=False)[cols].head(24).to_string(index=False))

    # ---- B. z-entry long-only plateau ----------------------------------------
    print("\n" + "=" * 100)
    print("B. LONG-ONLY z(delta) entry plateau — zthr x zlb x tf x exit  (FRICTIONLESS)")
    print("=" * 100)
    rows = []
    for tf in ("M5", "M15"):
        for zthr in (2.5, 3.0, 3.5, 4.0):
            for zlb in (100, 200):
                for ex in ("revert", "close"):
                    p = E.Params(tf=tf, session="rth", entry_mode="z", z_thr=zthr, z_lookback=zlb,
                                 exit_mode=ex, stop_atr_mult=3.5, longs=True, shorts=False)
                    m, _ = E.run(SYM, Y0, Y1, p, f"{tf} z={zthr} lb={zlb} {ex}")
                    m.update(dict(tf=tf, zthr=zthr, zlb=zlb, exit=ex)); rows.append(m)
    B = pd.DataFrame(rows)
    B.to_csv(f"{OUT}/deepen_B_zentry_plateau.csv", index=False)
    cols = ["tf", "zthr", "zlb", "exit", "sharpe", "ann_ret%", "maxDD%", "trades", "tr/yr",
            "win%", "avgR", "PF", "avg_hold"]
    print(B.sort_values("sharpe", ascending=False)[cols].head(20).to_string(index=False))

    # ---- C. per-year + drift-null + buy&hold for the leading candidates -------
    candidates = {
        "fixed2.0 M5 revert":  E.Params(tf="M5", session="rth", entry_mode="rel", threshold=0.02,
                                        exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False),
        "fixed1.5 M5 revert":  E.Params(tf="M5", session="rth", entry_mode="rel", threshold=0.015,
                                        exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False),
        "z3.0 M5 lb100 revert": E.Params(tf="M5", session="rth", entry_mode="z", z_thr=3.0, z_lookback=100,
                                         exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False),
    }
    print("\n" + "=" * 100)
    print("C. PER-YEAR + RANDOM-ENTRY NULL + BUY&HOLD  (is it timing or beta?)")
    print("=" * 100)
    for name, p in candidates.items():
        bars = E.prep_bars(SYM, Y0, Y1, p)
        _, tr = E.run(SYM, Y0, Y1, p, name)
        sd = np.sort(bars["day"].unique())
        # attach Params_like fields the null needs
        pl = Params_like()
        pl.kama_period = p.kama_period; pl.stop_atr_mult = p.stop_atr_mult; pl.exit_mode = p.exit_mode
        null = random_entry_null(bars, tr, pl, n_seeds=200)
        real_sh = sharpe(daily_series(tr, sd).to_numpy())
        bh = buyhold_rth(bars)
        c = corr_to_bh(tr, bars, sd)
        print(f"\n--- {name} ---")
        print(f"  REAL Sharpe = {real_sh:.2f} | random-entry null mean={null['null_mean']} "
              f"p95={null['null_p95']} max={null['null_max']} | edge over p95 = {real_sh - null['null_p95']:.2f}")
        print(f"  buy&hold-RTH Sharpe={bh['sharpe']} ann%={bh['ann%']} maxDD%={bh['maxDD%']} | corr(strat,BH)={c}")
        print(per_year(tr, sd).to_string(index=False))
