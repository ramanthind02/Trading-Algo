"""
Statistical-rigor layer for the Concretum intraday-momentum robustness study.
Closes the gaps the adversarial verification flagged:
  1. Block-bootstrap 95% CI on the headline Sharpe per window (full / paper / post).
  2. Bootstrap p-value on the paper-era -> post-publication Sharpe DECAY (is it real?).
  3. Net-of-cost escalation: the OOS persistence at a defensible 1.0 bps one-way cost.
  4. Rolling 252-day Sharpe distribution (stability), reported per instrument.

Usage: python -m research.experiments.concretum_intraday_momentum.analyze
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import run_matrix as RM

OUT = RM.OUT
PAPER_END = RM.PAPER_END
POST_START = RM.POST_START
SYMS = ["SPY", "QQQ", "SP500", "NDX"]
RNG = np.random.default_rng(12345)


def block_boot_sharpe(r: np.ndarray, block: int = 10, n: int = 4000) -> np.ndarray:
    r = r[np.isfinite(r)]
    T = len(r)
    if T < block + 5:
        return np.array([0.0])
    nb = int(np.ceil(T / block))
    starts = RNG.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    s = r[idx]
    mu = s.mean(axis=1)
    sd = s.std(axis=1)
    return np.where(sd > 0, mu / sd * np.sqrt(E.TRADING_DAYS), 0.0)


def headline_series(sym: str, cost_bps: float = 0.0) -> pd.Series:
    win = E.load_sessions(sym, RM.DATA_START[sym], 2026)
    dt = E.build_day_table(win, E.MARKS_ET_SEMI)
    p = E.Params(stop_mode="curr_vwap", sizing="vol_target", cost_bps_per_turn=cost_bps)
    res = E.simulate(dt, p, len(E.MARKS_ET_SEMI))
    col = "net_lev" if cost_bps > 0 else "lev_ret"
    s = res.loc[res["valid"], col]
    return s[np.isfinite(s)]


def win_slice(s: pd.Series, lo=None, hi=None) -> np.ndarray:
    x = s
    if lo is not None:
        x = x[x.index >= lo]
    if hi is not None:
        x = x[x.index <= hi]
    return x.to_numpy()


def sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(E.TRADING_DAYS)) if r.std() > 0 else 0.0


def main():
    boot_rows, net_rows, roll_rows = [], [], []
    for sym in SYMS:
        s0 = headline_series(sym, 0.0)        # gross
        s1 = headline_series(sym, 1.0)        # net @ 1.0 bps one-way
        windows = {"full": (None, None), "paper": (None, PAPER_END), "post": (POST_START, None)}
        boot = {}
        for w, (lo, hi) in windows.items():
            r = win_slice(s0, lo, hi)
            b = block_boot_sharpe(r)
            boot[w] = b
            boot_rows.append({
                "sym": sym, "window": w, "days": len(r),
                "sharpe": round(sharpe(r), 3),
                "ci_lo": round(float(np.percentile(b, 2.5)), 3),
                "ci_hi": round(float(np.percentile(b, 97.5)), 3),
                "p_sharpe>0": round(float((b > 0).mean()), 3),
            })
        # decay test: paired bootstrap delta = SR_paper - SR_post; p = P(no decay) = P(post>=paper)
        m = min(len(boot["paper"]), len(boot["post"]))
        delta = boot["paper"][:m] - boot["post"][:m]
        p_no_decay = float((delta <= 0).mean())
        boot_rows.append({"sym": sym, "window": "DECAY(paper-post)", "days": 0,
                          "sharpe": round(float(np.median(delta)), 3),
                          "ci_lo": round(float(np.percentile(delta, 2.5)), 3),
                          "ci_hi": round(float(np.percentile(delta, 97.5)), 3),
                          "p_sharpe>0": round(1 - p_no_decay, 3)})  # P(real decay)

        # net-of-cost escalation (paper vs post) at 1 bps
        for w, (lo, hi) in windows.items():
            net_rows.append({"sym": sym, "window": w,
                             "gross_sharpe": round(sharpe(win_slice(s0, lo, hi)), 3),
                             "net1bps_sharpe": round(sharpe(win_slice(s1, lo, hi)), 3)})

        # rolling 252d Sharpe distribution on the gross headline series
        roll = s0.rolling(252).apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0, raw=True).dropna()
        rv = roll.to_numpy()
        roll_rows.append({"sym": sym, "n_windows": len(rv),
                          "roll_sharpe_p05": round(float(np.percentile(rv, 5)), 2),
                          "roll_sharpe_med": round(float(np.percentile(rv, 50)), 2),
                          "roll_sharpe_p95": round(float(np.percentile(rv, 95)), 2),
                          "pct_windows>0": round(100 * float((rv > 0).mean()), 1),
                          "worst_roll12m": round(float(rv.min()), 2)})

    bdf = pd.DataFrame(boot_rows); bdf.to_csv(OUT / "bootstrap_significance.csv", index=False)
    ndf = pd.DataFrame(net_rows); ndf.to_csv(OUT / "net_escalation.csv", index=False)
    rdf = pd.DataFrame(roll_rows); rdf.to_csv(OUT / "rolling_stability.csv", index=False)

    pd.set_option("display.width", 200)
    print("===== BOOTSTRAP Sharpe CIs + OOS decay significance (headline vol-target, gross) =====")
    print(bdf.to_string(index=False))
    print("\n  DECAY row: 'sharpe'=median(paper-post) Sharpe drop; 'p_sharpe>0'=P(decay is real); ci on the drop")
    print("\n===== NET-OF-COST escalation: gross vs net @ 1.0 bps one-way =====")
    print(ndf.to_string(index=False))
    print("\n===== ROLLING 252d Sharpe distribution (stability) =====")
    print(rdf.to_string(index=False))


if __name__ == "__main__":
    main()
