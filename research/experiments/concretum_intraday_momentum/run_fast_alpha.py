"""
Robustness of the QuanTips #2 fast-alpha execution overlay.

Usage: python -m research.experiments.concretum_intraday_momentum.run_fast_alpha

Writes outputs/fa_*.csv:
  fa_streaks      streak-conditional next-bar returns (fast-alpha existence, per sym)
  fa_standalone   fast alpha gross vs net (informational-not-monetizable, per sym)
  fa_overlay      baseline vs {entry-only, exit-only, both, PLACEBO} x {gross, net@1bps}
                  x {full, paper<=2024-04, post>=2024-05}  (the core robustness panel)
  fa_cost         baseline vs full-overlay net Sharpe across one-way bps (does overlay lift breakeven?)
  fa_atr          ATR-multiplier sensitivity (band width plateau)
  fa_boot         paired block-bootstrap on the overlay - baseline net-Sharpe difference (significance)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import fast_alpha as FA
from research.experiments.concretum_intraday_momentum import run_matrix as RM

OUT = RM.OUT
PAPER_END, POST_START = RM.PAPER_END, RM.POST_START
SYMS = ["SPY", "QQQ", "SP500", "NDX"]
RNG = np.random.default_rng(7)
COST = 1.0   # one-way bps for the net comparison (~IBKR $0.0035/share on SPY)


def _load():
    data = {}
    for s in SYMS:
        data[s] = FA.load_5m(s, RM.DATA_START[s], 2026)
    return data


def streaks_standalone(data):
    srows, arows = [], []
    for s in SYMS:
        b, _ = data[s]
        t = FA.fast_alpha_streaks(b); t["sym"] = s; srows.append(t)
        for bps in [0.0, COST]:
            r = FA.fast_alpha_standalone(b, bps); r["sym"] = s; r["cost_bps"] = bps; arows.append(r)
    pd.concat(srows).to_csv(OUT / "fa_streaks.csv", index=False)
    pd.DataFrame(arows).to_csv(OUT / "fa_standalone.csv", index=False)


def overlay_panel(data):
    variants = [
        ("baseline", dict(overlay_entry=False, overlay_exit=False)),
        ("entry_only", dict(overlay_entry=True, overlay_exit=False)),
        ("exit_only", dict(overlay_entry=False, overlay_exit=True)),
        ("both", dict(overlay_entry=True, overlay_exit=True)),
        ("placebo_both", dict(overlay_entry=True, overlay_exit=True, placebo=True)),
    ]
    windows = [("full", None, None), ("paper", None, PAPER_END), ("post", POST_START, None)]
    rows = []
    for s in SYMS:
        b, d = data[s]
        for label, kw in variants:
            res = FA.simulate_bo(b, d, FA.BParams(sizing="vol_target", cost_bps_oneway=COST, **kw))
            for wl, lo, hi in windows:
                for gn, col in [("gross", "ret"), ("net1bps", "net")]:
                    m = FA.bo_metrics(res, col, lo, hi)
                    m.update(sym=s, variant=label, window=wl, pnl=gn)
                    rows.append(m)
    pd.DataFrame(rows).to_csv(OUT / "fa_overlay.csv", index=False)


def cost_sweep(data):
    rows = []
    for s in SYMS:
        b, d = data[s]
        for label, kw in [("baseline", {}), ("overlay_both", dict(overlay_entry=True, overlay_exit=True))]:
            for bps in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                res = FA.simulate_bo(b, d, FA.BParams(sizing="vol_target", cost_bps_oneway=bps, **kw))
                m = FA.bo_metrics(res, "net")
                rows.append({"sym": s, "variant": label, "oneway_bps": bps, "sharpe": m["sharpe"]})
    pd.DataFrame(rows).to_csv(OUT / "fa_cost.csv", index=False)


def atr_sensitivity(data):
    rows = []
    for s in SYMS:
        b, d = data[s]
        for am in [0.25, 0.5, 0.75, 1.0, 1.5]:
            for label, kw in [("baseline", {}), ("overlay_both", dict(overlay_entry=True, overlay_exit=True))]:
                res = FA.simulate_bo(b, d, FA.BParams(atr_mult=am, sizing="vol_target", cost_bps_oneway=COST, **kw))
                m = FA.bo_metrics(res, "net")
                rows.append({"sym": s, "atr_mult": am, "variant": label, "net_sharpe": m["sharpe"]})
    pd.DataFrame(rows).to_csv(OUT / "fa_atr.csv", index=False)


def boot_diff(data):
    """Paired block-bootstrap on (overlay_both - baseline) net daily-return Sharpe difference."""
    def blocks(idx, T, block, n):
        nb = int(np.ceil(T / block))
        st = RNG.integers(0, T - block + 1, size=(n, nb))
        return (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    rows = []
    for s in SYMS:
        b, d = data[s]
        rb = FA.simulate_bo(b, d, FA.BParams(sizing="vol_target", cost_bps_oneway=COST))["net"]
        ro = FA.simulate_bo(b, d, FA.BParams(sizing="vol_target", cost_bps_oneway=COST,
                                             overlay_entry=True, overlay_exit=True))["net"]
        df = pd.concat([rb.rename("base"), ro.rename("ovl")], axis=1).dropna()
        base = df["base"].to_numpy(); ovl = df["ovl"].to_numpy()
        T = len(base); idx = blocks(None, T, 20, 3000)
        def sh(x):
            sd = x.std(axis=1); return np.where(sd > 0, x.mean(axis=1) / sd * np.sqrt(252), 0.0)
        d_sh = sh(ovl[idx]) - sh(base[idx])
        rows.append({"sym": s,
                     "base_sharpe": round(float(base.mean() / base.std() * np.sqrt(252)), 3),
                     "ovl_sharpe": round(float(ovl.mean() / ovl.std() * np.sqrt(252)), 3),
                     "delta_med": round(float(np.median(d_sh)), 3),
                     "delta_ci_lo": round(float(np.percentile(d_sh, 2.5)), 3),
                     "delta_ci_hi": round(float(np.percentile(d_sh, 97.5)), 3),
                     "P(overlay>base)": round(float((d_sh > 0).mean()), 3)})
    pd.DataFrame(rows).to_csv(OUT / "fa_boot.csv", index=False)


def main():
    data = _load()
    print("loaded 5-min bars for", SYMS)
    streaks_standalone(data); print("  fa_streaks, fa_standalone")
    overlay_panel(data); print("  fa_overlay")
    cost_sweep(data); print("  fa_cost")
    atr_sensitivity(data); print("  fa_atr")
    boot_diff(data); print("  fa_boot")


if __name__ == "__main__":
    main()
