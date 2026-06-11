"""Vault-intraday frictionless screen + robustness + cost haircut.

Runs the 6 plausible vaulted node families on intraday timeframes (H4/H2/H1/M30)
resampled from MT5 M1, frictionless, lookahead-guarded. Reports gross effectiveness,
a robustness battery (param plateau via the TF×axis grid, sub-period split, sign-flip,
circular-shift null), and a measured-spread cost haircut.

Run (venv):
  .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.run_screen --mode control
  .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.run_screen --mode screen
  .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.run_screen           # both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lib.core.enums import Ticker, TimeFrame
from research.experiments.vault_intraday import data_io, engine

OUT = Path(__file__).resolve().parent / "outputs"
TFS = ["H4", "H2", "H1", "M30"]

# symbol -> (Ticker label for node construction [signal is ticker-independent], native?)
INSTR_TICKER = {
    "SP500": Ticker.ES,
    "NDX": Ticker.NQ,
    "XTIUSD": Ticker.CL,
    "XAUUSD": Ticker.GC,
    "XAGUSD": Ticker.SI,
    "TLT": Ticker.ES,  # TLT not in the Ticker enum; label-only, does not affect signal
}

# Known true price increments (probed from the live terminal; see
# data_platform/nautilus/ingest.py docstrings). Used to validate empirical inference.
KNOWN_POINT = {"SP500": 0.1, "NDX": 0.1, "XAUUSD": 0.01}

# ---- family configs: vaulted module + direction + a 1-D param scan (vaulted value
#      sits inside each axis) holding the other vaulted params fixed. ---------------
FAMILIES = [
    {
        # CL-faithful: vaulted cl_breakout is strategy_mode=long_short, lookback=55.
        "name": "robust_trend_breakout_ls",
        "module": "robust_trend_breakout",
        "direction": "long_short",
        "instruments": ["XTIUSD", "XAUUSD", "XAGUSD", "SP500", "NDX"],
        "axis": "lookback",
        "values": [10, 20, 40, 55, 60, 80, 120],
        "fixed": {"ema_period": 126, "atr_mult": 1.5, "atr_len": 14, "atr_vol_filter": 0,
                  "cooldown_bars": 0, "strategy_mode": "long_short"},
        "vaulted_native": "CL=55 (long_short)",
    },
    {
        # GC-faithful: vaulted gc_breakout is strategy 'long' (node default mode), lookback=20.
        "name": "robust_trend_breakout_long",
        "module": "robust_trend_breakout",
        "direction": "long",
        "instruments": ["XTIUSD", "XAUUSD", "XAGUSD", "SP500", "NDX"],
        "axis": "lookback",
        "values": [10, 20, 40, 55, 60, 80, 120],
        "fixed": {"ema_period": 126, "atr_mult": 1.5, "atr_len": 14, "atr_vol_filter": 0,
                  "cooldown_bars": 0, "strategy_mode": "long"},
        "vaulted_native": "GC=20 (long)",
    },
    {
        "name": "donchian_breakout",
        "module": "donchian_breakout_signal",
        "direction": "long_short",
        "instruments": ["XAGUSD", "XAUUSD", "XTIUSD", "SP500", "NDX"],
        "axis": "lookback",
        "values": [10, 20, 40, 60, 80, 120],
        "fixed": {"exit_bars": 5},
        "vaulted_native": "SI=20",
    },
    {
        "name": "regime_lrsi",
        "module": "regime_lrsi_signal",
        "direction": "long_short",
        "instruments": ["XTIUSD", "XAUUSD", "XAGUSD", "SP500", "NDX"],
        "axis": "regime_ma_period",
        "values": [20, 40, 60, 78, 100, 140],
        "fixed": {"hl_sum_period": 76, "hl_range_period": 64, "mr_avg_period": 29,
                  "mr_prank_period": 56, "long_threshold": 0.6767, "short_threshold": 0.3233,
                  "exit_bars": 1, "stop_loss_ticks": 0, "tick_size": 0.01, "strategy_mode": "long_short"},
        "vaulted_native": "CL=78",
    },
    {
        "name": "algomatic_rsi2",
        "module": "algomatic_momentum_signal",
        "direction": "long",
        "instruments": ["NDX", "SP500", "XAUUSD", "XTIUSD"],
        "axis": "momentum_lookback",
        "values": [5, 10, 20, 40, 60, 80],
        "fixed": {"rsi_period": 2, "rsi_max": 90.0, "exit_bars": 5},
        "vaulted_native": "NQ=10",
    },
    {
        "name": "ibs_lower_band",
        "module": "ibs_lower_band",
        "direction": "long",
        "instruments": ["NDX", "SP500", "TLT", "XAUUSD"],
        "axis": "hl_mean_lookback",
        "values": [10, 25, 50, 75, 100, 150],
        "fixed": {"band_high_lookback": 10, "band_width_mult": 2.5, "ibs_entry_max": 0.3},
        "vaulted_native": "NQ=25",
    },
    {
        "name": "double7s",
        "module": "double7s",
        "direction": "long",
        "instruments": ["SP500", "NDX", "TLT"],
        "axis": "short_period",
        "values": [5, 7, 10, 15, 20, 30],
        "fixed": {"ma_period": 200},
        "vaulted_native": "ES/NQ short=10 ma=200",
    },
]


def infer_point(symbol: str) -> float:
    """Empirical price increment = smallest positive step between distinct closes."""
    bars = data_io.load_bars(symbol, "H1")
    closes = np.unique(np.round(bars["close"].to_numpy(), 6))
    diffs = np.diff(closes)
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.01
    # robust small-step estimate: the 1st percentile of positive diffs
    return float(np.round(np.percentile(diffs, 1), 6))


def _half_spread_return(bars: pd.DataFrame, point: float) -> float:
    """Typical half-spread WHEN TRADING, as a log-return fraction = spread*point/2/close.

    Conditioned on a *live* quote (spread > 0): zero-spread bars are dead/illiquid
    minutes (e.g. SP500 is ~65% zero-spread at M1) and must not pull the cost of an
    actual fill to zero — otherwise the cost haircut is silently a no-op.
    """
    hs = (bars["spread"].to_numpy() * point / 2.0) / bars["close"].to_numpy()
    hs = hs[np.isfinite(hs) & (hs > 0.0)]
    return float(np.median(hs)) if hs.size else 0.0


# --------------------------------------------------------------------------- control
def run_control() -> None:
    """Prove the engine's streamed signal == the canonical BaseModel.predict path on
    identical Daily CFD bars (same node, same params) for a vaulted config."""
    from data_platform.providers.mt5.cfd_candles import load_cfd_candles_raw
    from ensemble.ensemble_utils import create_base_model_from_config

    print("=== CONTROL: engine vs BaseModel.predict (CL daily, vaulted robust_trend_breakout) ===")
    daily = load_cfd_candles_raw(Ticker.CL, TimeFrame.D)
    params = {"lookback": 55, "ema_period": 126, "atr_mult": 1.5, "atr_len": 14,
              "atr_vol_filter": 0, "cooldown_bars": 0, "strategy_mode": "long_short"}

    eng = engine.build_signals_fanout("robust_trend_breakout", Ticker.CL, [params], daily)
    eng_sig = list(eng.values())[0]

    cfg = {
        "name": "ctrl",
        "model_type": "signed_signal",
        "feature_column": "ctrl",
        "strategy": "long_short",
        "tickers": ["CL"],
        "bias_node_spec": {"module_name": "robust_trend_breakout", "timeframes": ["D"], "params": params},
    }
    bm = create_base_model_from_config(cfg, ticker=Ticker.CL, use_cache=False)
    cdf = daily.copy()
    cdf["ticker"] = Ticker.CL
    cdf["tf"] = TimeFrame.D
    bm_sig = bm.predict(cdf)

    joined = pd.DataFrame({"eng": eng_sig}).join(pd.DataFrame({"bm": bm_sig.values}, index=bm_sig.index), how="inner")
    match = np.allclose(joined["eng"].to_numpy(), joined["bm"].to_numpy(), atol=1e-9)
    nz = int((joined["eng"].abs() > 0).sum())
    print(f"  n_aligned={len(joined)}  nonzero_signal_bars={nz}  MATCH={match}")
    bpy = data_io.bars_per_year(pd.DatetimeIndex(daily['datetime']))
    res = engine.evaluate(eng_sig, pd.Series(daily['close'].values, index=pd.DatetimeIndex(daily['datetime'])), bpy)
    print(f"  daily gross sharpe={res.metrics['sharpe']:.3f}  (sanity anchor; bpy={bpy:.0f})")
    if not match:
        raise SystemExit("CONTROL FAILED: engine signal != BaseModel.predict — fix before trusting intraday")
    print("  CONTROL PASSED\n")


# ---------------------------------------------------------------------------- screen
def run_screen() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points = {s: (KNOWN_POINT.get(s) or infer_point(s)) for s in INSTR_TICKER}
    # validate inference method against the known points
    for s, known in KNOWN_POINT.items():
        est = infer_point(s)
        print(f"  point[{s}] known={known} inferred={est}")
    print(f"  points used: {points}\n")

    rows: list[dict] = []
    split = pd.Timestamp("2022-01-01")
    for fam in FAMILIES:
        for sym in fam["instruments"]:
            ticker = INSTR_TICKER[sym]
            point = points[sym]
            for tf in TFS:
                bars = data_io.load_bars(sym, tf)
                idx = pd.DatetimeIndex(bars["datetime"])
                close = pd.Series(bars["close"].to_numpy(), index=idx)
                bpy = data_io.bars_per_year(idx)
                hs_ret = _half_spread_return(bars, point)
                param_sets = [{**fam["fixed"], fam["axis"]: v} for v in fam["values"]]
                sigs = engine.build_signals_fanout(fam["module"], ticker, param_sets, bars)
                for v, params in zip(fam["values"], param_sets):
                    raw = sigs[engine._frozen_key(params)]
                    sig = engine.project(raw, fam["direction"])
                    r = engine.evaluate(sig, close, bpy)
                    m = r.metrics
                    pos = sig.shift(1)
                    # sub-period split
                    mask1 = idx < split
                    s1 = engine.evaluate(sig[mask1], close[mask1], data_io.bars_per_year(idx[mask1])).metrics["sharpe"]
                    s2 = engine.evaluate(sig[~mask1], close[~mask1], data_io.bars_per_year(idx[~mask1])).metrics["sharpe"]
                    # drift-stripped circular-shift null (95th pct |Sharpe| over many shifts).
                    # A genuine timing edge should sit well above this. (Sign-flip Sharpe is
                    # omitted: pnl(-sig) == -pnl(sig) so it is identically -gross, no information.)
                    null = engine.circular_shift_null(sig, close, bpy)
                    # cost haircut (per-fill half-spread)
                    net = {f"net_sharpe_{mult}x": engine.net_sharpe_after_cost(r.pnl, pos, hs_ret * mult, bpy)
                           for mult in (0.5, 1.0, 2.0)}
                    rows.append({
                        "family": fam["name"], "module": fam["module"], "instrument": sym,
                        "native": fam["vaulted_native"], "direction": fam["direction"],
                        "tf": tf, "axis": fam["axis"], "axis_value": v,
                        **{k: m[k] for k in ("sharpe", "ann_ret", "ann_vol", "max_dd", "turnover",
                                             "n_trades", "pct_in_mkt", "sharpe_lookahead", "n_bars")},
                        "sharpe_2018_2021": s1, "sharpe_2022_2026": s2,
                        "sharpe_shuffle_null": null,
                        "half_spread_ret": hs_ret, **net,
                    })
            print(f"  done {fam['name']:22s} {sym}")
    df = pd.DataFrame(rows)
    master = OUT / "screen_master.csv"
    df.to_csv(master, index=False)
    print(f"\nwrote {master}  ({len(df)} rows)")

    # plateau pivots (TF x axis_value gross sharpe) per family/instrument
    piv = OUT / "plateau_pivots.csv"
    with open(piv, "w") as fh:
        for (fam, sym), g in df.groupby(["family", "instrument"]):
            fh.write(f"# {fam} | {sym} | gross sharpe (rows=TF, cols={g['axis'].iloc[0]})\n")
            p = g.pivot_table(index="tf", columns="axis_value", values="sharpe").reindex(TFS)
            fh.write(p.round(2).to_csv())
            fh.write("\n")
    print(f"wrote {piv}")

    # concise leaderboard
    print("\n=== TOP gross-sharpe configs (any TF/param), net@1x in () ===")
    top = df.sort_values("sharpe", ascending=False).head(25)
    for _, r in top.iterrows():
        print(f"  {r['family']:24s} {r['instrument']:6s} {r['tf']:4s} {r['direction']:10s} {r['axis']}={int(r['axis_value']):3d} "
              f"Sh={r['sharpe']:5.2f} (net1x={r['net_sharpe_1.0x']:5.2f}) "
              f"DD={r['max_dd']:5.2f} turn={r['turnover']:.3f} sub=({r['sharpe_2018_2021']:.2f}/{r['sharpe_2022_2026']:.2f}) "
              f"null={r['sharpe_shuffle_null']:5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["control", "screen", "both"], default="both")
    args = ap.parse_args()
    if args.mode in ("control", "both"):
        run_control()
    if args.mode in ("screen", "both"):
        run_screen()


if __name__ == "__main__":
    main()
