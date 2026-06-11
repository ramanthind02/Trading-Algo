"""Buy-and-hold benchmark per (instrument, timeframe).

The headline survivors (robust_trend_breakout_long, double7s, ...) are LONG-BIASED on
up-drifting instruments, so their gross Sharpe partly reflects market drift (beta), not
intraday timing. This computes the always-long benchmark (constant position = 1, zero
turnover so net == gross) on the SAME resampled bars + the SAME realized-bars/year
annualization, so a strategy's Sharpe and drawdown can be read as a LIFT over simply
holding the instrument. A strategy whose Sharpe ~= buy-hold and whose DD ~= buy-hold is
beta, not alpha.

Run: .\.venv\Scripts\python.exe -m research.experiments.vault_intraday.benchmark
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.vault_intraday import data_io

OUT = Path(__file__).resolve().parent / "outputs"
TFS = ["H4", "H2", "H1", "M30"]
INSTRUMENTS = ["SP500", "NDX", "XTIUSD", "XAUUSD", "XAGUSD", "TLT"]
SPLIT = pd.Timestamp("2022-01-01")


def _bh_metrics(close: pd.Series, bpy: float) -> dict:
    ret = np.log(close).diff().dropna()
    if len(ret) < 10 or ret.std() == 0:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}
    eq = ret.cumsum().to_numpy()
    peak = np.maximum.accumulate(eq)
    return {
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(bpy)),
        "ann_ret": float(ret.mean() * bpy),
        "max_dd": float(np.min(eq - peak)),
    }


def main() -> None:
    rows = []
    for sym in INSTRUMENTS:
        for tf in TFS:
            bars = data_io.load_bars(sym, tf)
            idx = pd.DatetimeIndex(bars["datetime"])
            close = pd.Series(bars["close"].to_numpy(), index=idx)
            bpy = data_io.bars_per_year(idx)
            m = _bh_metrics(close, bpy)
            m1 = _bh_metrics(close[idx < SPLIT], data_io.bars_per_year(idx[idx < SPLIT]))
            m2 = _bh_metrics(close[idx >= SPLIT], data_io.bars_per_year(idx[idx >= SPLIT]))
            rows.append({"instrument": sym, "tf": tf, "bh_sharpe": m["sharpe"],
                         "bh_ann_ret": m["ann_ret"], "bh_max_dd": m["max_dd"],
                         "bh_sharpe_2018_2021": m1["sharpe"], "bh_sharpe_2022_2026": m2["sharpe"],
                         "n_bars": len(bars)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "benchmark.csv", index=False)
    print(df.pivot_table(index="instrument", columns="tf", values="bh_sharpe").reindex(columns=TFS).round(2))
    print(f"\nwrote {OUT / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
