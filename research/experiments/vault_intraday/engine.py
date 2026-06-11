"""Frictionless signal engine for the vault-intraday screen.

Drives the *exact* vaulted bias nodes (``helpers.create_fresh_bias_node`` +
``node.add_candle`` — the same path ``BaseModel._instantiate_bias_node`` uses) over
a resampled intraday candle stream, then computes a lookahead-guarded frictionless
P&L. No node logic is re-implemented here, so there is no replication risk.

Causality: a node sees candles strictly in time order and its output at bar ``t`` is
a function of bars ``<= t`` only (streaming). Position is ``signal.shift(1)`` so the
decision taken at the close of bar ``t`` is earned over bar ``t+1`` — no same-bar
lookahead. The ``tf`` label on the candle is cosmetic for these nodes (they consume
only OHLC + bar counts), so intraday bars are labelled ``TimeFrame.D``; annualization
uses the realized bars/year from the actual index, never the enum's D=252.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _frozen_key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def build_signals_fanout(
    module: str,
    ticker: Ticker,
    param_sets: list[dict],
    bars: pd.DataFrame,
) -> dict[tuple, pd.Series]:
    """One pass over the bar stream, fanned out to every param-node.

    Returns ``{frozen_param_key: raw_signed_signal_series}`` (clipped to [-2, 2],
    matching ``BaseModel._extract_feature_series``). Strategy projection (long /
    long_short) is applied later by :func:`project`.
    """
    nodes = [helpers.create_fresh_bias_node(module, ticker, TimeFrame.D, dict(p)) for p in param_sets]
    out: list[list[float]] = [[] for _ in param_sets]
    dt = bars["datetime"].to_numpy()
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    v = bars["volume"].to_numpy()
    for i in range(len(bars)):
        candle = Candle(
            datetime=pd.Timestamp(dt[i]).to_pydatetime(),
            open=float(o[i]),
            high=float(h[i]),
            low=float(low[i]),
            close=float(c[i]),
            volume=float(v[i]),
            ticker=ticker,
            tf=TimeFrame.D,
        )
        for j, node in enumerate(nodes):
            res = node.add_candle(candle)
            out[j].append(float(res[0]) if res else 0.0)
    index = pd.DatetimeIndex(pd.to_datetime(bars["datetime"]))
    return {
        _frozen_key(p): pd.Series(out[j], index=index, dtype="float64").clip(-2.0, 2.0)
        for j, p in enumerate(param_sets)
    }


def project(signal: pd.Series, direction: str) -> pd.Series:
    """Project a signed signal onto a trade direction (mirrors _project_signal_to_strategy)."""
    if direction == "long":
        return signal.clip(lower=0.0)
    if direction == "short":
        return signal.clip(upper=0.0)
    return signal  # long_short


@dataclass(frozen=True)
class Result:
    metrics: dict
    pnl: pd.Series


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))  # in cumulative-log-return units (<= 0)


def evaluate(
    signal: pd.Series,
    close: pd.Series,
    bars_per_year: float,
) -> Result:
    """Lookahead-guarded frictionless P&L + metrics for one signal series.

    ``pnl[t] = signal[t-1] * log(close[t]/close[t-1])``. Re-derived a second,
    independent way and asserted equal. Also reports ``sharpe_lookahead`` (the
    SAME-bar, un-lagged Sharpe) purely as a diagnostic of how much edge would be
    fictitious if the position were not lagged.
    """
    df = pd.DataFrame({"signal": signal, "close": close}).dropna()
    ret = np.log(df["close"]).diff()
    pos = df["signal"].shift(1)
    pnl = (pos * ret).dropna()

    # Independent re-derivation: lag the signal by one bar with a MANUAL numpy shift
    # (not pandas .shift / reindex) and confirm the P&L matches. Catches any
    # alignment/lookahead bug in the pandas path; vectorised so it runs on every call.
    sig_v = df["signal"].to_numpy()
    ret_v = ret.to_numpy()
    pos_manual = np.empty_like(sig_v)
    pos_manual[0] = np.nan
    pos_manual[1:] = sig_v[:-1]
    pnl_manual = (pos_manual * ret_v)[1:]
    assert np.allclose(pnl.to_numpy(), pnl_manual, equal_nan=False), "lookahead/alignment mismatch"

    n = len(pnl)
    if n < 10 or pnl.std() == 0 or not np.isfinite(pnl.std()):
        return Result({"n_bars": n, "sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0,
                       "max_dd": 0.0, "turnover": 0.0, "n_trades": 0, "pct_in_mkt": 0.0,
                       "sharpe_lookahead": 0.0, "bars_per_year": bars_per_year}, pnl)

    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(bars_per_year))
    ann_ret = float(pnl.mean() * bars_per_year)
    ann_vol = float(pnl.std() * np.sqrt(bars_per_year))
    equity = pnl.cumsum().to_numpy()
    max_dd = _max_drawdown(equity)

    pos_clean = pos.reindex(pnl.index).fillna(0.0)
    dpos = pos_clean.diff().abs()
    turnover = float(dpos.mean())                          # avg |Δposition| per bar
    n_trades = int((dpos > 1e-9).sum())                    # bars where position changed
    pct_in_mkt = float((pos_clean.abs() > 1e-9).mean())

    # Diagnostic: un-lagged (same-bar) Sharpe — a sane signal should NOT depend on
    # this; a big gap vs `sharpe` flags an accidental peek.
    pnl_la = (df["signal"] * ret).dropna()
    sharpe_la = float(pnl_la.mean() / pnl_la.std() * np.sqrt(bars_per_year)) if pnl_la.std() else 0.0

    return Result(
        {
            "n_bars": n,
            "sharpe": sharpe,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "max_dd": max_dd,
            "turnover": turnover,
            "n_trades": n_trades,
            "pct_in_mkt": pct_in_mkt,
            "sharpe_lookahead": sharpe_la,
            "bars_per_year": bars_per_year,
        },
        pnl,
    )


def _sharpe_only(pos_values: np.ndarray, ret_values: np.ndarray, bars_per_year: float) -> float:
    """Fast lagged Sharpe from aligned position/return arrays (no assert, no full metrics).

    ``pos_values`` is the (already time-indexed) position; it is lagged one bar here so
    pnl[t] = pos[t-1] * ret[t]. Used by the null sweep where only Sharpe is needed.
    """
    pnl = pos_values[:-1] * ret_values[1:]
    pnl = pnl[np.isfinite(pnl)]
    sd = pnl.std()
    if pnl.size < 10 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(pnl.mean() / sd * np.sqrt(bars_per_year))


def circular_shift_null(
    signal: pd.Series,
    close: pd.Series,
    bars_per_year: float,
    n_shifts: int = 24,
    rng_seed: int = 0,
) -> float:
    """Drift-stripped circular-shift null: the 95th percentile of |Sharpe| over many
    random shifts of the DEMEANED signal.

    Demeaning removes the signal's net exposure — a circular shift preserves mean
    exposure, so without demeaning a net-long signal on a drifting market keeps the
    drift Sharpe and the null can never collapse to ~0 (it would falsely flag a real
    timing edge as an artifact). Averaging over many shifts (vs one noisy ``k``) makes
    it a stable reference: a genuine timing edge should sit well ABOVE this null.
    """
    s = (signal - signal.mean()).to_numpy()
    ret = np.log(close.to_numpy())
    ret = np.concatenate([[np.nan], np.diff(ret)])
    n = len(s)
    if n < 50:
        return 0.0
    rng = np.random.default_rng(rng_seed)
    ks = rng.integers(int(0.1 * n), int(0.9 * n), size=n_shifts)
    nulls = np.fromiter(
        (_sharpe_only(np.roll(s, int(k)), ret, bars_per_year) for k in ks),
        dtype="float64",
        count=n_shifts,
    )
    return float(np.percentile(np.abs(nulls), 95))


def net_sharpe_after_cost(
    pnl: pd.Series,
    pos: pd.Series,
    cost_per_unit_turnover: float,
    bars_per_year: float,
) -> float:
    """Deduct a per-bar cost = cost_per_unit_turnover * |Δposition| from gross P&L.

    ``cost_per_unit_turnover`` is the round-trip cost in log-return units per unit of
    position change (e.g. half-spread/price for a 0→1 entry). Returns net Sharpe.
    """
    dpos = pos.reindex(pnl.index).fillna(0.0).diff().abs().fillna(0.0)
    net = pnl - cost_per_unit_turnover * dpos
    if net.std() == 0 or not np.isfinite(net.std()):
        return 0.0
    return float(net.mean() / net.std() * np.sqrt(bars_per_year))
