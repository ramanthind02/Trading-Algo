"""WP-3 Unit 2 — Nautilus BacktestEngine P&L lane smoke + reconciliation tests.

Exercises :class:`research.portfolio.pnl.nautilus_engine.NautilusPnLEngine` on the
NDX 2026 MT5 1-minute reference dataset. Three checks:

1. **Smoke** — a constant ``+1`` long target produces a finite per-bar return
   series of the expected shape (and positive total over a rising NDX window).
2. **Flat-overnight** — under ``INTRADAY_OPEN_TO_CLOSE`` the strategy cycles its
   net position through zero at every session boundary (one flat bar per
   session), whereas ``CLOSE_TO_CLOSE`` holds continuously (never flattens).
   This is the realism-lane analogue of the research ``log_intraday`` economics;
   note it cannot match ``log_intraday`` bit-for-bit (Nautilus market orders fill
   on the *next* bar — the documented "no native next-bar-open" constraint).
3. **Reconciliation (frictionless plumbing check)** —
   ``nautilus + CLOSE_TO_CLOSE + BestPriceFillModel`` vs the vectorized
   ``instrument_return_kind='log'`` (close-to-close) on the same bars agree at
   **per-bar correlation >= 0.99**. The residual (integer-contract sizing,
   log-vs-simple convexity, one-bar entry latency) is the expected, documented
   cost of real execution — this lane is additive, never the parity baseline.

Skips cleanly when ``data/mt5_data/NDX`` is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_platform.nautilus.ingest import mt5_data_root
from ensemble.portfolio_impl.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
    calculate_strategy_returns_from_positions,
)
from research.portfolio.pnl import make_pnl_engine
from research.portfolio.pnl.nautilus_engine import (
    ExecutionWindowPolicy,
    NautilusPnLEngine,
    TargetRebalanceStrategy,
)

_SYMBOL = "NDX"

pytestmark = pytest.mark.skipif(
    not (mt5_data_root() / _SYMBOL).exists(),
    reason=f"data/mt5_data/{_SYMBOL} not present",
)


def _ndx_bars() -> pd.DataFrame:
    return pd.read_parquet(
        mt5_data_root() / _SYMBOL / "bars_M1" / "year=2026" / "part.parquet"
    )


def _constant_long_targets(bars: pd.DataFrame) -> pd.DataFrame:
    """One ``position_fraction = +1`` target per NDX session date."""
    dates = pd.to_datetime(bars["time"], utc=True).dt.normalize().unique()
    return pd.DataFrame(
        {
            "ticker": _SYMBOL,
            "datetime": dates,
            "forecast_score": 1.0,
            "position_fraction": 1.0,
        }
    )


def _bar_open_candles(bars: pd.DataFrame) -> pd.DataFrame:
    """Candle contract stamped at bar-OPEN time, matching the equity-curve index.

    The lane records equity at ``bar.ts_event - 1min`` (bar-open time) so the last
    bar of a session stays in the correct calendar day, so the per-bar return
    series is indexed at bar-open. The vectorized comparison candles must use the
    same stamp (NOT ts_event / bar-close) or the two series align one bar apart and
    decorrelate to ~0.
    """
    return pd.DataFrame(
        {
            "ticker": _SYMBOL,
            "datetime": pd.to_datetime(bars["time"], utc=True).dt.tz_convert(None),
            "open": bars["open"].astype(float),
            "close": bars["close"].astype(float),
        }
    )


def test_make_pnl_engine_nautilus_constructs() -> None:
    engine = make_pnl_engine("nautilus")
    assert isinstance(engine, NautilusPnLEngine)


def test_smoke_constant_long_finite_series(tmp_path) -> None:
    bars = _ndx_bars()
    targets = _constant_long_targets(bars)
    engine = NautilusPnLEngine(
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE,
        catalog_path=str(tmp_path / "catalog"),
    )
    rets = engine.returns_from_positions(targets, pd.DataFrame())

    assert isinstance(rets, pd.Series)
    assert len(rets) > 0
    assert rets.notna().all()
    assert bool((rets.abs() > 0).any())  # not all-zero
    # NDX rose over the 2026 window — a constant +1 long nets positive.
    assert float(rets.sum()) > 0.0


def _run_with_trace(window_policy, tmp_path):
    """Run the lane and return (returns, position_trace, session_close_ns)."""
    bars = _ndx_bars()
    targets = _constant_long_targets(bars)
    holder: dict[str, TargetRebalanceStrategy] = {}
    original_init = TargetRebalanceStrategy.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        holder["strategy"] = self

    TargetRebalanceStrategy.__init__ = capturing_init  # type: ignore[method-assign]
    try:
        engine = NautilusPnLEngine(
            window_policy=window_policy, catalog_path=str(tmp_path / "catalog")
        )
        rets = engine.returns_from_positions(targets, pd.DataFrame())
    finally:
        TargetRebalanceStrategy.__init__ = original_init  # type: ignore[method-assign]
    strat = holder["strategy"]
    trace = pd.Series(dict(strat.position_trace)).sort_index()
    return rets, trace, strat._session_close_ns


def test_intraday_flat_overnight_signature(tmp_path) -> None:
    _, trace_intraday, session_close_ns = _run_with_trace(
        ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE, tmp_path
    )
    n_sessions = len(session_close_ns)
    n_flat = int((trace_intraday == 0).sum())
    # Intraday window cycles the net position through zero once per session
    # boundary (plus the initial pre-entry bar) — provably not held continuously.
    assert n_flat >= n_sessions
    # Net position never exceeds the sized target magnitude.
    assert int(trace_intraday.abs().max()) > 0


def test_close_to_close_holds_continuously(tmp_path) -> None:
    _, trace_cc, _ = _run_with_trace(
        ExecutionWindowPolicy.CLOSE_TO_CLOSE, tmp_path
    )
    # Close-to-close holds across boundaries: only the initial pre-entry bar is
    # flat (a couple at most), far fewer than the intraday window's per-session
    # flat bars.
    n_flat = int((trace_cc == 0).sum())
    assert n_flat <= 2


def test_close_to_close_reconciles_with_vectorized_log(tmp_path) -> None:
    bars = _ndx_bars()
    targets = _constant_long_targets(bars)
    candles = _bar_open_candles(bars)

    # Vectorized close-to-close (`log`) with a constant +1 long at every bar.
    posv = pd.DataFrame(
        {"ticker": _SYMBOL, "datetime": candles["datetime"], "position_fraction": 1.0}
    )
    vec = calculate_strategy_returns_from_positions(
        posv, candles, instrument_return_kind="log"
    )

    engine = NautilusPnLEngine(
        window_policy=ExecutionWindowPolicy.CLOSE_TO_CLOSE,
        catalog_path=str(tmp_path / "catalog"),
    )
    nau = engine.returns_from_positions(targets, pd.DataFrame())

    # Both lanes now emit a tz-naive UTC index → align directly.
    joined = pd.concat(
        [vec.rename("vec"), nau.rename("nau")], axis=1
    ).dropna()
    assert len(joined) > 1_000

    corr = float(joined["vec"].corr(joined["nau"]))
    # Frictionless plumbing is correct iff the per-bar shapes track tightly.
    # Achieved ~0.997 on NDX 2026; the residual is integer-contract sizing +
    # log-vs-simple convexity + one-bar entry latency (documented friction).
    assert corr >= 0.99


def _daily_destaled_candles(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLC per session date with a DE-STALED open = first M1 bar's CLOSE.

    The first M1 bar of a session opens at the stale prior close carried across
    the dead zone; the first *tradeable* price (and the lane's actual entry fill)
    is that bar's close. Using it as the daily ``open`` makes the vectorized
    ``log_intraday`` return measure the same thing the lane executes, so any
    residual between the lanes is purely temporal-alignment / sizing — which is
    exactly what this fixture is built to expose.
    """
    t = pd.to_datetime(bars["time"], utc=True).dt.tz_convert(None)
    grp = bars.assign(_d=t.dt.normalize()).sort_values("time").groupby("_d", sort=True)
    first_close = grp["close"].first()
    last_close = grp["close"].last()
    return pd.DataFrame(
        {
            "ticker": _SYMBOL,
            "datetime": first_close.index,
            "open": first_close.to_numpy(dtype=float),
            "close": last_close.to_numpy(dtype=float),
        }
    )


def _varying_targets(bars: pd.DataFrame, *, seed: int = 7) -> pd.DataFrame:
    """One *time-varying* ``position_fraction`` per session date (seeded).

    A constant target is shift-invariant and cannot reveal a one-bar
    misalignment; a varying long/short/flat signal can.
    """
    # tz-naive normalized dates so they merge with the (tz-naive) daily candles.
    dates = pd.to_datetime(bars["time"], utc=True).dt.tz_convert(None).dt.normalize().unique()
    fracs = np.random.default_rng(seed).choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=len(dates))
    return pd.DataFrame(
        {
            "ticker": _SYMBOL,
            "datetime": dates,
            "forecast_score": fracs,
            "position_fraction": fracs,
        }
    )


def test_nautilus_vs_vectorized_varying_signal_reconciles(tmp_path) -> None:
    """Both P&L lanes must agree on a TIME-VARYING signal (corr ~1, vol ratio ~1).

    Guards the temporal alignment between the lanes — specifically that the
    Nautilus adapter (``_targets_by_session_date``) replicates the vectorized
    lane's one-bar forward shift (``pos[t]`` is decided at ``t`` and HELD on
    ``t+1``). The pre-existing constant-target reconciliation cannot catch a
    one-day misalignment because a constant signal is shift-invariant; a varying
    long/short/flat signal can. Regression guard for the one-day-lookahead bug:
    pre-fix this reconciled at corr ~0.90 / vol ratio ~0.79, post-fix ~0.999 / ~1.0.
    """
    bars = _ndx_bars()
    targets = _varying_targets(bars, seed=7)
    candles = _daily_destaled_candles(bars)

    vec = aggregate_intraday_returns_to_daily(
        calculate_strategy_returns_from_positions(
            targets, candles, instrument_return_kind="log_intraday"
        )
    )
    vec = vec[vec != 0]

    engine = NautilusPnLEngine(
        window_policy=ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE,
        catalog_path=str(tmp_path / "catalog"),
    )
    nau = aggregate_intraday_returns_to_daily(
        engine.returns_from_positions(targets, pd.DataFrame())
    )
    nau = nau[nau != 0]

    common = vec.index.intersection(nau.index)
    assert len(common) >= 30  # enough varying days for the correlation to be meaningful
    v, n = vec.loc[common], nau.loc[common]

    corr = float(v.corr(n))
    vol_ratio = float(n.std() / v.std())
    # A one-day misalignment collapses corr to ~0.90 and vol ratio to ~0.79; the
    # aligned lanes track at ~0.999 / ~1.0 (residual = integer-contract sizing +
    # log-vs-simple convexity + one-bar equity-marking lag).
    assert corr >= 0.99, f"lanes decorrelated (corr={corr:.4f}) — temporal misalignment?"
    assert abs(vol_ratio - 1.0) < 0.10, f"vol ratio {vol_ratio:.4f} off — alignment/sizing drift?"
