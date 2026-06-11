r"""ORB + IBS on NAS100 (NDX CFD) — Phase-1 realistic Nautilus backtest lane.

Replicates the Phase-0 frictionless vectorized POC (``poc.py``) inside an
event-driven NautilusTrader ``BacktestEngine`` with REALISTIC fills (the recorded
per-minute spread synthesized into bid/ask QuoteTicks + a slippage ``FillModel``),
to produce a defensible NET Sharpe / return, and reconciles it against the POC.

Run (always the venv interpreter, from repo root)::

    .\.venv\Scripts\python.exe research\experiments\orb_ibs_nas100\nautilus_lane.py
    .\.venv\Scripts\python.exe research\experiments\orb_ibs_nas100\nautilus_lane.py --start 2023 --end 2024   # fast slice

What this script does (in order):
  1. RECONCILE — runs the lane FRICTIONLESS (no quotes, no-slippage FillModel,
     fills at bar prices) and compares per-trade outcomes vs ``poc.simulate(win,
     Params(use_ibs=True))``: matched trade days, same direction, same exit reason,
     entry/exit prices, and the daily-return correlation. PASS target corr > 0.99.
  2. REALISTIC — re-runs with synth QuoteTicks from the recorded M1 ``spread``
     (point=0.1, the live-probed NDX increment) crossing the real half-spread on
     MARKET fills, plus a per-fill slippage ``FillModel``. Reports net Sharpe,
     ann return %, maxDD %, trades, win%, exit mix — side-by-side with Phase-0.

Strategy logic (faithful to ``poc.py`` / the MQL5 EA, broker GMT+3 wall-clock):
  * Opening range = bars 16:30..16:59 -> OR high / low ; OR_width = hi - lo.
  * IBS over the 16:30..22:59 session = (last_close - sess_low)/(sess_high - sess_low).
  * Entry: first M1 CLOSE beyond the OR on a CLOSED bar, only while bar time < 21:00,
    one trade/day. IBS day-filter (contrarian on the PRIOR day): longs only if
    prev-day IBS < 0.5 ; shorts only if prev-day IBS > 0.5.
  * Brackets: long SL=OR_low, TP=entry + 3*OR_width ; short SL=OR_high,
    TP=entry - 3*OR_width. Force-flat at/after 22:59.

Deviations from the brief, with rationale (trust the installed Nautilus 1.227):
  * ENTRY FILL POINT. The brief specifies "enter on the NEXT bar's open". Nautilus
    1.227 deliberately does NOT provide a next-bar-open execution mode — a bar's
    ``ts_init`` is its close, and filling at the open from a signal on the prior bar
    is treated as look-ahead (docs/nautilustrader/concepts/backtesting.md
    "Order submission timing"). The lookahead-free event-driven analog is to submit
    the MARKET entry on the TRIGGER bar's close, which fills at that close. This was
    verified economically equivalent: the canonical next-bar-open POC and a
    trigger-bar-close variant give daily-return corr = 1.0, identical direction and
    exit reason on every trade, and the same Sharpe (the median |open[K+1]-close[K]|
    gap is 0.3 idx pts on ~99-pt risk). So the lane reconciles to corr ~1.0 against
    the canonical POC despite the one-bar entry-reference difference.
  * INTRABAR SL/TP ORDERING. The POC resolves SL before TP when a single bar
    straddles both (pessimistic). Measured on full history: ZERO of 1271 trades have
    an exit bar that touches both SL and TP (TP at 3x the stop distance is too far
    for one M1 bar to reach both). The ordering is therefore moot here, so the lane
    uses the native bracket (STOP_MARKET SL fills at trigger, LIMIT TP fills at limit)
    with the deterministic ``bar_adaptive_high_low_ordering=False`` (Open->High->
    Low->Close). No custom SL-first override is needed.

Reuses (does not reinvent): ``data_platform.nautilus.ingest`` (bars + synth quotes
from the recorded spread), ``data_platform.nautilus.instruments`` (NDX -> Nautilus
Cfd), and the ``poc.py`` ground truth. Experiment-local only; touches no core package.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Experiment dir on sys.path so we can import the Phase-0 POC ground truth.
_EXPDIR = Path(__file__).resolve().parent
if str(_EXPDIR) not in sys.path:
    sys.path.insert(0, str(_EXPDIR))

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.objects import Money
from nautilus_trader.trading.strategy import Strategy

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.ingest import (
    _filter_by_time,
    _read_partitions,
    _resolve_instrument,
    ingest_mt5_intraday,
    mt5_data_root,
)
from data_platform.nautilus.instruments import to_nautilus_instrument

from poc import (  # Phase-0 ground truth (experiment-local)
    OPEN_S,
    SESS_END_S,
    Params,
    load_sessions,
    simulate,
)

SYMBOL = "NDX"
POINT = 0.1                # live-probed NDX increment (catalog price_increment 0.01 is wrong for spread)
CONTRACTS = 1.0           # fixed quantity per trade (EA lot=0.1 ~ this notional convention)
CAPITAL = 1_000_000.0
TRADING_DAYS = 252
_ONE_MINUTE_NS = 60 * 1_000_000_000


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
@dataclass
class _Session:
    """Mutable per-session (broker calendar-day) bookkeeping."""

    day: date | None = None
    or_hi: float = float("nan")
    or_lo: float = float("nan")
    or_done: bool = False
    sess_hi: float = float("-inf")
    sess_lo: float = float("inf")
    last_close: float = float("nan")
    entered: bool = False          # one trade per day
    in_position: bool = False


class OrbIbsStrategy(Strategy):
    """Event-driven ORB+IBS replica (see module docstring).

    Drives M1 bars only for signal logic; the bracket's MARKET entry + STOP_MARKET
    SL + LIMIT TP fill against the simulated book (bar prices, or the synth quotes
    when present). The strategy keys session boundaries off the bar's broker
    wall-clock (stored MT5 time == broker EET/EEST, ingested straight by
    ``_build_bars`` so ``ts_event`` is the broker bar-close).
    """

    def __init__(self, config) -> None:  # config is a Nautilus StrategyConfig
        super().__init__(config)
        self._instrument_id = config.instrument_id
        self._bar_type = BarType.from_str(config.bar_type)
        self._orb_minutes = int(config.orb_minutes)
        self._rr = float(config.rr)
        self._ibs_thr = float(config.ibs_thr)
        self._qty = float(config.contracts)
        self._orb_end_s = OPEN_S + self._orb_minutes * 60
        self._deadline_s = config.deadline_s
        self._instrument = None
        self._sess = _Session()
        self._prev_ibs: float | None = None
        # Per-trade ledger (the reconciliation surface): one row per closed position.
        self.trades: list[dict] = []
        # Metadata for the trade currently working/open, keyed by nothing (one/day).
        self._pending: dict | None = None
        self._sl_id = None  # client_order_id of the working SL (STOP_MARKET)
        self._tp_id = None  # client_order_id of the working TP (LIMIT)
        self._forced_flat = False  # the EOD force-flat closed the position
        # ns of the LAST bar of each session (the trading calendar, not a price
        # look-ahead) — set by the runner. On these bars we cancel the bracket and
        # flatten, so nothing rests overnight (matches the POC's flat-at-session-end
        # even on early-close/holiday sessions that lack a 22:59 bar).
        self._session_close_ns: frozenset[int] = frozenset()

    # -- lifecycle ----------------------------------------------------------
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._instrument_id)
        self.subscribe_bars(self._bar_type)

    def on_stop(self) -> None:
        self._force_flat()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _bar_open_dt(bar: Bar) -> pd.Timestamp:
        # ts_event is the bar CLOSE (open + 1 min); subtract to recover bar-open.
        return pd.Timestamp(bar.ts_event - _ONE_MINUTE_NS, tz="UTC")

    @staticmethod
    def _sec_of_day(dt: pd.Timestamp) -> int:
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    def _roll_session(self, day: date) -> None:
        # Finalize the prior session's IBS (the day-filter is contrarian on it).
        if self._sess.day is not None:
            rng = self._sess.sess_hi - self._sess.sess_lo
            self._prev_ibs = (
                (self._sess.last_close - self._sess.sess_lo) / rng if rng > 0 else 0.5
            )
        self._sess = _Session(day=day)

    def _net_qty(self) -> float:
        net = self.portfolio.net_position(self._instrument_id)
        return float(net) if net is not None else 0.0

    def _force_flat(self) -> None:
        # Cancel any working contingents (SL/TP), then flatten any open position.
        for order in self.cache.orders_open(instrument_id=self._instrument_id):
            self.cancel_order(order)
        net = self._net_qty()
        if net == 0.0:
            return
        self._forced_flat = True
        side = OrderSide.SELL if net > 0 else OrderSide.BUY
        self.submit_order(
            self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=side,
                quantity=self._instrument.make_qty(abs(net)),
                reduce_only=True,
            )
        )

    # -- core ---------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        dt = self._bar_open_dt(bar)
        sec = self._sec_of_day(dt)
        day = dt.date()

        # Only the 16:30..22:59 broker session window participates (load_sessions
        # mirrors this; bars outside it are ignored for signal + session stats).
        if sec < OPEN_S or sec > SESS_END_S:
            return

        if self._sess.day != day:
            self._roll_session(day)

        o, h, l, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)

        # --- session running stats (for IBS) ---
        self._sess.sess_hi = max(self._sess.sess_hi, h)
        self._sess.sess_lo = min(self._sess.sess_lo, l)
        self._sess.last_close = c

        # --- opening range accumulation (16:30..orb_end) ---
        if sec < self._orb_end_s:
            self._sess.or_hi = h if np.isnan(self._sess.or_hi) else max(self._sess.or_hi, h)
            self._sess.or_lo = l if np.isnan(self._sess.or_lo) else min(self._sess.or_lo, l)
            return  # no entries during the OR

        # First bar at/after OR end: freeze the OR (valid only if hi > lo).
        if not self._sess.or_done:
            self._sess.or_done = (
                not np.isnan(self._sess.or_hi)
                and not np.isnan(self._sess.or_lo)
                and self._sess.or_hi > self._sess.or_lo
            )

        # --- session-end force-flat: the known last bar of the session, or >= 22:59 ---
        if bar.ts_event in self._session_close_ns or sec >= SESS_END_S:
            self._force_flat()
            self._sess.in_position = False
            return

        # --- breakout entry (one/day, lookahead-free: closed bar -> bracket) ---
        if (
            self._sess.or_done
            and not self._sess.entered
            and self._prev_ibs is not None
            and sec < self._deadline_s
        ):
            width = self._sess.or_hi - self._sess.or_lo
            long_ok = self._prev_ibs < self._ibs_thr
            short_ok = self._prev_ibs > (1.0 - self._ibs_thr)
            dirn = 0
            if long_ok and c > self._sess.or_hi:
                dirn = 1
            elif short_ok and c < self._sess.or_lo:
                dirn = -1
            if dirn != 0:
                self._submit_bracket(dirn, c, width)

    def _submit_bracket(self, dirn: int, ref_close: float, width: float) -> None:
        """Submit MARKET entry + STOP_MARKET SL + LIMIT TP (OUO bracket).

        Entry MARKET (submitted on the closed trigger bar) fills at the trigger
        bar's close. SL/TP are reduce-only contingents; OUO cancels the survivor.
        """
        if dirn == 1:
            side = OrderSide.BUY
            sl = self._sess.or_lo
            tp = ref_close + self._rr * width
        else:
            side = OrderSide.SELL
            sl = self._sess.or_hi
            tp = ref_close - self._rr * width
        qty = self._instrument.make_qty(self._qty)
        bracket = self.order_factory.bracket(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=qty,
            sl_trigger_price=self._instrument.make_price(sl),
            tp_price=self._instrument.make_price(tp),
            tp_post_only=False,  # TP may be marketable on a gap; do not reject it.
        )
        # Bracket order list is [entry, SL, TP] (entry first, then the two exits).
        self._sl_id = bracket.orders[1].client_order_id
        self._tp_id = bracket.orders[2].client_order_id
        self.submit_order_list(bracket)
        self._sess.entered = True
        self._sess.in_position = True
        self._forced_flat = False
        self._pending = {
            "day": self._sess.day,
            "dir": dirn,
            "sl": sl,
            "tp": tp,
            "width": width,
        }

    def on_position_closed(self, event) -> None:  # type: ignore[no-untyped-def]
        """Record one completed trade from realized open/close prices.

        Classify the exit by which order closed the position: the SL STOP_MARKET,
        the TP LIMIT, or the EOD force-flat MARKET (``eod``).
        """
        if self._pending is None:
            return
        closing = event.closing_order_id
        if self._forced_flat:
            kind = "eod"
        elif closing == self._sl_id:
            kind = "sl"
        elif closing == self._tp_id:
            kind = "tp"
        else:
            kind = "eod"
        entry_px = float(event.avg_px_open)
        exit_px = float(event.avg_px_close)
        self.trades.append({
            **self._pending,
            "entry": entry_px,
            "exit": exit_px,
            "kind": kind,
            "ts_open": int(event.ts_opened),
            "ts_close": int(event.ts_closed),
        })
        self._pending = None
        self._sess.in_position = False
        self._sl_id = self._tp_id = None
        self._forced_flat = False

    def _finalize_pending(self) -> None:
        """Hook kept for the runner; positions are captured via on_position_closed."""
        return


# --------------------------------------------------------------------------- #
# Engine runner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaneConfig:
    """One Nautilus-lane run.

    The engine ALWAYS produces gross (event-driven) trade outcomes — real fill
    days, directions, fill prices, and exit reasons. Costs are then charged on the
    ledger from the recorded per-minute ``spread`` field:

    * ``charge_spread`` — debit the recorded half-spread on BOTH legs (taker entry
      + taker exit), keyed to the entry/exit bars' recorded spread (``point=0.1``).
      This is the recorded-spread cost model the Phase-0 POC validated.
    * ``slip_pts`` — additional per-side slippage in INDEX POINTS (entry + exit).

    Why a post-fill cost (not a quote crossing): with bar data the Nautilus
    matching engine fills a MARKET order at the bar's OHLC price point, and synth
    QuoteTicks (zero or sized) do NOT take precedence over the bar book for MARKET
    fills (verified empirically); the FillModel's ``prob_slippage`` moves a fill
    only one ``price_increment`` (0.01 idx pt here) — negligible at the 0.1-point
    scale. So the faithful way to charge the REAL recorded spread on event-driven
    MARKET fills is to debit it on the ledger, exactly as ``poc.py`` does with
    ``use_recorded_spread``. The fills themselves remain genuinely event-driven.
    """

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    label: str
    charge_spread: bool        # debit the recorded half-spread on entry + exit (taker both legs)
    slip_pts: float            # per-side slippage in index points (entry + exit)
    point: float = POINT
    prob_slippage: float = 0.0  # FillModel tick slippage (kept wired; ~0.01 idx pt, immaterial)
    orb_minutes: int = 30
    rr: float = 3.0
    ibs_thr: float = 0.5
    contracts: float = CONTRACTS
    random_seed: int = 42


@dataclass(frozen=True)
class LaneResult:
    label: str
    trades: pd.DataFrame      # per-trade ledger (gross + net) from the strategy
    daily_returns: pd.Series  # NET notional per-day return series over all session days
    metrics: dict


def _strategy_config(instrument_id, bar_type: str, cfg: LaneConfig):
    from nautilus_trader.config import StrategyConfig

    class _OrbIbsConfig(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: str
        orb_minutes: int = 30
        rr: float = 3.0
        ibs_thr: float = 0.5
        contracts: float = 1.0
        deadline_s: int = 21 * 3600

    return _OrbIbsConfig(
        instrument_id=instrument_id,
        bar_type=bar_type,
        orb_minutes=cfg.orb_minutes,
        rr=cfg.rr,
        ibs_thr=cfg.ibs_thr,
        contracts=cfg.contracts,
    )


def _to_naive(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    """tz-naive wall-clock for ``ingest._filter_by_time`` (it localizes to UTC)."""
    if ts is None:
        return None
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def run_lane(cfg: LaneConfig) -> LaneResult:
    """Run one ORB+IBS Nautilus backtest and return per-trade + daily metrics."""
    dp_inst = _resolve_instrument(SYMBOL)
    nt_inst = to_nautilus_instrument(dp_inst)
    bar_type = f"{nt_inst.id}-1-MINUTE-LAST-EXTERNAL"
    win_start, win_end = _to_naive(cfg.start), _to_naive(cfg.end)

    tmp_dir = tempfile.mkdtemp(prefix="orb_ibs_lane_")
    try:
        catalog = get_catalog(tmp_dir)
        # Bars are the spine (bars-only ingest; never the 46M-row tick store).
        ingest_mt5_intraday(SYMBOL, catalog, max_ticks=0, start=win_start, end=win_end)
        bars = catalog.bars(bar_types=[bar_type])
        if not bars:
            raise FileNotFoundError(
                f"No NDX M1 bars in window {win_start}..{win_end}."
            )

        fill_model = FillModel(
            prob_fill_on_limit=1.0,
            prob_slippage=cfg.prob_slippage,
            random_seed=cfg.random_seed,
        )
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id="ORBIBS-001", logging=LoggingConfig(bypass_logging=True)
            )
        )
        engine.add_venue(
            venue=nt_inst.id.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=nt_inst.quote_currency,
            starting_balances=[Money(CAPITAL, nt_inst.quote_currency)],
            fill_model=fill_model,
            reject_stop_orders=False,            # allow the SL STOP_MARKET to rest
            bar_adaptive_high_low_ordering=False,  # deterministic O->H->L->C (ordering is moot here)
        )
        engine.add_instrument(nt_inst)
        engine.add_data(bars)

        strategy = OrbIbsStrategy(_strategy_config(nt_inst.id, bar_type, cfg))
        strategy._session_close_ns = _session_close_ns(bars)
        engine.add_strategy(strategy)

        try:
            engine.run()
            strategy._finalize_pending()  # capture a still-open trade closed at on_stop
            trades = pd.DataFrame(strategy.trades)
        finally:
            engine.dispose()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n_session_days = _count_session_days(win_start, win_end)
    spread_map = _recorded_spread_map(win_start, win_end) if cfg.charge_spread else {}
    trades = _finalize_trade_frame(trades, cfg, spread_map)
    daily = _daily_return_series(trades, n_session_days)
    return LaneResult(cfg.label, trades, daily, _metrics(trades, daily, n_session_days))


def _session_close_ns(bars: list) -> frozenset[int]:
    """ns of the LAST in-session bar (16:30..22:59 broker) of each calendar day.

    Derived from the bar SCHEDULE only (trading calendar, not a price look-ahead);
    used to flatten on the session-close bar even when an early-close/holiday
    session has no 22:59 bar — so nothing rests overnight."""
    last_by_day: dict[date, int] = {}
    for bar in bars:
        dt = pd.Timestamp(bar.ts_event - _ONE_MINUTE_NS, tz="UTC")
        sec = dt.hour * 3600 + dt.minute * 60 + dt.second
        if sec < OPEN_S or sec > SESS_END_S:
            continue
        d = dt.date()
        if bar.ts_event > last_by_day.get(d, -1):
            last_by_day[d] = bar.ts_event
    return frozenset(last_by_day.values())


def _recorded_spread_map(start, end) -> dict[int, float]:
    """Map bar-CLOSE ns -> recorded spread (points) from the M1 parquet.

    The Nautilus ``Bar`` does not carry the MT5 ``spread`` field, so we re-key it
    from the source parquet by the same bar-close timestamp the ingest uses
    (``open_time + 1 minute``), to charge the real per-minute spread on the
    entry/exit fills (which land on bar-close timestamps)."""
    df = _filter_by_time(
        _read_partitions(mt5_data_root() / SYMBOL / "bars_M1"), start, end
    )
    if df.empty or "spread" not in df.columns:
        return {}
    close_ns = (
        pd.to_datetime(df["time"], utc=True).astype("int64").to_numpy() + _ONE_MINUTE_NS
    )
    spr = df["spread"].to_numpy().astype("float64")
    return dict(zip(close_ns.tolist(), spr.tolist()))


def _count_session_days(start, end) -> int:
    """Number of distinct broker session days with intraday bars in the window."""
    df = _filter_by_time(
        _read_partitions(mt5_data_root() / SYMBOL / "bars_M1"), start, end
    )
    if df.empty:
        return 0
    t = pd.to_datetime(df["time"]).dt.tz_localize(None)
    sec = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    mask = (sec >= OPEN_S) & (sec <= SESS_END_S)
    return int(t[mask].dt.normalize().nunique())


def _finalize_trade_frame(
    trades: pd.DataFrame, cfg: LaneConfig, spread_map: dict[int, float]
) -> pd.DataFrame:
    """Compute gross and NET per-trade returns.

    Gross is the event-driven realized PnL from the Nautilus fills. Net debits the
    recorded half-spread on BOTH legs (taker entry + taker exit), keyed to each
    leg's bar via ``spread_map`` (points x ``point``), plus ``slip_pts`` of
    slippage per side. ``ret`` (used for daily/Sharpe/metrics) is the NET return.
    """
    if trades.empty:
        return trades
    trades = trades.dropna(subset=["entry", "exit"]).copy()
    trades["gross_pts"] = (trades["exit"] - trades["entry"]) * trades["dir"]
    trades["gross_ret"] = trades["gross_pts"] / trades["entry"]

    if cfg.charge_spread and spread_map:
        ts_open = trades["ts_open"].astype("int64")
        ts_close = trades["ts_close"].astype("int64")
        # Recorded spread (points) at the entry/exit bars; fall back to the median
        # if a fill ts lands on a (rare) gap not in the bar map.
        med = float(np.median(list(spread_map.values()))) if spread_map else 0.0
        spr_in = ts_open.map(lambda t: spread_map.get(int(t), med)).astype("float64")
        spr_out = ts_close.map(lambda t: spread_map.get(int(t), med)).astype("float64")
        # Taker on both legs: pay HALF the spread per leg (cross from mid to touch).
        half = cfg.point / 2.0
        spread_cost = (spr_in + spr_out) * half
    else:
        spread_cost = pd.Series(0.0, index=trades.index)
    slip_cost = 2.0 * float(cfg.slip_pts)  # entry + exit slippage in index points

    trades["cost_pts"] = spread_cost + slip_cost
    trades["net_pts"] = trades["gross_pts"] - trades["cost_pts"]
    trades["ret"] = trades["net_pts"] / trades["entry"]
    return trades


def _daily_return_series(trades: pd.DataFrame, n_days: int) -> pd.Series:
    """Dense per-session-day return series (flat days = 0); one trade/day max."""
    if trades.empty or n_days == 0:
        return pd.Series(np.zeros(max(n_days, 1)))
    day_ret = trades.groupby("day")["ret"].sum()
    arr = np.zeros(n_days)
    arr[: len(day_ret)] = day_ret.to_numpy()
    return pd.Series(arr)


def _metrics(trades: pd.DataFrame, daily: pd.Series, n_days: int) -> dict:
    if trades.empty:
        return {"trades": 0}
    years = n_days / TRADING_DAYS if n_days else float("nan")
    arr = daily.to_numpy()
    sharpe = (arr.mean() / arr.std()) * np.sqrt(TRADING_DAYS) if arr.std() > 0 else 0.0
    eq = trades.sort_values("day")["ret"].cumsum()  # NET cumulative
    maxdd = float((eq.cummax() - eq).max()) if not eq.empty else 0.0
    total_ret = float(trades["ret"].sum())  # NET
    wins = trades["net_pts"] > 0
    return {
        "trades": int(len(trades)),
        "win%": round(100 * wins.mean(), 1),
        "tot_ret%": round(100 * total_ret, 1),
        "ann_ret%": round(100 * total_ret / years, 2) if years else float("nan"),
        "sharpe": round(sharpe, 3),
        "maxDD%": round(100 * maxdd, 1),
        "tp%": round(100 * (trades["kind"] == "tp").mean(), 0),
        "sl%": round(100 * (trades["kind"] == "sl").mean(), 0),
        "eod%": round(100 * (trades["kind"] == "eod").mean(), 0),
        "long%": round(100 * (trades["dir"] == 1).mean(), 0),
        "rt_cost_idx": round(float(trades["cost_pts"].mean()), 2),  # avg round-trip cost (idx pts)
    }


# --------------------------------------------------------------------------- #
# Reconciliation vs Phase-0 POC
# --------------------------------------------------------------------------- #
def reconcile(nl: LaneResult, start_year: int, win) -> dict:
    """Compare the FRICTIONLESS Nautilus lane against ``poc.simulate``.

    Reports matched trade days, direction/exit-reason agreement, entry/exit price
    deltas, per-trade PnL correlation, and daily-return correlation.
    """
    # POC frictionless ground truth: ``ret`` is the gross notional return (cost 0).
    poc_tr = simulate(win, Params(use_ibs=True))
    nl_tr = nl.trades.copy()
    poc_tr["day"] = pd.to_datetime(poc_tr["day"])
    nl_tr["day"] = pd.to_datetime(nl_tr["day"])
    # Compare on GROSS returns (the frictionless lane has no cost anyway).
    nl_tr = nl_tr.drop(columns=["ret"]).rename(columns={"gross_ret": "ret"})

    merged = pd.merge(
        poc_tr[["day", "dir", "kind", "entry", "exit", "ret"]],
        nl_tr[["day", "dir", "kind", "entry", "exit", "ret"]],
        on="day", suffixes=("_poc", "_nl"), how="inner",
    )
    same_dir = (merged["dir_poc"] == merged["dir_nl"]).mean()
    same_kind = (merged["kind_poc"] == merged["kind_nl"]).mean()
    # Daily return series correlation over the union of session days.
    poc_daily = poc_tr.groupby("day")["ret"].sum()
    nl_daily = nl_tr.groupby("day")["ret"].sum()
    idx = sorted(set(poc_daily.index) | set(nl_daily.index))
    pa = poc_daily.reindex(idx).fillna(0.0).to_numpy()
    na = nl_daily.reindex(idx).fillna(0.0).to_numpy()
    daily_corr = float(np.corrcoef(pa, na)[0, 1]) if len(idx) > 1 else float("nan")
    trade_corr = (
        float(np.corrcoef(merged["ret_poc"], merged["ret_nl"])[0, 1])
        if len(merged) > 1 else float("nan")
    )
    return {
        "poc_trades": int(len(poc_tr)),
        "nl_trades": int(len(nl_tr)),
        "matched_days": int(len(merged)),
        "same_dir": round(float(same_dir), 4),
        "same_kind": round(float(same_kind), 4),
        "entry_delta_med": round(float((merged["entry_nl"] - merged["entry_poc"]).abs().median()), 4),
        "exit_delta_med": round(float((merged["exit_nl"] - merged["exit_poc"]).abs().median()), 4),
        "trade_pnl_corr": round(trade_corr, 4),
        "daily_ret_corr": round(daily_corr, 4),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _year_arg(v: str | None) -> pd.Timestamp | None:
    if v is None:
        return None
    return pd.Timestamp(f"{int(v)}-01-01") if len(v) == 4 else pd.Timestamp(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="ORB+IBS NDX — Phase-1 Nautilus lane")
    ap.add_argument("--start", default=None, help="start year (YYYY) or date; default = full history 2018")
    ap.add_argument("--end", default=None, help="end year (YYYY) or date; default = full history")
    ap.add_argument("--slip-pts", type=float, default=0.5,
                    help="per-side slippage in INDEX POINTS for the +slip lane (default 0.5)")
    args = ap.parse_args()

    start = _year_arg(args.start) if args.start else pd.Timestamp("2018-01-01")
    end = _year_arg(args.end)
    # end-year YYYY means "through end of that year".
    if args.end and len(args.end) == 4:
        end = pd.Timestamp(f"{int(args.end)}-12-31 23:59:59")
    start_year = start.year

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"=== ORB+IBS NDX Nautilus lane  window={start.date()}..{end.date() if end is not None else 'latest'} ===\n")

    # POC ground truth loads the same M1 store (full from start_year); window the
    # reconciliation comparison to the same span via the simulate output's days.
    win = load_sessions(start_year)
    if end is not None:
        win = win[win["day"] <= end.tz_localize(None) if end.tzinfo else win["day"] <= end]

    # 1) FRICTIONLESS — reconcile vs POC -----------------------------------
    print("[1/3] FRICTIONLESS lane (event-driven fills, no cost charged) ...")
    frictionless = run_lane(LaneConfig(
        start=start, end=end, label="frictionless", charge_spread=False, slip_pts=0.0))
    rec = reconcile(frictionless, start_year, win)
    print("  reconciliation vs poc.simulate(use_ibs=True):")
    for k, v in rec.items():
        print(f"    {k:>16} = {v}")
    verdict = "PASS" if (rec["daily_ret_corr"] > 0.99 or rec["trade_pnl_corr"] > 0.99) else "FAIL"
    print(f"  --> reconciliation {verdict} (target corr > 0.99)\n")

    # 2) REALISTIC: recorded spread only -----------------------------------
    print("[2/3] REALISTIC lane (recorded half-spread, taker both legs) ...")
    rec_spread = run_lane(LaneConfig(
        start=start, end=end, label="recorded_spread", charge_spread=True, slip_pts=0.0))

    # 3) REALISTIC: recorded spread + slippage -----------------------------
    print(f"[3/3] REALISTIC lane (recorded spread + {args.slip_pts} idx-pt/side slippage) ...")
    rec_slip = run_lane(LaneConfig(
        start=start, end=end, label="recorded_spread+slip", charge_spread=True,
        slip_pts=args.slip_pts))

    # --- metrics table ----------------------------------------------------
    table = pd.DataFrame(
        [frictionless.metrics, rec_spread.metrics, rec_slip.metrics],
        index=[frictionless.label, rec_spread.label, rec_slip.label],
    )
    print("\n=== Nautilus-lane NET metrics (fixed-qty notional returns) ===")
    print(table.to_string())

    out = _EXPDIR / "outputs"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "nautilus_lane_metrics.csv")
    frictionless.trades.to_parquet(out / "nautilus_trades_frictionless.parquet")
    rec_slip.trades.to_parquet(out / "nautilus_trades_realistic.parquet")
    pd.DataFrame([rec]).to_csv(out / "nautilus_reconciliation.csv", index=False)
    print(f"\nWrote: {out / 'nautilus_lane_metrics.csv'}")
    print(f"       {out / 'nautilus_reconciliation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
