r"""Peter / CrackingMarkets volatility breakout — Phase-1 realistic Nautilus CFD lane.

Replicates the frictionless vectorized model (``peter_breakout.py``) inside an
event-driven NautilusTrader ``BacktestEngine`` with REALISTIC CFD fills, then charges
the REAL recorded per-minute spread + slippage, to produce a defensible NET Sharpe on
the actual broker economics — and reconciles the frictionless lane against the POC.

Run (venv interpreter, from repo root)::

    .\.venv\Scripts\python.exe -m research.experiments.concretum_intraday_momentum.peter_nautilus_lane --sym NDX
    .\.venv\Scripts\python.exe -m research.experiments.concretum_intraday_momentum.peter_nautilus_lane --sym SP500 --start 2018 --end 2026

Faithful structure (this is the live-traded ruleset, not the academic Noise-Area one):
  * Bands off the session OPEN: upper = open + 0.4*ATR(5), lower = open - 0.4*ATR(5).
  * Entries are RESTING STOP-MARKET orders at the bands (buy-stop @ upper, sell-stop @
    lower), placed once at the first RTH bar. A stop entry is a TAKER fill at the trigger
    (+ FillModel slippage) — the realistic analog of a breakout fill. One attempt per side
    is AUTOMATIC: an entry order is consumed on fill, so it cannot re-fire intraday.
  * Protective stop = retrace to the session OPEN, placed as a reduce-only STOP-MARKET on
    fill. Risk per trade = 0.4*ATR = exactly 1R.
  * EOD: cancel resting orders + flatten at the session-close bar (market-on-close).

Cost model (identical philosophy to the ORB-IBS lane): the engine produces gross
event-driven fills; the REAL recorded half-spread (taker both legs, point=0.1) + a
per-side slippage in index points are charged on the ledger. PnL is reported in
R-multiples (1R = 0.4*ATR) to match Peter's fixed-fractional risk framing and to
reconcile apples-to-apples against ``peter_breakout``.

Reuses ``data_platform.nautilus`` (ingest, instruments, catalog) and ``peter_breakout``
(the daily ATR(5) table + the frictionless reconciliation reference). Experiment-local.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
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

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import peter_breakout as PB

OPEN_S = E._bsec(9, 30)        # 59400 = 16:30 broker = 09:30 ET cash open
SESS_END_S = E._bsec(16, 0)    # 82800 = 23:00 broker = 16:00 ET cash close
POINT = {"NDX": 0.1, "SP500": 0.1}   # live-probed CFD increment (catalog 0.01 is wrong for spread)
K = 0.4
ATR_N = 5
CAPITAL = 1_000_000.0
TRADING_DAYS = 252
_ONE_MINUTE_NS = 60 * 1_000_000_000


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class PeterBreakoutStrategy(Strategy):
    """Resting-stop breakout: buy-stop @ upper / sell-stop @ lower, protective stop @ open."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._instrument_id = config.instrument_id
        self._bar_type = BarType.from_str(config.bar_type)
        self._qty = float(config.contracts)
        self._instrument = None
        # set by the runner after construction:
        self._atr5: dict[date, float] = {}             # day -> ATR(5), strictly past
        self._session_close_ns: frozenset[int] = frozenset()
        # per-session state
        self._day: date | None = None
        self._open_px = float("nan")
        self._placed = False                           # resting entries placed this day
        self._entry_ids: set = set()                   # working entry client_order_ids
        self._prot_id = None                           # working protective-stop id
        self._pending: dict | None = None              # the currently-open trade meta
        self._forced_flat = False
        self.trades: list[dict] = []

    # -- lifecycle ----------------------------------------------------------
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._instrument_id)
        self.subscribe_bars(self._bar_type)

    def on_stop(self) -> None:
        self._force_flat()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _bar_open_dt(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event - _ONE_MINUTE_NS, tz="UTC")

    @staticmethod
    def _sec_of_day(dt: pd.Timestamp) -> int:
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    def _net_qty(self) -> float:
        net = self.portfolio.net_position(self._instrument_id)
        return float(net) if net is not None else 0.0

    def _cancel_resting(self) -> None:
        for order in self.cache.orders_open(instrument_id=self._instrument_id):
            self.cancel_order(order)
        self._entry_ids.clear()
        self._prot_id = None

    def _force_flat(self) -> None:
        self._cancel_resting()
        net = self._net_qty()
        if net == 0.0:
            return
        self._forced_flat = True
        side = OrderSide.SELL if net > 0 else OrderSide.BUY
        self.submit_order(self.order_factory.market(
            instrument_id=self._instrument_id, order_side=side,
            quantity=self._instrument.make_qty(abs(net)), reduce_only=True))

    def _roll(self, day: date) -> None:
        self._day = day
        self._open_px = float("nan")
        self._placed = False
        self._entry_ids = set()
        self._prot_id = None

    def _place_entries(self, open_px: float, atr: float) -> None:
        R = K * atr
        upper = self._instrument.make_price(open_px + R)
        lower = self._instrument.make_price(open_px - R)
        qty = self._instrument.make_qty(self._qty)
        buy = self.order_factory.stop_market(
            instrument_id=self._instrument_id, order_side=OrderSide.BUY,
            quantity=qty, trigger_price=upper, time_in_force=TimeInForce.GTC)
        sell = self.order_factory.stop_market(
            instrument_id=self._instrument_id, order_side=OrderSide.SELL,
            quantity=qty, trigger_price=lower, time_in_force=TimeInForce.GTC)
        self._entry_ids = {buy.client_order_id, sell.client_order_id}
        self.submit_order(buy)
        self.submit_order(sell)

    # -- core ---------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        dt = self._bar_open_dt(bar)
        sec = self._sec_of_day(dt)
        if sec < OPEN_S or sec > SESS_END_S:
            return
        day = dt.date()
        if self._day != day:
            self._roll(day)

        # first in-session bar: fix the open, place resting band stop-entries
        if not self._placed:
            self._open_px = float(bar.open)
            atr = self._atr5.get(day, float("nan"))
            self._placed = True
            if np.isfinite(atr) and atr > 0 and self._open_px > 0:
                self._place_entries(self._open_px, atr)

        # session-end force-flat (known last bar, or >= 23:00 broker)
        if bar.ts_event in self._session_close_ns or sec >= SESS_END_S:
            self._force_flat()

    # -- order/position events ---------------------------------------------
    def on_position_opened(self, event) -> None:  # type: ignore[no-untyped-def]
        net = self._net_qty()
        if net == 0.0:
            return
        dirn = 1 if net > 0 else -1
        # protective stop = retrace to the session open (reduce-only STOP-MARKET)
        side = OrderSide.SELL if dirn == 1 else OrderSide.BUY
        prot = self.order_factory.stop_market(
            instrument_id=self._instrument_id, order_side=side,
            quantity=self._instrument.make_qty(abs(net)),
            trigger_price=self._instrument.make_price(self._open_px),
            time_in_force=TimeInForce.GTC, reduce_only=True)
        self._prot_id = prot.client_order_id
        self.submit_order(prot)
        self._forced_flat = False
        self._pending = {"day": self._day, "dir": dirn,
                         "atr": float(self._atr5.get(self._day, float("nan")))}

    def on_position_closed(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._pending is None:
            return
        closing = event.closing_order_id
        kind = "eod" if self._forced_flat else ("stop" if closing == self._prot_id else "other")
        self.trades.append({
            **self._pending,
            "entry": float(event.avg_px_open),
            "exit": float(event.avg_px_close),
            "kind": kind,
            "ts_open": int(event.ts_opened),
            "ts_close": int(event.ts_closed),
        })
        self._pending = None
        self._prot_id = None
        self._forced_flat = False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaneConfig:
    sym: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    label: str
    charge_spread: bool
    slip_pts: float
    prob_slippage: float = 0.0
    contracts: float = 1.0
    random_seed: int = 42

    @property
    def point(self) -> float:
        return POINT[self.sym]


@dataclass(frozen=True)
class LaneResult:
    label: str
    trades: pd.DataFrame
    daily_R: pd.Series
    metrics: dict


def _strategy_config(instrument_id, bar_type: str, cfg: LaneConfig):
    class _Cfg(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: str
        contracts: float = 1.0
    return _Cfg(instrument_id=instrument_id, bar_type=bar_type, contracts=cfg.contracts)


def _to_naive(ts):
    if ts is None:
        return None
    return ts.tz_convert("UTC").tz_localize(None) if getattr(ts, "tzinfo", None) else ts


def _atr5_by_day(sym: str, y0: int, y1: int) -> dict[date, float]:
    """Strictly-past Wilder ATR(5) per broker day (reused from peter_breakout)."""
    win = E.load_sessions(sym, y0, y1)
    d = PB._daily_table(win, PB.PParams(atr_n=ATR_N, k=K))
    return {ts.date(): float(a) for ts, a in d["atr"].items()}


def _session_close_ns(bars: list) -> frozenset[int]:
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


def _recorded_spread_map(sym: str, start, end) -> dict[int, float]:
    df = _filter_by_time(_read_partitions(mt5_data_root() / sym / "bars_M1"), start, end)
    if df.empty or "spread" not in df.columns:
        return {}
    close_ns = pd.to_datetime(df["time"], utc=True).astype("int64").to_numpy() + _ONE_MINUTE_NS
    spr = df["spread"].to_numpy().astype("float64")
    return dict(zip(close_ns.tolist(), spr.tolist()))


def run_lane(cfg: LaneConfig) -> LaneResult:
    sym = cfg.sym
    y0 = cfg.start.year if cfg.start is not None else 2008
    y1 = cfg.end.year if cfg.end is not None else 2026
    dp_inst = _resolve_instrument(sym)
    nt_inst = to_nautilus_instrument(dp_inst)
    bar_type = f"{nt_inst.id}-1-MINUTE-LAST-EXTERNAL"
    win_start, win_end = _to_naive(cfg.start), _to_naive(cfg.end)

    tmp_dir = tempfile.mkdtemp(prefix="peter_lane_")
    try:
        catalog = get_catalog(tmp_dir)
        ingest_mt5_intraday(sym, catalog, max_ticks=0, start=win_start, end=win_end)
        bars = catalog.bars(bar_types=[bar_type])
        if not bars:
            raise FileNotFoundError(f"No {sym} M1 bars in window {win_start}..{win_end}.")

        engine = BacktestEngine(config=BacktestEngineConfig(
            trader_id="PETER-001", logging=LoggingConfig(bypass_logging=True)))
        engine.add_venue(
            venue=nt_inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
            base_currency=nt_inst.quote_currency,
            starting_balances=[Money(CAPITAL, nt_inst.quote_currency)],
            fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=cfg.prob_slippage,
                                 random_seed=cfg.random_seed),
            reject_stop_orders=False,
            bar_adaptive_high_low_ordering=False)
        engine.add_instrument(nt_inst)
        engine.add_data(bars)

        strat = PeterBreakoutStrategy(_strategy_config(nt_inst.id, bar_type, cfg))
        strat._atr5 = _atr5_by_day(sym, y0, y1)
        strat._session_close_ns = _session_close_ns(bars)
        engine.add_strategy(strat)
        try:
            engine.run()
            trades = pd.DataFrame(strat.trades)
        finally:
            engine.dispose()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    spread_map = _recorded_spread_map(sym, win_start, win_end) if cfg.charge_spread else {}
    trades = _finalize(trades, cfg, spread_map)
    daily = trades.groupby("day")["R"].sum() if not trades.empty else pd.Series(dtype=float)
    return LaneResult(cfg.label, trades, daily, _metrics(trades, daily))


def _finalize(trades: pd.DataFrame, cfg: LaneConfig, spread_map: dict[int, float]) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.dropna(subset=["entry", "exit", "atr"]).copy()
    trades["gross_pts"] = (trades["exit"] - trades["entry"]) * trades["dir"]
    if cfg.charge_spread and spread_map:
        med = float(np.median(list(spread_map.values())))
        spr_in = trades["ts_open"].astype("int64").map(lambda t: spread_map.get(int(t), med))
        spr_out = trades["ts_close"].astype("int64").map(lambda t: spread_map.get(int(t), med))
        spread_cost = (spr_in.astype(float) + spr_out.astype(float)) * (cfg.point / 2.0)
    else:
        spread_cost = pd.Series(0.0, index=trades.index)
    trades["cost_pts"] = spread_cost + 2.0 * float(cfg.slip_pts)
    trades["net_pts"] = trades["gross_pts"] - trades["cost_pts"]
    trades["R"] = trades["net_pts"] / (K * trades["atr"])           # net R-multiple (1R = 0.4*ATR)
    trades["grossR"] = trades["gross_pts"] / (K * trades["atr"])
    return trades


def _metrics(trades: pd.DataFrame, daily: pd.Series) -> dict:
    if trades.empty or daily.empty or daily.std() == 0:
        return {"trades": int(len(trades))}
    r = daily.to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    # avg round-trip cost expressed in bps of price and in R
    cost_bps = float((trades["cost_pts"] / trades["entry"]).mean() * 1e4)
    return {
        "trades": int(len(trades)),
        "trade_days": int(daily.shape[0]),
        "sharpe_R": round(r.mean() / r.std() * np.sqrt(TRADING_DAYS), 3),
        "meanR/day": round(float(r.mean()), 4),
        "win%": round(100 * (trades["net_pts"] > 0).mean(), 1),
        "sumR": round(float(trades["R"].sum()), 1),
        "grossSumR": round(float(trades["grossR"].sum()), 1),
        "maxDD_R": round(dd, 1),
        "avg_rt_cost_pts": round(float(trades["cost_pts"].mean()), 3),
        "avg_rt_cost_bps": round(cost_bps, 3),
        "stop%": round(100 * (trades["kind"] == "stop").mean(), 0),
        "eod%": round(100 * (trades["kind"] == "eod").mean(), 0),
        "long%": round(100 * (trades["dir"] == 1).mean(), 0),
    }


def reconcile(frictionless: LaneResult, sym: str, y0: int, y1: int, start, end) -> dict:
    """Compare the frictionless lane's daily R vs peter_breakout.simulate (vectorized POC)."""
    poc = PB.simulate(sym, y0, y1, PB.PParams(atr_n=ATR_N, k=K, cost_bps_oneway=0.0))
    poc_daily = poc["R"]
    poc_daily.index = pd.to_datetime(poc_daily.index).date
    nl_daily = frictionless.daily_R.copy()
    if start is not None:
        poc_daily = poc_daily[[d >= _to_naive(start).date() for d in poc_daily.index]]
    if end is not None:
        poc_daily = poc_daily[[d <= _to_naive(end).date() for d in poc_daily.index]]
    idx = sorted(set(poc_daily.index) | set(nl_daily.index))
    pa = pd.Series(poc_daily).reindex(idx).fillna(0.0).to_numpy()
    na = pd.Series(nl_daily).reindex(idx).fillna(0.0).to_numpy()
    corr = float(np.corrcoef(pa, na)[0, 1]) if len(idx) > 1 else float("nan")
    return {
        "poc_sumR": round(float(poc_daily.sum()), 1),
        "lane_sumR": round(float(nl_daily.sum()), 1),
        "poc_trade_days": int((poc_daily != 0).sum()),
        "lane_trade_days": int((nl_daily != 0).sum()),
        "daily_R_corr": round(corr, 4),
    }


# --------------------------------------------------------------------------- #
def _year(v):
    if v is None:
        return None
    return pd.Timestamp(f"{int(v)}-01-01") if len(str(v)) == 4 else pd.Timestamp(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Peter volatility breakout — Nautilus CFD lane")
    ap.add_argument("--sym", default="NDX", choices=["NDX", "SP500"])
    ap.add_argument("--start", default="2018")
    ap.add_argument("--end", default=None)
    ap.add_argument("--slip-pts", type=float, default=0.5,
                    help="per-side slippage in INDEX POINTS for the +slip lane")
    args = ap.parse_args()
    sym = args.sym
    start = _year(args.start)
    end = _year(args.end)
    if args.end and len(str(args.end)) == 4:
        end = pd.Timestamp(f"{int(args.end)}-12-31 23:59:59")
    y0 = start.year if start is not None else 2008
    y1 = end.year if end is not None else 2026

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(f"=== Peter breakout Nautilus lane  sym={sym}  window={start.date()}..{end.date() if end is not None else 'latest'} ===\n")

    print("[1/3] FRICTIONLESS lane (event-driven stop fills, no cost) ...")
    fr = run_lane(LaneConfig(sym, start, end, "frictionless", charge_spread=False, slip_pts=0.0))
    rec = reconcile(fr, sym, y0, y1, start, end)
    print("  reconciliation vs peter_breakout.simulate (frictionless):")
    for k, v in rec.items():
        print(f"    {k:>16} = {v}")
    verdict = "PASS" if rec["daily_R_corr"] > 0.95 else "CHECK"
    print(f"  --> reconciliation {verdict} (target daily-R corr > 0.95)\n")

    print("[2/3] REALISTIC lane (recorded half-spread, taker both legs) ...")
    rs = run_lane(LaneConfig(sym, start, end, "recorded_spread", charge_spread=True, slip_pts=0.0))

    print(f"[3/3] REALISTIC lane (recorded spread + {args.slip_pts} idx-pt/side slippage) ...")
    rsl = run_lane(LaneConfig(sym, start, end, "recorded_spread+slip", charge_spread=True,
                              slip_pts=args.slip_pts))

    table = pd.DataFrame([fr.metrics, rs.metrics, rsl.metrics],
                         index=[fr.label, rs.label, rsl.label])
    print("\n=== Nautilus-lane NET metrics (R-multiple, 1R = 0.4*ATR) ===")
    print(table.to_string())

    out = E.__file__.rsplit("\\", 1)[0] + "\\outputs"
    import os
    os.makedirs(out, exist_ok=True)
    table.to_csv(f"{out}\\peter_nautilus_{sym}.csv")
    pd.DataFrame([rec]).to_csv(f"{out}\\peter_nautilus_recon_{sym}.csv", index=False)
    print(f"\nWrote {out}\\peter_nautilus_{sym}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
