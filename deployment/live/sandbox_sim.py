r"""Sandbox live-trading simulation: Darwinex signal + FTMO execution, sim clock.

Emulates the live runtime end-to-end with markets closed, exploiting Nautilus's
backtest/live invariance: the **REAL** :class:`~deployment.live.vault_strategy.VaultRebalanceStrategy`
and :class:`~deployment.live.forecast_engine.VaultForecastEngine` run inside a
``BacktestEngine`` over a recent window. Because it is the same code the live node
runs, the daily 16:05-ET decision, the per-symbol market-hours gate, the quote
window, and the crash-restart decision-state all behave exactly as live.

Data planes (mirroring live):

* **Signal** — the shared Darwinex signal cache, built by the production
  :func:`deployment.live.broker_data.refresh_signal_daily` (Darwinex CFD daily
  bars + bias artifacts). The strategy reads it via ``cache_root=signal_cache_root()``.
* **Execution** — the **FTMO** venue: instruments use FTMO native symbols
  (``brokers.resolve("ftmo", …)``) and quotes carry a MODELED FTMO bid/ask spread;
  overnight **swap** is charged by a :class:`_FinancingActor`. FTMO M1/spread data is
  not yet scraped, so execution prices are the Darwinex series with a per-asset-class
  spread; :class:`SimCostModel` is the seam to swap in measured FTMO spreads/swaps
  (``execution.rollover_overlay.half_spread_bps`` on real FTMO quotes) once available.

Run it via ``scripts/run_sandbox_sim.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_platform.nautilus.ingest import _resolve_instrument, mt5_data_root
from data_platform.nautilus.instruments import to_nautilus_instrument
from data_platform.providers.mt5 import brokers
from deployment.live.runtime.rollover_market import SwapInfo
from deployment.live.vault_strategy import VaultRebalanceConfig, VaultRebalanceStrategy
# Reuse the validation lane's tested helpers (timestamp conversion, equity, metrics).
from research.validation.validation_lane import (
    _equity_series,
    _make_equity_recorder,
    _metrics,
    _to_real_utc,
)

_ANN = 252


# ---------------------------------------------------------------------------
# Modeled FTMO execution cost (per asset class) — the seam to real FTMO data.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimCostModel:
    """Modeled FTMO costs until real FTMO M1/spread data is scraped.

    ``half_spread_bps`` is applied to each quote (bid/ask = mid ∓ hs), so crossing
    the spread is realised on every fill. ``swap_bps_per_day`` is the overnight
    financing drag charged on held positions by :class:`_FinancingActor`.
    """

    half_spread_bps: dict[str, float] = field(
        default_factory=lambda: {
            "index": 1.0, "metal": 2.0, "energy": 3.0, "fx": 0.5, "crypto": 5.0, "other": 2.0,
        }
    )
    swap_bps_per_day: dict[str, float] = field(
        default_factory=lambda: {
            "index": 0.6, "metal": 0.4, "energy": 0.8, "fx": 0.2, "crypto": 1.0, "other": 0.5,
        }
    )

    def half_spread(self, asset_class: str) -> float:
        return self.half_spread_bps.get(asset_class, self.half_spread_bps["other"])

    def swap_day(self, asset_class: str) -> float:
        return self.swap_bps_per_day.get(asset_class, self.swap_bps_per_day["other"])


@dataclass(frozen=True)
class SandboxSimConfig:
    """Knobs for one sandbox live-trading simulation."""

    start: pd.Timestamp
    end: pd.Timestamp
    vault_root: str = "vault"
    signal_broker: str = "darwinex"     # source of truth for signals
    exec_broker: str = "ftmo"           # execution venue (symbols, costs)
    venue: str = "MT5"
    tickers: tuple[str, ...] = ("ES", "NQ", "GC", "CL", "SI")
    starting_balance: float = 100_000.0
    account_currency: str = "USD"
    poll_interval_secs: int = 60
    quote_window_timeout_secs: int = 600
    warmup_min_bars: int = 500
    prediction_daily_max_bars: int = 750
    target_volatility: float = 0.15
    max_position_pct: float = 2.5
    idm_max: float = 2.5
    lot_size_ceiling: float = 100.0
    quote_stride_min: int = 5
    # FTMO trades later than Darwinex (index/energy: FTMO closes 23:15 EET vs
    # Darwinex 23:00). Darwinex price bars stop at its close, so without this the
    # 16:05-ET decision has no fresh execution quote for index/energy and they
    # never fill in the sim (in live, FTMO's own feed quotes that window). Carry the
    # last Darwinex price forward this many minutes so the FTMO instrument is quoted
    # across its session; the market-hours gate still enforces FTMO's real close.
    decision_tail_minutes: int = 25
    startup_grace_secs: int = 0          # sandbox: account exists at t0
    refresh_signal_cache: bool = True    # build the Darwinex signal cache first
    cost: SimCostModel = field(default_factory=SimCostModel)


@dataclass(frozen=True)
class SandboxSimResult:
    gross_equity: pd.Series
    net_equity: pd.Series              # after modeled swap drag
    returns: pd.Series                 # net daily returns
    metrics: dict[str, float]
    n_fills: int
    swap_cost_total: float
    resolved_tickers: tuple[str, ...]
    deferred_closed: tuple[str, ...]   # symbols never traded (market closed all window)
    fills_report: pd.DataFrame
    positions_report: pd.DataFrame


# ---------------------------------------------------------------------------
# Instruments + quotes (FTMO venue, Darwinex price series + modeled FTMO spread)
# ---------------------------------------------------------------------------
def _build_exec_instrument(signal_native: str, exec_native: str, venue: str):
    """FTMO-named MT5 instrument built from the Darwinex contract specs.

    The strategy looks up ``InstrumentId(f"{exec_native}.{venue}")`` (the FTMO
    symbol). FTMO instrument specs aren't catalogued yet, so we reuse the Darwinex
    specs (price precision, multiplier, size step) under the FTMO name — accurate
    enough to validate the workflow; refine when an FTMO catalog exists.
    """
    from types import SimpleNamespace

    from mt5connect.parsing import parse_symbol_info  # type: ignore import-not-found

    nt = to_nautilus_instrument(_resolve_instrument(signal_native))
    info = SimpleNamespace(
        name=exec_native,
        digits=int(nt.price_precision),
        volume_step=float(nt.size_increment),
        volume_min=float(nt.size_increment),
        volume_max=1_000_000.0,
        trade_contract_size=float(nt.multiplier),
        margin_initial=0.0,
        margin_maintenance=0.0,
        currency_base="USD",
        currency_profit="USD",
    )
    return parse_symbol_info(info)


def _extend_to_exec_session(df: pd.DataFrame, config: SandboxSimConfig) -> pd.DataFrame:
    """Carry each day's last Darwinex price forward into the FTMO post-close window.

    Per UTC day, append carry-forward quotes at ``quote_stride_min`` cadence for
    ``decision_tail_minutes`` after the day's last bar, at the last close. This gives
    the FTMO instrument quotes across the 16:05-ET decision window even though
    Darwinex stopped quoting at its earlier close. The market-hours gate
    (``is_market_open(ftmo, …)``) still bounds actual tradeability.
    """
    if df.empty or config.decision_tail_minutes <= 0:
        return df
    stride = max(1, config.quote_stride_min)
    n = max(1, config.decision_tail_minutes // stride)
    extra: list[dict] = []
    for _, day in df.groupby(df["_utc"].dt.normalize()):
        last = day.iloc[-1]
        for k in range(1, n + 1):
            extra.append({"_utc": last["_utc"] + pd.Timedelta(minutes=stride * k), "close": last["close"]})
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True).sort_values("_utc")


def _synth_quotes(exec_instrument, signal_native: str, half_spread_bps: float, config: SandboxSimConfig) -> list:
    """QuoteTicks on the FTMO instrument from the Darwinex M1 closes + FTMO spread.

    Timestamps are converted from stored broker time to real UTC so the live
    strategy's UTC-reckoned decision clock + market-hours gate fire correctly.
    """
    from nautilus_trader.model.data import QuoteTick

    parts = sorted((mt5_data_root() / signal_native / "bars_M1").glob("year=*/part.parquet"))
    if not parts:
        return []
    df = pd.concat(
        [pd.read_parquet(p, columns=["time", "close"]) for p in parts], ignore_index=True
    )
    wall = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.assign(_utc=_to_real_utc(wall, config.signal_broker)).dropna(subset=["_utc"])
    start = pd.Timestamp(config.start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = pd.Timestamp(config.end)
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    df = df[(df["_utc"] >= start) & (df["_utc"] < end)].sort_values("_utc")
    if config.quote_stride_min > 1:
        df = df.iloc[:: config.quote_stride_min]
    df = _extend_to_exec_session(df[["_utc", "close"]], config)
    size = exec_instrument.make_qty(10)
    hs = half_spread_bps / 1e4
    out = []
    for utc_t, close in zip(df["_utc"], df["close"]):
        px = float(close)
        ts = int(utc_t.value)
        out.append(
            QuoteTick(
                instrument_id=exec_instrument.id,
                bid_price=exec_instrument.make_price(px * (1.0 - hs)),
                ask_price=exec_instrument.make_price(px * (1.0 + hs)),
                bid_size=size,
                ask_size=size,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Financing actor — daily overnight swap accrual on held positions
# ---------------------------------------------------------------------------
def _make_financing_actor(venue_str: str, exec_broker: str, cost: SimCostModel):
    """Daily-sampling Actor that accrues modeled overnight swap on open positions."""
    from nautilus_trader.common.actor import Actor
    from nautilus_trader.config import ActorConfig
    from nautilus_trader.model.identifiers import Venue

    class _FinConfig(ActorConfig, frozen=True):
        venue: str = "MT5"

    class _FinancingActor(Actor):
        def __init__(self, config: _FinConfig) -> None:
            super().__init__(config)
            self._venue = Venue(config.venue)
            self.swap_accruals: list[tuple[int, float]] = []  # (ts_ns, swap_cost_usd)

        def on_start(self) -> None:
            self.clock.set_timer(name="sim-financing", interval=timedelta(days=1), callback=self._charge)

        def _charge(self, _event) -> None:
            try:
                cost_today = 0.0
                for position in self.cache.positions_open(venue=self._venue):
                    instrument = self.cache.instrument(position.instrument_id)
                    if instrument is None:
                        continue
                    native = position.instrument_id.symbol.value
                    ac = brokers.asset_class(exec_broker, native).value
                    px = float(position.avg_px_open)
                    qty = abs(float(position.quantity))
                    notional = qty * px * float(instrument.multiplier)
                    cost_today += notional * cost.swap_day(ac) / 1e4
                if cost_today:
                    self.swap_accruals.append((self.clock.timestamp_ns(), cost_today))
            except Exception as exc:  # never crash the sim on a financing accrual
                self.log.warning(f"financing accrual skipped: {exc}")

    return _FinancingActor(_FinConfig(venue=venue_str))


# ---------------------------------------------------------------------------
# Signal-cache build (the production Darwinex path)
# ---------------------------------------------------------------------------
def _required_tickers(config: SandboxSimConfig) -> tuple[str, ...]:
    from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine

    eng = VaultForecastEngine(
        ForecastEngineConfig(vault_root=config.vault_root, cache_root=str(_signal_root()))
    ).load()
    return eng.required_tickers


def _signal_root() -> Path:
    from deployment.live.broker_data import signal_cache_root

    return signal_cache_root()


def build_signal_cache(config: SandboxSimConfig) -> None:
    """Build the shared Darwinex signal cache via the production refresh path."""
    from deployment.live.broker_data import bind_signal_cache, refresh_signal_daily
    from lib.core import research_feed

    research_feed.set_research_feed("cfd")  # Darwinex CFD loaders for the candle source
    store = bind_signal_cache()
    refresh_signal_daily(config.tickers, store=store, vault_root=config.vault_root, populate_bias=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class _SimRolloverStrategy(VaultRebalanceStrategy):
    """The live rollover strategy, with its MT5-derived seams driven by the SIM.

    In a BacktestEngine there is no live terminal, so:
    * ``_now_broker`` reads the **sim clock** (the synthetic quotes carry Darwinex
      M1 timestamps, which ARE broker wall-clock, so sim-UTC == broker time);
    * ``_fetch_swap`` returns a deterministic **negative-carry** swap (long pays),
      matching our long book — the modeled :class:`_FinancingActor` accrues the
      actual swap cost separately;
    * ``_load_sessions`` uses the real per-broker session file.
    """

    def _now_broker(self):  # type: ignore[override]
        return self.clock.utc_now().replace(tzinfo=None)

    def _fetch_swap(self, native: str):  # type: ignore[override]
        # Representative negative long carry (Sun=0..Sat=6; 5=Fri triple for indices).
        return SwapInfo(native, swap_long_points=-50.0, swap_short_points=+5.0,
                        point=0.01, contract_size=1.0, swap_mode=1, triple_weekday=5)


def run_sandbox_sim(config: SandboxSimConfig) -> SandboxSimResult:
    """Run the real live strategy over ``config``'s window: Darwinex signal, FTMO exec."""
    from nautilus_trader.analysis.reporter import ReportProvider
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Currency, Money

    from lib.core import research_feed

    research_feed.set_research_feed("cfd")
    if config.refresh_signal_cache:
        build_signal_cache(config)

    venue = Venue(config.venue)
    currency = Currency.from_str(config.account_currency)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="SANDBOX-001", logging=LoggingConfig(bypass_logging=True)
        )
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=currency,
        starting_balances=[Money(config.starting_balance, currency)],
    )

    resolved: list[str] = []
    for canonical in config.tickers:
        try:
            exec_native = brokers.resolve(config.exec_broker, canonical)
            signal_native = brokers.resolve(config.signal_broker, canonical)
        except (ValueError, KeyError):
            continue
        try:
            instrument = _build_exec_instrument(signal_native, exec_native, config.venue)
        except Exception:
            continue
        hs = config.cost.half_spread(brokers.asset_class(config.exec_broker, canonical).value)
        quotes = _synth_quotes(instrument, signal_native, hs, config)
        if not quotes:
            continue
        engine.add_instrument(instrument)
        engine.add_data(quotes)
        resolved.append(canonical)

    if not resolved:
        raise FileNotFoundError(
            f"No tradeable tickers with Darwinex M1 data in {config.start}..{config.end} "
            f"(signal_broker={config.signal_broker}, exec_broker={config.exec_broker})."
        )

    strategy = _SimRolloverStrategy(
        VaultRebalanceConfig(
            broker=config.exec_broker,          # execution venue (FTMO symbols)
            venue=config.venue,
            vault_root=config.vault_root,
            cache_root=str(_signal_root()),     # signals from the shared Darwinex cache
            tickers=tuple(resolved),
            target_volatility=config.target_volatility,
            max_position_pct=config.max_position_pct,
            idm_max=config.idm_max,
            warmup_min_bars=config.warmup_min_bars,
            prediction_daily_max_bars=config.prediction_daily_max_bars,
            poll_interval_secs=config.poll_interval_secs,
            quote_window_timeout_secs=config.quote_window_timeout_secs,
            sizing_basis_usd=0.0,
            account_currency=config.account_currency,
            lot_size_ceiling=config.lot_size_ceiling,
            startup_grace_secs=config.startup_grace_secs,
        )
    )
    engine.add_strategy(strategy)
    recorder = _make_equity_recorder(config.venue, config.account_currency)
    financing = _make_financing_actor(config.venue, config.exec_broker, config.cost)
    engine.add_actor(recorder)
    engine.add_actor(financing)

    try:
        engine.run()
        gross = _equity_series(recorder.equity_curve, config.starting_balance)
        net, swap_total = _apply_swap_drag(gross, financing.swap_accruals)
        returns = net.pct_change().dropna()
        orders = engine.cache.orders()
        positions = engine.cache.positions()
        fills_report = ReportProvider.generate_order_fills_report(orders)
        positions_report = ReportProvider.generate_positions_report(positions)
        n_fills = int((fills_report["filled_qty"].astype(float) > 0).sum()) if not fills_report.empty else 0
    finally:
        engine.dispose()

    deferred = tuple(sorted(set(resolved) - set(_traded_tickers(fills_report))))
    return SandboxSimResult(
        gross_equity=gross,
        net_equity=net,
        returns=returns,
        metrics=_metrics(returns),
        n_fills=n_fills,
        swap_cost_total=float(swap_total),
        resolved_tickers=tuple(resolved),
        deferred_closed=deferred,
        fills_report=fills_report,
        positions_report=positions_report,
    )


def _apply_swap_drag(gross: pd.Series, accruals: list[tuple[int, float]]) -> tuple[pd.Series, float]:
    """Subtract cumulative daily swap accruals from the gross equity curve."""
    if gross.empty or not accruals:
        return gross, 0.0
    s = pd.Series(
        {pd.Timestamp(ts).tz_localize(None): cost for ts, cost in
         ((pd.Timestamp(t, unit="ns"), c) for t, c in accruals)}
    ).sort_index()
    cum = s.cumsum().reindex(gross.index, method="ffill").fillna(0.0)
    return (gross - cum).rename("net_equity"), float(s.sum())


def _traded_tickers(fills_report: pd.DataFrame) -> set[str]:
    if fills_report.empty or "instrument_id" not in fills_report.columns:
        return set()
    out: set[str] = set()
    for iid in fills_report["instrument_id"].astype(str):
        # InstrumentId is "{native}.{venue}"; the native may contain dots
        # (FTMO "US500.cash"), so strip only the trailing ".{venue}".
        native = iid.rsplit(".", 1)[0]
        try:
            out.add(brokers.canonical_for("ftmo", native))
        except Exception:
            pass
    return out


__all__ = ["SandboxSimConfig", "SandboxSimResult", "SimCostModel", "run_sandbox_sim", "build_signal_cache"]
