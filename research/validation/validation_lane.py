r"""Final-validation lane — the REAL live strategy, signals generated on-the-fly.

This is the gold-standard, lookahead-PROOF check to run once on a promotion
candidate. It drives the actual live
:class:`~deployment.live.vault_strategy.VaultRebalanceStrategy` through a Nautilus
``BacktestEngine`` over historical data. The signal is produced in-loop by the
real :class:`~deployment.live.forecast_engine.VaultForecastEngine`, now bounded to
the simulation clock (``evaluate(as_of=self.clock.utc_now())``), so the forecast
engine only ever sees candles dated <= the current sim time.

Why this is the strongest validation we have:

* **Lookahead-free by construction** — signal AND execution share one causal,
  event-driven clock. There is no precomputed ``positions_df`` and no hand-applied
  forward shift, so the boundary-misalignment class of bug (see
  :mod:`ensemble.portfolio_impl.backtest_conventions`) cannot occur here.
* **Research <-> live consistency** — it runs the *exact* code the live node runs
  (same strategy, same forecast engine), so a passing validation is also evidence
  the live path behaves as researched.

It is deliberately the EXPENSIVE lane (the forecast engine refits per decision):
run it on the handful of strategies you actually promote, not on research sweeps.
The fast vectorized lane stays the workhorse for exploration.

Data model: the forecast engine reads DAILY candles from the central cache (the
same cache the research/live paths use); the BacktestEngine stream only needs
QUOTES (for sizing + fills) spanning the decision times — synthesized here from
the stored M1 closes. Symbols use the Darwinex broker mapping by default so the
native symbol == the MT5 data-store key (e.g. ``NQ`` -> ``NDX``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_platform.nautilus.ingest import _resolve_instrument, mt5_data_root
from data_platform.nautilus.instruments import to_nautilus_instrument
from data_platform.providers.mt5 import brokers
from deployment.live.config_mt5_sandbox import vendored_adapter_path
from deployment.live.vault_strategy import VaultRebalanceConfig, VaultRebalanceStrategy

# Make the vendored mt5connect symbol parser importable (no terminal/connection),
# so we can build MT5-venue instruments that match what the live strategy resolves.
_ADAPTER = str(vendored_adapter_path())
if _ADAPTER not in sys.path:
    sys.path.insert(0, _ADAPTER)

_ANN = 252


@dataclass(frozen=True)
class ValidationConfig:
    """Knobs for one final-validation backtest run.

    ``start`` / ``end`` bound the simulation window (UTC). Keep it modest — this
    lane refits the vault per decision. ``broker`` defaults to the Darwinex mapping
    so each native symbol equals its MT5 data-store key. ``cache_root=""`` reads
    the shared (research) central cache for daily candles.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    vault_root: str = "vault"
    broker: str = brokers.DEFAULT_BROKER
    venue: str = "MT5"
    tickers: tuple[str, ...] = ("ES", "NQ", "GC", "CL", "SI")
    starting_balance: float = 1_000_000.0
    account_currency: str = "USD"
    decision_time_et: str = "16:05"
    poll_interval_secs: int = 60
    quote_window_timeout_secs: int = 600
    target_volatility: float = 0.15
    max_position_pct: float = 2.5
    idm_max: float = 2.5
    warmup_min_bars: int = 500
    # Headroom above warmup_min_bars: the query cap and the warmup gate must NOT be
    # equal, or a ticker whose calendar yields one fewer session in the capped
    # window fails the >= gate by a single bar (live uses 500==500 — a latent
    # off-by-one; here we give slack so every required ticker clears warmup).
    prediction_daily_max_bars: int = 750
    cache_root: str = ""
    quote_stride_min: int = 5          # downsample M1 quotes to bound data volume
    half_spread_frac: float = 0.0      # 0.0 = frictionless quotes (bid == ask == close)
    lot_size_ceiling: float = 100.0
    populate_central_cache: bool = True  # write daily candles + bias artifacts the engine reads
    warmup_tail_days: int = 1100         # bias-artifact lookback before ``start`` (~500 trading bars + buffer)


@dataclass(frozen=True)
class ValidationResult:
    """Output of a validation run."""

    returns: pd.Series                # daily account returns
    equity_curve: pd.Series           # daily account equity (realized balance)
    metrics: dict[str, float]         # headline metrics on ``returns``
    n_decisions: int                  # number of fills (proxy for decisions acted)
    positions_report: pd.DataFrame
    fills_report: pd.DataFrame
    resolved_tickers: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Instruments + quotes (MT5 venue, matching what the live strategy resolves)
# ---------------------------------------------------------------------------


def _build_mt5_instrument(native: str, venue: str):
    """An MT5-venue Nautilus instrument for ``native``, specs from the data store.

    The live strategy looks up ``InstrumentId(f"{native}.{venue}")``; the stored
    catalog instrument lives on a different venue, so we re-build it on the MT5
    venue from its numeric specs via the vendored MT5 symbol parser (no terminal).
    """
    from types import SimpleNamespace

    from mt5connect.parsing import parse_symbol_info  # type: ignore import-not-found

    nt = to_nautilus_instrument(_resolve_instrument(native))
    info = SimpleNamespace(
        name=native,
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


def _to_real_utc(stored_naive: pd.Series, broker: str) -> pd.Series:
    """Stored MT5 wall digits (broker EET/EEST, mislabelled UTC) -> real UTC.

    MT5 stores the broker server clock (EET/EEST) but labels it UTC; the live
    runtime's :mod:`deployment.live.runtime.broker_clock` treats the Nautilus
    clock as REAL UTC and derives ET (decision) / broker-EET (rollover dead-zone)
    from it. So the lane's synthetic quotes must carry real-UTC timestamps or the
    decision timer fires at the wrong instant and no rebalance happens. We
    re-interpret the stored digits in the SAME broker tz ``broker_clock`` uses
    (``brokers.broker_tzinfo``) and convert to UTC, so the conversion round-trips:
    ``broker_clock.to_broker(real_utc)`` recovers the original stored wall time.
    The handful of DST-transition-edge bars (ambiguous/nonexistent) are dropped.
    """
    broker_tz = brokers.broker_tzinfo(broker)
    return (
        stored_naive.dt.tz_localize(broker_tz, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )


def _synth_quotes(instrument, native: str, config: ValidationConfig) -> list:
    """QuoteTicks from the stored M1 closes over the window (bid/ask = close ∓ hs).

    Timestamps are converted from stored broker time to real UTC (see
    :func:`_to_real_utc`) so the live strategy's UTC-reckoned decision clock fires.
    """
    from nautilus_trader.model.data import QuoteTick

    parts = sorted((mt5_data_root() / native / "bars_M1").glob("year=*/part.parquet"))
    if not parts:
        return []
    df = pd.concat(
        [pd.read_parquet(p, columns=["time", "close"]) for p in parts], ignore_index=True
    )
    # Stored wall digits (strip any false UTC label), then re-interpret as broker tz -> real UTC.
    wall = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.assign(_utc=_to_real_utc(wall, config.broker)).dropna(subset=["_utc"])
    start = pd.Timestamp(config.start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = pd.Timestamp(config.end)
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    df = df[(df["_utc"] >= start) & (df["_utc"] < end)].sort_values("_utc")
    if config.quote_stride_min > 1:
        df = df.iloc[:: config.quote_stride_min]
    size = instrument.make_qty(10)
    out = []
    for utc_t, close in zip(df["_utc"], df["close"]):
        px = float(close)
        hs = config.half_spread_frac * px
        ts = int(utc_t.value)
        out.append(
            QuoteTick(
                instrument_id=instrument.id,
                bid_price=instrument.make_price(px - hs),
                ask_price=instrument.make_price(px + hs),
                bid_size=size,
                ask_size=size,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Equity recorder actor (daily realized-balance sampling)
# ---------------------------------------------------------------------------


def _make_equity_recorder(venue_str: str, currency_str: str):
    """Build a daily equity-sampling Actor (defined lazily to avoid import cost)."""
    from nautilus_trader.common.actor import Actor
    from nautilus_trader.config import ActorConfig
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Currency

    class _EquityRecorderConfig(ActorConfig, frozen=True):
        venue: str = "MT5"
        account_currency: str = "USD"

    class _EquityRecorder(Actor):
        def __init__(self, config: _EquityRecorderConfig) -> None:
            super().__init__(config)
            self._venue = Venue(config.venue)
            self._currency = Currency.from_str(config.account_currency)
            self.equity_curve: list[tuple[int, float]] = []

        def on_start(self) -> None:
            self.clock.set_timer(
                name="validation-equity", interval=timedelta(days=1), callback=self._rec
            )

        def _rec(self, _event) -> None:
            account = self.cache.account_for_venue(self._venue)
            if account is None:
                return
            money = account.balance_total(self._currency)
            equity = money.as_double() if money is not None else 0.0
            self.equity_curve.append((self.clock.timestamp_ns(), equity))

    return _EquityRecorder(_EquityRecorderConfig(venue=venue_str, account_currency=currency_str))


# ---------------------------------------------------------------------------
# Central-cache preflight (the engine reads daily candles + bias artifacts here)
# ---------------------------------------------------------------------------


def ensure_central_cache_coverage(config: ValidationConfig, required_tickers: tuple[str, ...]) -> None:
    """Populate the central cache the forecast engine reads, from the canonical loader.

    The live forecast engine pulls daily candles + bias artifacts from the
    :class:`~cache.runtime.central_cache.CentralCacheStore`; in a research/CI
    environment that store is not populated by the live data pipeline. This writes
    the SAME daily series the research lane uses (via ``data_platform.loaders``,
    research feed) so the validation lane runs on consistent data. Causality is
    still enforced by the engine's ``as_of`` read-bound — writing full history is
    safe. Bias artifacts are (re)built over ``[start - warmup_tail, end]``.
    """
    from cache.runtime.central_cache import CentralCacheStore
    from data_platform.loaders import load_data
    from deployment.live.broker_data import _populate_bias, _resample_monthly
    from lib.core import research_feed
    from lib.core.enums import Ticker, TimeFrame

    research_feed.set_research_feed("cfd")  # match the live (Darwinex CFD) signal source
    store = CentralCacheStore.get_instance()
    end_ts = pd.Timestamp(config.end)
    end_naive = end_ts.tz_localize(None) if end_ts.tzinfo else end_ts
    starts: list[pd.Timestamp] = []
    for canonical in required_tickers:
        if canonical not in Ticker.__members__:
            continue
        ticker = Ticker[canonical]
        try:
            daily = load_data(ticker, TimeFrame.D)
        except Exception:
            continue
        if daily is None or daily.empty:
            continue
        daily = daily.copy()
        daily["datetime"] = pd.to_datetime(daily["datetime"])
        daily = daily[daily["datetime"] <= end_naive]
        if daily.empty:
            continue
        store.set_candles(ticker, TimeFrame.D, daily)
        store.set_candles(ticker, TimeFrame.M, _resample_monthly(daily, canonical))
        starts.append(daily["datetime"].min())
    if starts:
        bias_start = (end_naive - pd.Timedelta(days=config.warmup_tail_days)).to_pydatetime()
        _populate_bias(config.vault_root, bias_start, end_naive.to_pydatetime(), store)


def _required_tickers_for_vault(config: ValidationConfig) -> tuple[str, ...]:
    """Discover the vault's required tickers (primary + cross-reference) cheaply."""
    from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine

    eng = VaultForecastEngine(
        ForecastEngineConfig(
            vault_root=config.vault_root,
            target_volatility=config.target_volatility,
            max_position_pct=config.max_position_pct,
            idm_max=config.idm_max,
            cache_root=config.cache_root or None,
        )
    ).load()
    return eng.required_tickers


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_validation_backtest(config: ValidationConfig) -> ValidationResult:
    """Run the real live strategy over ``config``'s window and return metrics."""
    from nautilus_trader.analysis.reporter import ReportProvider
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Currency, Money

    from lib.core import research_feed

    # The vault signal is the Darwinex CFD series (live source); pin the feed so the
    # forecast engine reads the CFD-namespaced central cache consistently — whether
    # or not the preflight runs.
    research_feed.set_research_feed("cfd")
    if config.populate_central_cache:
        ensure_central_cache_coverage(config, _required_tickers_for_vault(config))

    venue = Venue(config.venue)
    currency = Currency.from_str(config.account_currency)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="VALIDATE-001", logging=LoggingConfig(bypass_logging=True)
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
            native = brokers.resolve(config.broker, canonical)
        except (ValueError, KeyError):
            continue
        try:
            instrument = _build_mt5_instrument(native, config.venue)
        except Exception:
            continue
        quotes = _synth_quotes(instrument, native, config)
        if not quotes:
            continue
        engine.add_instrument(instrument)
        engine.add_data(quotes)
        resolved.append(canonical)

    if not resolved:
        raise FileNotFoundError(
            f"No tradeable tickers with M1 data in {config.start}..{config.end} "
            f"for broker {config.broker!r} (tickers={config.tickers})."
        )

    strategy = VaultRebalanceStrategy(
        VaultRebalanceConfig(
            broker=config.broker,
            venue=config.venue,
            vault_root=config.vault_root,
            cache_root=config.cache_root,
            tickers=tuple(resolved),
            target_volatility=config.target_volatility,
            max_position_pct=config.max_position_pct,
            idm_max=config.idm_max,
            warmup_min_bars=config.warmup_min_bars,
            prediction_daily_max_bars=config.prediction_daily_max_bars,
            decision_time_et=config.decision_time_et,
            poll_interval_secs=config.poll_interval_secs,
            quote_window_timeout_secs=config.quote_window_timeout_secs,
            sizing_basis_usd=0.0,  # read the sandbox account equity
            account_currency=config.account_currency,
            lot_size_ceiling=config.lot_size_ceiling,
        )
    )
    engine.add_strategy(strategy)
    recorder = _make_equity_recorder(config.venue, config.account_currency)
    engine.add_actor(recorder)

    try:
        engine.run()
        equity = _equity_series(recorder.equity_curve, config.starting_balance)
        returns = equity.pct_change().dropna()
        orders = engine.cache.orders()
        positions = engine.cache.positions()
        fills_report = ReportProvider.generate_order_fills_report(orders)
        positions_report = ReportProvider.generate_positions_report(positions)
        n_fills = int((fills_report["filled_qty"].astype(float) > 0).sum()) if not fills_report.empty else 0
    finally:
        engine.dispose()

    return ValidationResult(
        returns=returns,
        equity_curve=equity,
        metrics=_metrics(returns),
        n_decisions=n_fills,
        positions_report=positions_report,
        fills_report=fills_report,
        resolved_tickers=tuple(resolved),
    )


def _equity_series(curve: list[tuple[int, float]], starting_balance: float) -> pd.Series:
    if not curve:
        return pd.Series([starting_balance], index=pd.DatetimeIndex([pd.Timestamp(0)]), name="equity")
    ts = pd.to_datetime([c[0] for c in curve], utc=True).tz_localize(None)
    s = pd.Series([c[1] for c in curve], index=pd.DatetimeIndex(ts, name="datetime"), name="equity")
    return s[~s.index.duplicated(keep="last")].sort_index()


def _metrics(returns: pd.Series) -> dict[str, float]:
    if returns.empty or returns.std() == 0:
        return {"sharpe": float("nan"), "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "total_return": float(returns.sum()) if not returns.empty else 0.0, "n_obs": float(len(returns))}
    return {
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(_ANN)),
        "ann_return_pct": float(returns.mean() * _ANN * 100),
        "ann_vol_pct": float(returns.std() * np.sqrt(_ANN) * 100),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "n_obs": float(len(returns)),
    }


__all__ = ["ValidationConfig", "ValidationResult", "run_validation_backtest"]
