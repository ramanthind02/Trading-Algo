"""Nautilus ``BacktestEngine`` P&L lane (WP-3 Unit 2) — ADDITIVE / opt-in.

This module is the realistic-execution counterpart to the frozen vectorized
lane (:class:`research.portfolio.pnl.pnl_engine.VectorizedPnLEngine`). It drives
a Nautilus ``BacktestEngine`` with a thin :class:`TargetRebalanceStrategy` that
*works a daily* ``position_fraction`` *target intraday* against sub-daily bars,
and returns the account's realized return series.

Why this is additive (and never the parity baseline)
-----------------------------------------------------
A Nautilus bar's ``ts_init`` is its *close* (see ``data_platform/nautilus/
ingest.py``). Filling at the *next* bar's open would be look-ahead, so this lane
**cannot** reproduce the research default ``log_intraday`` (enter ``open[t+1]``,
exit ``close[t+1]``) bit-for-bit. It therefore reconciles only against the
close-to-close ``log`` kind (see ``returns_close_to_close`` parity check) and is
selected solely by the opt-in ``pnl_engine="nautilus"`` config — it must never
overwrite the vectorized baseline.

Execution policy defaults (the "simulate the research setup" recipe)
--------------------------------------------------------------------
* ``ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE`` (default): enter on the
  session-OPEN bar via a ``MARKET`` order to the sized target, flatten on the
  session-CLOSE bar — i.e. flat overnight, mirroring the research economics.
* ``ExecutionWindowPolicy.CLOSE_TO_CLOSE``: enter and hold across sessions
  (overnight-inclusive); used only for the frictionless reconciliation.

Position sizing reuses the live math in ``execution/position_sizer.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timezone
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import BestPriceFillModel, FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Instrument as NTInstrument
from nautilus_trader.model.objects import Currency, Money, Quantity
from nautilus_trader.trading.strategy import Strategy

from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.ingest import _resolve_instrument, ingest_mt5_intraday
from data_platform.nautilus.instruments import to_nautilus_instrument
from execution.position_sizer import ContractSpec, PositionSizer, RoundingMethod

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class ExecutionWindowPolicy(Enum):
    """When the daily target is held relative to the session.

    * ``INTRADAY_OPEN_TO_CLOSE`` — enter on the session-open bar, flatten on the
      session-close bar (flat overnight). Mirrors the research ``log_intraday``
      economics as closely as Nautilus allows (open/close are distinct bars).
    * ``CLOSE_TO_CLOSE`` — enter and hold across session boundaries
      (overnight-inclusive). Used for the frictionless ``log`` reconciliation.
    """

    INTRADAY_OPEN_TO_CLOSE = "intraday_open_to_close"
    CLOSE_TO_CLOSE = "close_to_close"


class ExecutionPolicy(Enum):
    """How the entry order is worked. Default crosses at the session open."""

    MARKET_ON_OPEN = "market_on_open"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Mutable per-session bookkeeping for the rebalancer."""

    current_date: date | None = None
    entered_today: bool = False


class TargetRebalanceStrategy(Strategy):
    """Work a precomputed daily ``position_fraction`` target intraday.

    The strategy does **no signal recompute** — it receives a precomputed map of
    ``session-date -> position_fraction`` (from the frozen alpha core) and a
    :class:`execution.position_sizer.PositionSizer`, and at each session boundary
    emits a ``MARKET`` order to the sized target (``MARKET_ON_OPEN``), flattening
    or holding per the :class:`ExecutionWindowPolicy`.

    Equity (``balance_total + unrealized_pnl``) is recorded on every bar close so
    the engine can return a per-bar return series aligned to the vectorized lane.
    """

    def __init__(
        self,
        *,
        instrument: NTInstrument,
        bar_type: BarType,
        targets_by_date: dict[date, float],
        sizer: PositionSizer,
        ticker: str,
        window_policy: ExecutionWindowPolicy,
        execution_policy: ExecutionPolicy,
        session_close_ns: frozenset[int],
    ) -> None:
        super().__init__()
        self._instrument = instrument
        self._instrument_id = instrument.id
        self._bar_type = bar_type
        self._targets_by_date = targets_by_date
        self._sizer = sizer
        self._ticker = ticker
        self._window_policy = window_policy
        self._execution_policy = execution_policy
        # Timestamps (ns) of the last bar of each session — known from the bar
        # schedule (not look-ahead on price). In the intraday window we flatten on
        # these bars so no position is carried overnight.
        self._session_close_ns = session_close_ns
        self._session = _SessionState()
        # Equity curve: list of (ts_ns, equity_float). Read by the engine.
        self.equity_curve: list[tuple[int, float]] = []
        # Position trace: list of (ts_ns, net_signed_qty) after processing each
        # bar — used to prove flat-overnight behaviour in tests.
        self.position_trace: list[tuple[int, int]] = []

    # -- lifecycle ----------------------------------------------------------

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)

    def on_bar(self, bar: Bar) -> None:
        bar_dt = pd.Timestamp(bar.ts_event, tz="UTC")
        session_date = bar_dt.date()

        if self._session.current_date is None:
            self._session.current_date = session_date

        is_new_session = session_date != self._session.current_date
        if is_new_session:
            self._session.current_date = session_date
            self._session.entered_today = False

        is_intraday = (
            self._window_policy is ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE
        )
        is_session_close = bar.ts_event in self._session_close_ns

        # Enter once per session on the first (open) bar of the session — unless
        # this very bar is also the session close (single-bar session) under the
        # intraday window, where holding then flattening is a no-op.
        if not self._session.entered_today and not (is_intraday and is_session_close):
            self._enter_for_session(session_date, float(bar.close))
        self._session.entered_today = True

        # Record equity at this bar close (mark-to-market reflects the close)
        # BEFORE flattening, so the held position's intraday move is captured.
        self._record_equity(bar.ts_event)

        # Intraday window: flatten on the last bar of the session so nothing is
        # carried overnight (the overnight gap is never realized).
        if is_intraday and is_session_close:
            self._flatten()

        self.position_trace.append((bar.ts_event, self._net_signed_qty()))

    def on_stop(self) -> None:
        self._flatten()

    # -- helpers ------------------------------------------------------------

    def _target_contracts(self, session_date: date, price: float) -> int:
        fraction = self._targets_by_date.get(session_date, 0.0)
        if fraction == 0.0:
            return 0
        self._sizer.update_prices({self._ticker: price})
        frame = pd.DataFrame(
            [
                {
                    "ticker": self._ticker,
                    "forecast_score": fraction,
                    "position_fraction": fraction,
                }
            ]
        )
        sized = self._sizer.calculate_positions(frame)
        return int(sized.iloc[0]["contracts"])

    def _enter_for_session(self, session_date: date, price: float) -> None:
        target = self._target_contracts(session_date, price)
        current = self._net_signed_qty()
        delta = target - current
        if delta == 0:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=Quantity.from_int(abs(delta)),
        )
        self.submit_order(order)

    def _flatten(self) -> None:
        net = self._net_signed_qty()
        if net == 0:
            return
        side = OrderSide.SELL if net > 0 else OrderSide.BUY
        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=Quantity.from_int(abs(net)),
        )
        self.submit_order(order)

    def _net_signed_qty(self) -> int:
        net = self.portfolio.net_position(self._instrument_id)
        return int(net) if net is not None else 0

    def _record_equity(self, ts_event: int) -> None:
        account = self.portfolio.account(self._instrument_id.venue)
        if account is None:
            return
        currency = account.base_currency or self._instrument.quote_currency
        balance = float(account.balance_total(currency))
        upnl = self.portfolio.unrealized_pnl(self._instrument_id)
        upnl_val = float(upnl) if upnl is not None else 0.0
        self.equity_curve.append((ts_event, balance + upnl_val))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NautilusPnLEngine:
    """OPT-IN realistic-execution lane backed by a Nautilus ``BacktestEngine``.

    Builds a single-instrument backtest from intraday bars (sourced from the
    Nautilus catalog, ingesting from the MT5 provider on demand), works the daily
    ``position_fraction`` target with :class:`TargetRebalanceStrategy`, and
    returns the account's per-bar log-return series.

    Determinism: one ``BacktestEngine`` per call (Nautilus global-singleton
    constraint), a pinned ``FillModel`` ``random_seed``, and a default
    ``BestPriceFillModel`` (frictionless) so the lane is reproducible.
    """

    window_policy: ExecutionWindowPolicy = (
        ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE
    )
    execution_policy: ExecutionPolicy = ExecutionPolicy.MARKET_ON_OPEN
    fill_model: FillModel | None = None
    starting_balance: float = 1_000_000.0
    capital: float = 1_000_000.0
    rounding_method: RoundingMethod = RoundingMethod.ROUND
    random_seed: int = 42
    max_ticks: int | None = 0  # 0 → ingest no ticks (bars only) for speed
    catalog_path: str | None = None

    def returns_from_positions(
        self,
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> pd.Series:
        """Return the account per-bar log returns for the positions frame.

        ``positions_df`` carries ``['ticker','datetime','position_fraction']``;
        the daily target is taken as the (last) ``position_fraction`` per
        (ticker, session-date). Only a single ticker is supported per call (the
        realism lane runs one instrument at a time); multi-ticker callers should
        invoke once per instrument and aggregate downstream.
        """
        if positions_df.empty:
            return _empty_returns()

        tickers = positions_df["ticker"].map(str).unique().tolist()
        if len(tickers) != 1:
            raise ValueError(
                "NautilusPnLEngine handles one ticker per call; got "
                f"{tickers!r}. Invoke per instrument."
            )
        ticker = tickers[0]

        targets_by_date = _targets_by_session_date(positions_df, ticker)
        if not targets_by_date:
            return _empty_returns()

        equity = self._run_backtest(ticker, targets_by_date)
        return _equity_to_log_returns(equity)

    # -- internals ----------------------------------------------------------

    def _run_backtest(
        self, ticker: str, targets_by_date: dict[date, float]
    ) -> pd.Series:
        catalog = get_catalog(self.catalog_path)
        dp_inst = _resolve_instrument(ticker)
        nt_inst = to_nautilus_instrument(dp_inst)
        bar_type = BarType.from_str(f"{nt_inst.id}-1-MINUTE-LAST-EXTERNAL")

        bars = catalog.bars(bar_types=[str(bar_type)])
        if not bars:
            ingest_mt5_intraday(ticker, catalog, max_ticks=self.max_ticks)
            bars = catalog.bars(bar_types=[str(bar_type)])
        if not bars:
            raise FileNotFoundError(
                f"No intraday bars available for {ticker!r} in the Nautilus "
                f"catalog and none could be ingested from the MT5 provider."
            )

        session_close_ns = _session_close_ns(bars)
        venue = nt_inst.id.venue
        currency = nt_inst.quote_currency
        fill_model = self.fill_model or _default_fill_model(self.random_seed)

        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id="BACKTESTER-001",
                logging=LoggingConfig(bypass_logging=True),
            )
        )
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=currency,
            starting_balances=[Money(self.starting_balance, currency)],
            fill_model=fill_model,
        )
        engine.add_instrument(nt_inst)
        engine.add_data(bars)

        sizer = PositionSizer(
            capital=self.capital,
            contract_specs={
                ticker: ContractSpec(
                    ticker=ticker,
                    price=float(bars[0].close),
                    multiplier=float(nt_inst.multiplier),
                    fx_rate=1.0,
                    min_tick=float(nt_inst.price_increment),
                )
            },
            rounding_method=self.rounding_method,
        )

        strategy = TargetRebalanceStrategy(
            instrument=nt_inst,
            bar_type=bar_type,
            targets_by_date=targets_by_date,
            sizer=sizer,
            ticker=ticker,
            window_policy=self.window_policy,
            execution_policy=self.execution_policy,
            session_close_ns=session_close_ns,
        )
        engine.add_strategy(strategy)

        try:
            engine.run()
            equity = _equity_curve_series(strategy.equity_curve)
        finally:
            engine.dispose()
        return equity


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _session_close_ns(bars: list[Bar]) -> frozenset[int]:
    """Nanosecond ts of the last bar of each session (UTC calendar date).

    Derived from the (sorted) bar schedule only — this is the trading-session
    calendar, not a price look-ahead. Used to flatten intraday positions on the
    session-close bar so nothing is held overnight.
    """
    last_by_date: dict[date, int] = {}
    for bar in bars:
        d = pd.Timestamp(bar.ts_event, tz="UTC").date()
        ts = bar.ts_event
        if ts > last_by_date.get(d, -1):
            last_by_date[d] = ts
    return frozenset(last_by_date.values())


def _default_fill_model(random_seed: int) -> FillModel:
    """Frictionless default — cross at the best available price, deterministic."""
    return BestPriceFillModel(random_seed=random_seed)


def _empty_returns() -> pd.Series:
    return pd.Series(
        [], index=pd.DatetimeIndex([], name="datetime"),
        dtype="float64", name="strategy_return",
    )


def _targets_by_session_date(
    positions_df: pd.DataFrame, ticker: str
) -> dict[date, float]:
    """Reduce the positions frame to one ``position_fraction`` per session date.

    Takes the last position per (session-date) so an intraday-stamped positions
    frame collapses to a single daily target. Datetimes are interpreted in UTC.
    """
    sub = positions_df[positions_df["ticker"].map(str) == ticker].copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"], utc=True)
    sub = sub.sort_values("datetime")
    sub["session_date"] = sub["datetime"].dt.date
    last = sub.groupby("session_date")["position_fraction"].last()
    return {d: float(v) for d, v in last.items()}


def _equity_curve_series(curve: list[tuple[int, float]]) -> pd.Series:
    if not curve:
        return pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
    # tz-naive UTC index to match the vectorized lane's convention
    # (calculate_strategy_returns_from_positions floors to naive datetime64[ns]),
    # so the two lanes' return series align directly in reports.
    ts = pd.to_datetime([c[0] for c in curve], utc=True).tz_convert("UTC").tz_localize(None)
    vals = [c[1] for c in curve]
    s = pd.Series(vals, index=pd.DatetimeIndex(ts, name="datetime"), dtype="float64")
    # Collapse duplicate timestamps (keep last) and sort.
    return s[~s.index.duplicated(keep="last")].sort_index()


def _equity_to_log_returns(equity: pd.Series) -> pd.Series:
    """Per-bar log returns of the equity curve, indexed by realization time."""
    if equity.empty:
        return _empty_returns()
    rets = np.log(equity / equity.shift(1)).dropna()
    rets.name = "strategy_return"
    rets.index.name = "datetime"
    return rets
