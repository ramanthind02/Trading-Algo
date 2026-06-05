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

from nautilus_trader.analysis.reporter import ReportProvider
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import BestPriceFillModel, FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Instrument as NTInstrument
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
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
    * ``ROLLOVER_FLATTEN_REENTER`` — hold the daily target across sessions, but
      each day flatten just *before* the financing rollover and re-enter just
      *after* (both via the same passive-limit anchoring as the open path). Models
      the swap-avoidance overlay; pairs with a windowed-tick catalog (quotes only
      around the rollover) — see ``data_platform.nautilus.ingest
      .ingest_mt5_intraday_windowed`` and ``docs/refactor/nautilus/
      backtest_speed_benchmark.md``.
    """

    INTRADAY_OPEN_TO_CLOSE = "intraday_open_to_close"
    CLOSE_TO_CLOSE = "close_to_close"
    ROLLOVER_FLATTEN_REENTER = "rollover_flatten_reenter"


class ExecutionPolicy(Enum):
    """How the entry order is worked at the session open.

    * ``MARKET_ON_OPEN`` — cross the spread immediately (taker). Baseline
      aggressive fill; pays the half-spread vs mid.
    * ``LIMIT_AT_TOUCH`` — rest a ``LIMIT`` at the near touch (BUY at bid, SELL
      at ask). Fills as **maker** only when the market trades to the touch,
      capturing the half-spread; carries fill risk.
    * ``LIMIT_IMPROVE`` — rest a ``LIMIT`` *inside* the spread by
      ``improve_ticks`` (BUY at ``bid + n*tick``, SELL at ``ask - n*tick``).
      Less spread captured, higher fill probability than at-touch.

    The limit policies require **QuoteTick** (bid/ask) data and a
    :class:`CrossAfterPolicy` fallback (convert to ``MARKET`` if unfilled within
    the session) so a daily target is never silently dropped.
    """

    MARKET_ON_OPEN = "market_on_open"
    LIMIT_AT_TOUCH = "limit_at_touch"
    LIMIT_IMPROVE = "limit_improve"


@dataclass(frozen=True)
class CrossAfterPolicy:
    """Fallback that converts an unfilled entry ``LIMIT`` to ``MARKET``.

    ``session_fraction`` is the fraction of the trading session (0..1) after
    which an unfilled working limit is cancelled and re-submitted as a market
    order, guaranteeing the daily target is reached. ``1.0`` disables the
    fallback (pure passive — accept fill risk; the target is simply missed if
    the limit never fills).
    """

    session_fraction: float = 0.75

    def is_enabled(self) -> bool:
        return self.session_fraction < 1.0


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Mutable per-session bookkeeping for the rebalancer."""

    current_date: date | None = None
    entered_today: bool = False
    # Rollover-policy per-day legs (flatten before the rollover, re-enter after).
    exited_today: bool = False
    reentered_today: bool = False
    # ns of the open and close bar of the *current* session — used to map the
    # CROSS_AFTER cutoff onto wall-clock time within the session.
    session_open_ns: int | None = None
    session_close_ns: int | None = None
    # The working entry LIMIT order id for this session (None once filled or
    # crossed). Tracked so CROSS_AFTER can cancel it and fall back to MARKET.
    working_entry_id: object | None = None
    crossed: bool = False


@dataclass(frozen=True)
class FillDiagnostic:
    """One entry-order fill, with the realized price vs the mid at fill time.

    ``signed_spread_vs_mid`` is positive when the fill is *better than mid* for
    the trade direction (spread **captured** — a maker fill at the touch) and
    negative when *worse than mid* (spread **paid** — a taker cross). It is
    expressed in price units; a half-spread capture on a 1.0-wide market is
    ``+0.5``.
    """

    ts_event: int
    order_side: str
    liquidity_side: str  # "MAKER" / "TAKER"
    fill_px: float
    mid_at_fill: float
    signed_spread_vs_mid: float
    half_spread_at_fill: float
    commission: float
    is_entry: bool

    @property
    def liquidity_signed_spread(self) -> float:
        """Economically-robust signed half-spread keyed on the liquidity side.

        ``+half_spread`` for a **MAKER** fill (spread captured), ``-half_spread``
        for a **TAKER** fill (spread paid). This is timing-robust (it does not
        depend on the bar-vs-quote price alignment that ``signed_spread_vs_mid``
        inherits from the raw fill price) and is the quantity reported in the
        spread-vs-market study.
        """
        if self.liquidity_side == "MAKER":
            return self.half_spread_at_fill
        if self.liquidity_side == "TAKER":
            return -self.half_spread_at_fill
        return 0.0


@dataclass(frozen=True)
class LaneResult:
    """Output of :meth:`NautilusPnLEngine.run_with_diagnostics`.

    ``returns`` is the per-bar account log-return series (identical to
    :meth:`returns_from_positions`). ``fill_diagnostics`` are the per-fill
    realized-price-vs-mid records. ``fills_report`` / ``positions_report`` are
    the Nautilus ``ReportProvider`` DataFrames (additive diagnostic surface).
    """

    returns: pd.Series
    fill_diagnostics: list[FillDiagnostic]
    fills_report: pd.DataFrame
    positions_report: pd.DataFrame
    entry_rejects: int = 0


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
        session_open_ns: frozenset[int] | None = None,
        improve_ticks: int = 1,
        cross_after: CrossAfterPolicy | None = None,
        subscribe_quotes: bool = False,
        rollover_minute: int | None = None,
        rollover_half_width: int = 20,
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
        self._session_open_ns = session_open_ns or frozenset()
        self._tick_size = float(instrument.price_increment)
        self._improve_ticks = int(improve_ticks)
        self._cross_after = cross_after or CrossAfterPolicy()
        # Rollover policy: daily minute-of-day (UTC) of the financing rollover and
        # the ± window (minutes) in which the flatten/re-enter legs are worked.
        self._rollover_minute = rollover_minute
        self._rollover_half = int(rollover_half_width)
        self._uses_limit = execution_policy in (
            ExecutionPolicy.LIMIT_AT_TOUCH,
            ExecutionPolicy.LIMIT_IMPROVE,
        )
        # Subscribe to quotes when working limits (needed for touch prices) or
        # when measurement of realized-price-vs-mid is requested (e.g. to record
        # the half-spread a MARKET cross pays).
        self._subscribe_quotes = self._uses_limit or subscribe_quotes
        # Count of entry limits rejected as would-be-marketable (post_only) —
        # this is realized fill risk under the pure-passive measurement.
        self.entry_rejects = 0
        self._session = _SessionState()
        # Latest quote (bid, ask) seen — the touch prices for limit placement.
        self._bid: float | None = None
        self._ask: float | None = None
        # Equity curve: list of (ts_ns, equity_float). Read by the engine.
        self.equity_curve: list[tuple[int, float]] = []
        # Position trace: list of (ts_ns, net_signed_qty) after processing each
        # bar — used to prove flat-overnight behaviour in tests.
        self.position_trace: list[tuple[int, int]] = []
        # Per-fill execution diagnostics (entry + flatten), the additive
        # measurement surface for the spread-vs-market study.
        self.fill_diagnostics: list[FillDiagnostic] = []
        # client_order_ids of entry orders, so on_order_filled can tag entries.
        self._entry_order_ids: set[object] = set()

    # -- lifecycle ----------------------------------------------------------

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)
        if self._subscribe_quotes:
            # Quote ticks supply the bid/ask touch prices for limit placement and
            # are what the matching engine fills resting limits against; they also
            # provide the mid reference for realized-spread measurement.
            self.subscribe_quote_ticks(self._instrument_id)

    def on_quote_tick(self, tick) -> None:  # type: ignore[no-untyped-def]
        self._bid = float(tick.bid_price)
        self._ask = float(tick.ask_price)

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._handle_entry_terminal(event.client_order_id, rejected=True)

    def on_order_denied(self, event) -> None:  # type: ignore[no-untyped-def]
        self._handle_entry_terminal(event.client_order_id, rejected=True)

    def _handle_entry_terminal(self, client_order_id, *, rejected: bool) -> None:
        if client_order_id == self._session.working_entry_id:
            self._session.working_entry_id = None
            if rejected and client_order_id in self._entry_order_ids:
                self.entry_rejects += 1

    def on_bar(self, bar: Bar) -> None:
        bar_dt = pd.Timestamp(bar.ts_event, tz="UTC")
        session_date = bar_dt.date()

        if self._session.current_date is None:
            self._session.current_date = session_date
            self._roll_session(session_date)

        is_new_session = session_date != self._session.current_date
        if is_new_session:
            self._session.current_date = session_date
            self._session.entered_today = False
            self._session.exited_today = False
            self._session.reentered_today = False
            self._roll_session(session_date)

        # Rollover policy has its own (flatten-before / re-enter-after) control
        # flow; it still records equity + the position trace like the open path.
        if self._window_policy is ExecutionWindowPolicy.ROLLOVER_FLATTEN_REENTER:
            self._on_bar_rollover(bar, session_date)
            self._record_equity(bar.ts_event)
            self.position_trace.append((bar.ts_event, self._net_signed_qty()))
            return

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

        # CROSS_AFTER: if a working entry limit is still unfilled past the
        # session cutoff, cancel it and cross with a market order so the daily
        # target is reached (taker fallback).
        self._maybe_cross_after(bar.ts_event)

        # Record equity at this bar close (mark-to-market reflects the close)
        # BEFORE flattening, so the held position's intraday move is captured.
        self._record_equity(bar.ts_event)

        # Intraday window: flatten on the last bar of the session so nothing is
        # carried overnight (the overnight gap is never realized).
        if is_intraday and is_session_close:
            self._cancel_working_entry()
            self._flatten()

        self.position_trace.append((bar.ts_event, self._net_signed_qty()))

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        """Capture per-fill execution diagnostics (realized price vs mid)."""
        liq = event.liquidity_side
        liq_name = "MAKER" if int(liq) == 1 else ("TAKER" if int(liq) == 2 else "NONE")
        side_name = "BUY" if int(event.order_side) == 1 else "SELL"
        fill_px = float(event.last_px)
        mid = self._current_mid(fill_px)
        half_spread = (
            0.5 * (self._ask - self._bid)
            if self._bid is not None and self._ask is not None
            else 0.0
        )
        # Signed spread vs mid (raw realized price): positive = better than mid
        # for the trade side (spread captured); negative = worse (spread paid).
        # NOTE: this inherits any bar-vs-quote timing skew in the fill price; the
        # timing-robust measure is `liquidity_signed_spread` (keyed on
        # MAKER/TAKER), which the study reports.
        if side_name == "BUY":
            signed = mid - fill_px
        else:
            signed = fill_px - mid
        is_entry = event.client_order_id in self._entry_order_ids
        commission = float(event.commission) if event.commission is not None else 0.0
        self.fill_diagnostics.append(
            FillDiagnostic(
                ts_event=event.ts_event,
                order_side=side_name,
                liquidity_side=liq_name,
                fill_px=fill_px,
                mid_at_fill=mid,
                signed_spread_vs_mid=signed,
                half_spread_at_fill=half_spread,
                commission=commission,
                is_entry=is_entry,
            )
        )
        # Clear the working-entry handle once it fills.
        if event.client_order_id == self._session.working_entry_id:
            self._session.working_entry_id = None

    def on_stop(self) -> None:
        self._cancel_working_entry()
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
        """Open path: rebalance to the session's daily target at the open bar."""
        self._rebalance_to_target(self._target_contracts(session_date, price), price)

    def _on_bar_rollover(self, bar: Bar, session_date: date) -> None:
        """Rollover policy: flatten to flat just *before* the financing rollover
        and restore the daily target just *after* — both via the shared passive
        limit anchoring (:meth:`_rebalance_to_target`).

        A leg whose resting limit never fills (no in-window quote trades to the
        touch) simply carries the position into the next leg/day — the realistic
        "missed the fill / paid the swap" outcome, which the next rebalance sees
        as the new delta.
        """
        if self._rollover_minute is None:
            return
        mod = self._minute_of_day(bar.ts_event)
        ref = float(bar.close)
        center, half = self._rollover_minute, self._rollover_half
        # EXIT leg: flatten to flat in [center - half, center).
        if (center - half) <= mod < center and not self._session.exited_today:
            self._cancel_working_entry()
            self._rebalance_to_target(0, ref)
            self._session.exited_today = True
        # RE-ENTER leg: restore the daily target in (center, center + half].
        if center < mod <= (center + half) and not self._session.reentered_today:
            self._cancel_working_entry()
            self._rebalance_to_target(self._target_contracts(session_date, ref), ref)
            self._session.reentered_today = True

    @staticmethod
    def _minute_of_day(ts_ns: int) -> int:
        t = pd.Timestamp(ts_ns, tz="UTC")
        return t.hour * 60 + t.minute

    def _rebalance_to_target(self, target: int, price: float) -> None:
        """Submit the order(s) to move the net position to ``target`` contracts.

        Shared by the open path and the rollover legs. Works a passive limit
        (``_limit_price`` anchoring) when the execution policy uses limits, else a
        market cross; falls back to market if no quote has arrived yet so the
        target is still reached deterministically.
        """
        current = self._net_signed_qty()
        delta = target - current
        if delta == 0:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = Quantity.from_int(abs(delta))

        if not self._uses_limit:
            order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=side,
                quantity=qty,
            )
            self._entry_order_ids.add(order.client_order_id)
            self.submit_order(order)
            return

        # Limit entry: place at/inside the touch. If no quote has arrived yet
        # (limit policy but quote stream lagging), fall back to a market cross so
        # the target is still reached deterministically.
        limit_px = self._limit_price(side, price)
        if limit_px is None:
            order = self.order_factory.market(
                instrument_id=self._instrument_id, order_side=side, quantity=qty
            )
            self._entry_order_ids.add(order.client_order_id)
            self.submit_order(order)
            return
        order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=qty,
            price=Price(limit_px, self._instrument.price_precision),
            # post_only guarantees a passive (maker) resting order: if the order
            # would be immediately marketable against the current book it is
            # rejected rather than crossing — so a fill genuinely captures the
            # spread, and a reject surfaces as fill risk (handled by CROSS_AFTER).
            post_only=True,
        )
        self._entry_order_ids.add(order.client_order_id)
        self._session.working_entry_id = order.client_order_id
        self.submit_order(order)

    def _limit_price(self, side: OrderSide, ref_price: float) -> float | None:
        """Touch (or improved) passive limit price for *side*, or None if no quote.

        The resting limit is anchored on the **entry-bar reference price**
        (``ref_price``) offset by the contemporaneous half-spread — not on the
        raw quote bid/ask. The MT5 NDX feed delivers quotes and M1 bars from
        independent streams whose instantaneous prices can disagree by more than
        the spread; anchoring on the price the *bar* will fill at keeps a BUY
        strictly **below** (SELL **above**) the trade so ``post_only`` does not
        reject it as marketable, and the order rests as a genuine maker. The
        order then fills when a later bar trades to the touch (spread captured).

        ``LIMIT_IMPROVE`` rests ``improve_ticks`` *inside* the spread (closer to
        mid): better fill probability, less spread captured.
        """
        if self._bid is None or self._ask is None:
            return None
        half_spread = max(0.5 * (self._ask - self._bid), self._tick_size)
        improve = (
            self._improve_ticks * self._tick_size
            if self._execution_policy is ExecutionPolicy.LIMIT_IMPROVE
            else 0.0
        )
        if side is OrderSide.BUY:
            # Rest below the trade price by the half-spread (the implied bid),
            # nudged up toward mid by `improve`. Cap strictly below ref_price.
            return min(ref_price - half_spread + improve, ref_price - self._tick_size)
        return max(ref_price + half_spread - improve, ref_price + self._tick_size)

    def _current_mid(self, fallback: float) -> float:
        if self._bid is not None and self._ask is not None:
            return 0.5 * (self._bid + self._ask)
        return fallback

    def _roll_session(self, session_date: date) -> None:
        """Reset per-session working state at a new session boundary."""
        self._session.working_entry_id = None
        self._session.crossed = False
        self._session.session_open_ns = None
        self._session.session_close_ns = None

    def _maybe_cross_after(self, now_ns: int) -> None:
        if not self._uses_limit or not self._cross_after.is_enabled():
            return
        if self._session.working_entry_id is None or self._session.crossed:
            return
        cutoff = self._session_cutoff_ns(now_ns)
        if cutoff is None or now_ns < cutoff:
            return
        # Past cutoff and still working: cancel the limit and cross with market.
        self._cancel_working_entry()
        # Resubmit the *remaining* target as a market order. Compute fresh delta.
        net = self._net_signed_qty()
        target = self._target_contracts(
            self._session.current_date, self._current_mid(0.0) or 1.0
        ) if self._session.current_date is not None else net
        delta = target - net
        self._session.crossed = True
        if delta == 0:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        mkt = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=Quantity.from_int(abs(delta)),
        )
        self._entry_order_ids.add(mkt.client_order_id)
        self.submit_order(mkt)

    def _session_cutoff_ns(self, now_ns: int) -> int | None:
        """Wall-clock ns at which CROSS_AFTER converts the limit to market.

        Maps ``session_fraction`` onto the [session_open, session_close] span.
        The session close is the next ns in ``_session_close_ns`` at or after
        ``now``; the open is the first bar seen this session.
        """
        if self._session.session_open_ns is None:
            self._session.session_open_ns = now_ns
        open_ns = self._session.session_open_ns
        if self._session.session_close_ns is None:
            closes = [c for c in self._session_close_ns if c >= now_ns]
            self._session.session_close_ns = min(closes) if closes else None
        close_ns = self._session.session_close_ns
        if close_ns is None or close_ns <= open_ns:
            return None
        return open_ns + int((close_ns - open_ns) * self._cross_after.session_fraction)

    def _cancel_working_entry(self) -> None:
        wid = self._session.working_entry_id
        if wid is None:
            return
        order = self.cache.order(wid)
        if order is not None and order.is_open:
            self.cancel_order(order)
        self._session.working_entry_id = None

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
    improve_ticks: int = 1
    cross_after: CrossAfterPolicy = field(default_factory=CrossAfterPolicy)
    # Subscribe to quotes even for MARKET policies so the realized fill price can
    # be measured against the mid (the half-spread a taker pays). Limit policies
    # always subscribe regardless.
    measure_spread: bool = False
    fill_model: FillModel | None = None
    starting_balance: float = 1_000_000.0
    capital: float = 1_000_000.0
    rounding_method: RoundingMethod = RoundingMethod.ROUND
    random_seed: int = 42
    max_ticks: int | None = 0  # 0 → ingest no ticks (bars only) for speed
    catalog_path: str | None = None
    # Rollover policy (only used when window_policy is ROLLOVER_FLATTEN_REENTER):
    # daily minute-of-day (UTC) of the financing rollover and the ± window width.
    rollover_minute: int | None = None
    rollover_half_width_min: int = 20

    def _needs_quotes(self) -> bool:
        return self.measure_spread or self.execution_policy in (
            ExecutionPolicy.LIMIT_AT_TOUCH,
            ExecutionPolicy.LIMIT_IMPROVE,
        )

    def run_with_diagnostics(
        self,
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> "LaneResult":
        """Run the lane and return returns + execution diagnostics.

        Additive sibling to :meth:`returns_from_positions` for the spread-vs-
        market study — exposes the per-fill :class:`FillDiagnostic` records and
        the Nautilus ``ReportProvider`` order-fills / positions DataFrames (built
        before the engine is disposed).
        """
        if positions_df.empty:
            return LaneResult(_empty_returns(), [], pd.DataFrame(), pd.DataFrame())
        tickers = positions_df["ticker"].map(str).unique().tolist()
        if len(tickers) != 1:
            raise ValueError(
                "NautilusPnLEngine handles one ticker per call; got "
                f"{tickers!r}. Invoke per instrument."
            )
        ticker = tickers[0]
        targets_by_date = _targets_by_session_date(positions_df, ticker)
        if not targets_by_date:
            return LaneResult(_empty_returns(), [], pd.DataFrame(), pd.DataFrame())
        return self._run_backtest(ticker, targets_by_date, collect_diagnostics=True)

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

        result = self._run_backtest(ticker, targets_by_date)
        return result.returns

    # -- internals ----------------------------------------------------------

    def _run_backtest(
        self,
        ticker: str,
        targets_by_date: dict[date, float],
        *,
        collect_diagnostics: bool = False,
    ) -> "LaneResult":
        catalog = get_catalog(self.catalog_path)
        dp_inst = _resolve_instrument(ticker)
        nt_inst = to_nautilus_instrument(dp_inst)
        bar_type = BarType.from_str(f"{nt_inst.id}-1-MINUTE-LAST-EXTERNAL")

        # Limit policies fill against quote ticks; make sure they are ingested.
        ingest_ticks = self.max_ticks if not self._needs_quotes() else (
            self.max_ticks if self.max_ticks not in (0, None) else None
        )

        bars = catalog.bars(bar_types=[str(bar_type)])
        quotes = (
            catalog.quote_ticks(instrument_ids=[str(nt_inst.id)])
            if self._needs_quotes()
            else []
        )
        if not bars or (self._needs_quotes() and not quotes):
            ingest_mt5_intraday(ticker, catalog, max_ticks=ingest_ticks)
            bars = catalog.bars(bar_types=[str(bar_type)])
            quotes = (
                catalog.quote_ticks(instrument_ids=[str(nt_inst.id)])
                if self._needs_quotes()
                else []
            )
        if not bars:
            raise FileNotFoundError(
                f"No intraday bars available for {ticker!r} in the Nautilus "
                f"catalog and none could be ingested from the MT5 provider."
            )
        if self._needs_quotes() and not quotes:
            raise FileNotFoundError(
                f"Execution policy {self.execution_policy.value!r} requires "
                f"bid/ask QuoteTicks for {ticker!r}, but none are available in "
                f"the catalog or MT5 provider."
            )

        session_close_ns = _session_close_ns(bars)
        session_open_ns = _session_open_ns(bars)
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
        if self._needs_quotes() and quotes:
            engine.add_data(quotes)

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
            session_open_ns=session_open_ns,
            improve_ticks=self.improve_ticks,
            cross_after=self.cross_after,
            subscribe_quotes=self._needs_quotes(),
            rollover_minute=self.rollover_minute,
            rollover_half_width=self.rollover_half_width_min,
        )
        engine.add_strategy(strategy)

        try:
            engine.run()
            equity = _equity_curve_series(strategy.equity_curve)
            diagnostics = list(strategy.fill_diagnostics)
            entry_rejects = strategy.entry_rejects
            if collect_diagnostics:
                orders = engine.cache.orders()
                positions = engine.cache.positions()
                fills_report = ReportProvider.generate_order_fills_report(orders)
                positions_report = ReportProvider.generate_positions_report(
                    positions
                )
            else:
                fills_report = pd.DataFrame()
                positions_report = pd.DataFrame()
        finally:
            engine.dispose()
        return LaneResult(
            returns=_equity_to_log_returns(equity),
            fill_diagnostics=diagnostics,
            fills_report=fills_report,
            positions_report=positions_report,
            entry_rejects=entry_rejects,
        )


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


def _session_open_ns(bars: list[Bar]) -> frozenset[int]:
    """Nanosecond ts of the first bar of each session (UTC calendar date)."""
    first_by_date: dict[date, int] = {}
    for bar in bars:
        d = pd.Timestamp(bar.ts_event, tz="UTC").date()
        ts = bar.ts_event
        if ts < first_by_date.get(d, 1 << 62):
            first_by_date[d] = ts
    return frozenset(first_by_date.values())


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
