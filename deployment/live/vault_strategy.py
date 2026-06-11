"""Nautilus strategy that trades the vault portfolio on an MT5 venue, aligned to
the **financing rollover** (not a fixed cash-close clock).

Imperative shell over pure cores. Two scheduled events per broker-day, timed in
**broker wall-clock** (from a fresh tick — never the host clock):

* **EXIT @ T-15 before 00:00 broker** — for each *held* leg, read its live spread
  and swap; ``execution.rollover_overlay.decide_exit_leg`` flattens (MARKET) only
  the swap-negative legs whose round-trip half-spread is cheaper than the swap
  they'd save tonight (×3 on the leg's triple night). Positive-carry legs are
  **held** to collect the swap; legs whose spread exceeds the swap are held too.
* **ENTRY @ each symbol's reopen + per-leg settle** — evaluate the fresh forecast
  (today's just-closed daily bar) and ``plan_rebalance`` the **delta** vs the
  live position (re-establishes flattened legs, rebalances held legs). Metals wait
  ``entry_settle_metal_min`` for the reopen spike to decay.

Both passes are **position-driven and idempotent** (a flat leg is never exited; a
leg already at target is never re-entered), so a crash + restart self-heals from
the live book — no decision-state file needed. The strategy holds flat while not
warm, in the 00:00–01:00 dead-zone, or when no fresh quote is available.

``_build_engine`` / ``_evaluate_targets`` / ``_now_broker`` / ``_fetch_swap`` are
the test seams used by the deterministic integration tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import OrderSide as NautilusOrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy

from data_platform.providers.mt5 import brokers
from data_platform.providers.mt5.brokers import AssetClass
from deployment.live.forecast_engine import (
    ForecastEngineConfig,
    ForecastResult,
    VaultForecastEngine,
)
from deployment.live.monitoring import live_state
from deployment.live.runtime import rollover_market, rollover_schedule
from deployment.live.runtime.sizing import (
    build_sizing_config,
    net_rebalance,
    symbol_info_from_nautilus_instrument,
    target_signed_lots,
)
from deployment.live.runtime.tick_controller import DemandTickController
from execution.mt5_models import OrderSide as MT5OrderSide
from execution.rollover_overlay import (
    LegAction,
    OverlayParams,
    decide_exit_leg,
    half_spread_bps,
)

_DECISION_TIMER = "vault-rollover"
_SNAPSHOT_TIMER = "vault-live-snapshot"
_SNAPSHOT_SECS = 5
# Every Nth snapshot tick (~60s) do the heavier work: sample equity to the history
# file. The dashboard forecast (target-vs-actual) is reused from the decision window
# and only independently recomputed at most once per _FORECAST_MAX_AGE_SECS while IDLE.
_SLOW_REFRESH_EVERY = 12
_FORECAST_MAX_AGE_SECS = 3600


class _State(str, Enum):
    IDLE = "idle"
    AWAITING_EXIT = "awaiting_exit"
    AWAITING_ENTRY = "awaiting_entry"


class VaultRebalanceConfig(StrategyConfig, frozen=True):
    """Configuration for :class:`VaultRebalanceStrategy` (rollover-aligned).

    ``tickers`` are repo-canonical names (e.g. ``("ES", "NQ", "GC", "SI")``);
    leave empty to trade every forecast ticker the broker can resolve.
    ``sizing_basis_usd`` of ``0`` means "read account equity from the portfolio".
    """

    broker: str = "ftmo"
    venue: str = "MT5"
    vault_root: str = "vault"
    cache_root: str = ""  # per-broker cache namespace ("" = shared default cache)
    tickers: tuple[str, ...] = ()
    target_volatility: float = 0.15
    max_position_pct: float = 2.5
    idm_max: float = 2.5
    warmup_min_bars: int = 500
    prediction_daily_max_bars: int = 750
    poll_interval_secs: int = 20
    quote_window_timeout_secs: int = 90
    sizing_basis_usd: float = 0.0
    account_currency: str = "USD"
    lot_size_ceiling: float = 100.0
    min_rebalance_lots: float = 0.01
    min_rebalance_notional_usd: float = 50.0
    # Rollover overlay knobs (universal; broker minutes):
    exit_lead_min: int = 15          # exit at T-15 before the 00:00-broker rollover
    entry_settle_min: int = 0        # default wait after reopen before entering
    entry_settle_metal_min: int = 5  # metals wait longer for the reopen spike to decay
    cross_margin: float = 1.0        # overlay iff round-trip half-spread < cross_margin*|swap|
    # Robustness: withhold the first action this long after start so broker
    # position/account reconciliation can populate (a restart never trades against
    # a phantom-flat book). A quote older than max_quote_age_secs defers the symbol.
    startup_grace_secs: int = 20
    max_quote_age_secs: float = 600.0
    # ── live-state publication (node-mediated dashboard feed) ────────────
    # An empty ``live_state_dir`` DISABLES publishing (the default — keeps tests
    # and sandbox_sim file-free). The live node sets it via ``node_builder`` to
    # ``data/broker_cache/<broker>/live_state``.
    live_state_dir: str = ""
    exec_tier: str = "sandbox"        # sandbox | demo | live (safety classification)
    account_login: int = 0            # display only
    account_server: str = ""          # display only
    initial_balance: float = 0.0      # total-loss anchor; 0 = first observed equity


class VaultRebalanceStrategy(Strategy):
    def __init__(self, config: VaultRebalanceConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._currency = Currency.from_str(config.account_currency)
        self._params = rollover_schedule.RolloverParams(exit_lead_min=config.exit_lead_min)
        self._overlay = OverlayParams(
            exit_lead_min=config.exit_lead_min,
            entry_settle_min=config.entry_settle_min,
            cross_margin=config.cross_margin,
        )
        self._engine: VaultForecastEngine | None = None
        self._ticks: DemandTickController | None = None
        self._iid_by_canonical: dict[str, InstrumentId] = {}
        self._native_by_canonical: dict[str, str] = {}
        self._sessions: dict[str, rollover_schedule.SymbolSession] = {}
        self._state: _State = _State.IDLE
        self._pending: dict[str, float] = {}   # canonical -> target frac (entry) for the in-flight window
        self._exit_done_for = None              # rollover-boundary date the exit pass ran for
        self._entry_done_for = None             # session date the entry pass ran for
        self._window_deadline: datetime | None = None
        self._ready_to_trade = False
        self._started_at: datetime | None = None
        # ── live-state publication ───────────────────────────────────────
        self._live_dir: Path | None = None
        self._baseline = live_state.RiskBaseline()
        self._halted = False
        self._processed_command_ids: set[str] = set()
        self._snap_ticks = 0
        self._last_forecast: ForecastResult | None = None
        self._last_forecast_refreshed_at: datetime | None = None
        self._seen_broker_date: str | None = None
        self._live_started_iso: str | None = None
        self._last_command_result: dict | None = None

    # ── test seams ───────────────────────────────────────────────────────
    def _build_engine(self) -> VaultForecastEngine:
        cfg = self.config
        return VaultForecastEngine(
            ForecastEngineConfig(
                vault_root=cfg.vault_root,
                target_volatility=cfg.target_volatility,
                max_position_pct=cfg.max_position_pct,
                idm_max=cfg.idm_max,
                warmup_min_bars=cfg.warmup_min_bars,
                prediction_daily_max_bars=cfg.prediction_daily_max_bars,
                cache_root=cfg.cache_root or None,
            )
        )

    def _evaluate_targets(self) -> ForecastResult:
        assert self._engine is not None
        return self._engine.evaluate(as_of=self.clock.utc_now())

    def _now_broker(self) -> datetime | None:
        """Current BROKER wall-clock (from a fresh tick). Overridden in tests."""
        return rollover_market.broker_now()

    def _fetch_swap(self, native: str):
        """Live swap facts for ``native`` from the terminal. Overridden in tests."""
        return rollover_market.fetch_swap(native)

    def _load_sessions(self) -> dict[str, rollover_schedule.SymbolSession]:
        return rollover_market.load_broker_sessions(self.config.broker)

    # ── lifecycle ────────────────────────────────────────────────────────
    def on_start(self) -> None:
        self._live_started_iso = self.clock.utc_now().isoformat()
        self._start_live_state()
        try:
            self._engine = self._build_engine().load()
        except Exception as exc:  # vault missing / unloadable — stay flat
            self.log.error(f"VaultForecastEngine failed to load; not trading: {exc}")
            return

        self._ticks = DemandTickController(self.subscribe_quote_ticks, self.unsubscribe_quote_ticks)
        self._resolve_instruments()
        if not self._iid_by_canonical:
            self.log.error("No tradeable instruments resolved for broker "
                           f"{self.config.broker!r}; not trading.")
            return
        self._sessions = self._load_sessions()

        self._started_at = self.clock.utc_now()
        self._ready_to_trade = True
        self.clock.set_timer(
            name=_DECISION_TIMER,
            interval=timedelta(seconds=self.config.poll_interval_secs),
            callback=self._on_timer,
        )
        self.log.info(
            f"VaultRebalanceStrategy (rollover) started: broker={self.config.broker} "
            f"instruments={sorted(self._iid_by_canonical)} "
            f"exit@T-{self.config.exit_lead_min}m reopen=per-symbol(+settle)"
        )

    def _resolve_instruments(self) -> None:
        assert self._engine is not None
        candidates = self.config.tickers or self._engine.required_tickers
        for canonical in candidates:
            try:
                native = brokers.resolve(self.config.broker, canonical)
            except (ValueError, KeyError) as exc:
                self.log.warning(f"Skipping {canonical}: not resolvable on "
                                 f"{self.config.broker} ({exc})")
                continue
            iid = InstrumentId.from_str(f"{native}.{self.config.venue}")
            if self.cache.instrument(iid) is None:
                self.log.warning(f"Skipping {canonical}->{native}: instrument {iid} "
                                 "not loaded in cache")
                continue
            self._iid_by_canonical[canonical] = iid
            self._native_by_canonical[canonical] = native

    def on_stop(self) -> None:
        if self._ticks is not None and self._ticks.active():
            self._ticks.close_window()
        if _DECISION_TIMER in self.clock.timer_names:
            self.clock.cancel_timer(_DECISION_TIMER)
        if _SNAPSHOT_TIMER in self.clock.timer_names:
            self.clock.cancel_timer(_SNAPSHOT_TIMER)

    # ── decision loop (rollover-aligned, broker time) ────────────────────
    def _on_timer(self, event) -> None:
        if self._halted or not self._ready_to_trade:
            return
        if self._state in (_State.AWAITING_EXIT, _State.AWAITING_ENTRY):
            now = self.clock.utc_now()
            if self._window_deadline is not None and now >= self._window_deadline:
                self.log.warning("Quote window timed out; executing with available quotes")
                self._execute()
            return

        if not self._startup_ready(self.clock.utc_now()):
            return
        bnow = self._now_broker()
        if bnow is None:
            self.log.warning("No broker tick (cannot read broker time); holding")
            return

        # EXIT phase: T-15 → 00:00 broker. Flatten swap-negative legs worth overlaying.
        if rollover_schedule.in_exit_window(bnow, self._params):
            boundary = rollover_schedule.exit_boundary_date(bnow)
            if self._exit_done_for != boundary:
                self._open_exit_window(bnow, boundary)
            return

        # ENTRY phase: each symbol past its reopen+settle, once per session.
        s_date = rollover_schedule.session_date(bnow)
        if self._entry_done_for != s_date:
            self._open_entry_window(bnow, s_date)

    def _startup_ready(self, now: datetime) -> bool:
        if self.portfolio.account(self._venue) is None:
            return False
        grace = self.config.startup_grace_secs
        if grace > 0 and self._started_at is not None:
            if (now - self._started_at).total_seconds() < grace:
                return False
        return True

    def _settle_for(self, canonical: str) -> int:
        try:
            ac = brokers.asset_class(self.config.broker, canonical)
        except Exception:
            ac = None
        if ac is AssetClass.METAL:
            return self.config.entry_settle_metal_min
        return self.config.entry_settle_min

    def _held_canonicals(self) -> list[str]:
        held = []
        for canonical, iid in self._iid_by_canonical.items():
            if abs(float(self.portfolio.net_position(iid))) > 1e-12:
                held.append(canonical)
        return held

    def _open_exit_window(self, bnow: datetime, boundary) -> None:
        held = self._held_canonicals()
        if not held:
            self._exit_done_for = boundary  # nothing to overlay; latch this rollover
            return
        self._pending = {c: 0.0 for c in held}  # target 0 (flatten candidates)
        self._state = _State.AWAITING_EXIT
        self._ticks.open_window(self._iid_by_canonical[c] for c in held)  # type: ignore[union-attr]
        self._window_deadline = self.clock.utc_now() + timedelta(
            seconds=self.config.quote_window_timeout_secs
        )
        self.log.info(f"EXIT window (rollover {boundary}): held={sorted(held)}")

    def _open_entry_window(self, bnow: datetime, s_date) -> None:
        # Which resolved symbols have passed their reopen+settle for this session?
        due: list[str] = []
        for canonical in self._iid_by_canonical:
            native = self._native_by_canonical[canonical]
            session = self._sessions.get(native)
            settle = self._settle_for(canonical)
            if rollover_schedule.in_entry_window(bnow, session, settle, self._params):
                due.append(canonical)
        if not due:
            return
        try:
            result = self._evaluate_targets()
        except Exception as exc:
            self.log.error(f"Forecast evaluation failed; holding this tick: {exc}")
            return
        if not result.ready:
            self.log.warning(f"Warmup not ready; holding. Pending: {result.warmup.not_ready()}")
            return
        # Reuse this decision-window forecast for the dashboard target-vs-actual so the
        # snapshot path doesn't have to run its own (heavy) fit+predict on the loop.
        self._last_forecast = result
        self._last_forecast_refreshed_at = self.clock.utc_now()
        targets = {c: float(result.targets.get(c, 0.0)) for c in due}
        self._pending = targets
        self._record_forecast_decision(s_date, due, result)
        self._state = _State.AWAITING_ENTRY
        self._ticks.open_window(self._iid_by_canonical[c] for c in due)  # type: ignore[union-attr]
        self._window_deadline = self.clock.utc_now() + timedelta(
            seconds=self.config.quote_window_timeout_secs
        )
        self.log.info(f"ENTRY window (session {s_date}): due={sorted(due)} targets={targets}")

    def on_quote_tick(self, tick) -> None:
        if self._halted:
            return
        if self._state not in (_State.AWAITING_EXIT, _State.AWAITING_ENTRY) or self._ticks is None:
            return
        self._ticks.record(tick.instrument_id, tick)
        if self._ticks.ready():
            self._execute()

    # ── execution ────────────────────────────────────────────────────────
    def _fresh_quote(self, canonical: str):
        """Return (mid, bid, ask, instrument) for a fresh quote, else None."""
        assert self._ticks is not None
        iid = self._iid_by_canonical[canonical]
        quote = self._ticks.latest(iid)
        instrument = self.cache.instrument(iid)
        if quote is None or instrument is None:
            self.log.warning(f"No quote/instrument for {canonical}; deferring")
            return None
        age = self.clock.utc_now().timestamp() - quote.ts_event / 1e9
        if age > self.config.max_quote_age_secs:
            self.log.warning(f"Stale quote for {canonical} ({age:.0f}s old); deferring")
            return None
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        return (bid + ask) / 2.0, bid, ask, instrument

    def _execute(self) -> None:
        if self._state is _State.AWAITING_EXIT:
            self._execute_exit()
        elif self._state is _State.AWAITING_ENTRY:
            self._execute_entry()
        self._finish_window()

    def _execute_exit(self) -> None:
        """Flatten the swap-negative legs worth overlaying (carry-aware, MARKET)."""
        bnow = self._now_broker() or self.clock.utc_now()
        flattened, held = [], []
        for canonical in list(self._pending):
            fq = self._fresh_quote(canonical)
            if fq is None:
                continue  # market closed / stale → leg held this rollover
            mid, bid, ask, instrument = fq
            iid = self._iid_by_canonical[canonical]
            native = self._native_by_canonical[canonical]
            position = float(self.portfolio.net_position(iid))
            if abs(position) <= 1e-12:
                continue
            swap = self._fetch_swap(native)
            if swap is None:
                self.log.warning(f"No swap info for {native}; holding {canonical}")
                held.append(canonical)
                continue
            triple = rollover_schedule.is_triple_night(bnow, swap.triple_weekday)
            swap_bps = rollover_market.swap_bps_for(swap, position=position, mid=mid, triple=triple)
            round_trip = 2.0 * half_spread_bps(bid, ask)  # exit + expected reopen half-spread
            action = decide_exit_leg(
                position=position, swap_bps_position=swap_bps,
                round_trip_half_spread_bps=round_trip, params=self._overlay,
            )
            if action is LegAction.CROSS_MARKET:
                self._market_close(iid, instrument, position, canonical, bid, ask)
                flattened.append(f"{canonical}(swap={swap_bps:.2f}bps,rt={round_trip:.2f})")
            else:
                held.append(f"{canonical}(swap={swap_bps:.2f}bps)")
        self.log.info(f"EXIT done: flattened={flattened} held={held}")
        # Latch only once we acted with an open market (got at least one fresh quote),
        # else retry next poll so we don't miss the window on a transient no-quote.
        if flattened or held:
            self._exit_done_for = rollover_schedule.exit_boundary_date(bnow)

    def _execute_entry(self) -> None:
        """Re-establish / rebalance to the fresh forecast (delta, MARKET)."""
        basis = self._sizing_basis()
        if basis is None or basis <= 0:
            self.log.error("No account equity / sizing basis; skipping entry")
            return
        sizing = build_sizing_config(
            sizing_basis_usd=basis,
            lot_size_ceiling=self.config.lot_size_ceiling,
            min_rebalance_lots=self.config.min_rebalance_lots,
            min_rebalance_notional_usd=self.config.min_rebalance_notional_usd,
        )
        acted, orders_sent = [], 0
        for canonical, frac in self._pending.items():
            fq = self._fresh_quote(canonical)
            if fq is None:
                continue
            mid, _bid, _ask, instrument = fq
            iid = self._iid_by_canonical[canonical]
            symbol = symbol_info_from_nautilus_instrument(instrument)
            target = target_signed_lots(position_fraction=frac, price=mid, symbol=symbol, sizing=sizing)
            if target is None:
                acted.append(canonical)  # rounds below volume_min → no opinion, treat as done
                continue
            current = float(self.portfolio.net_position(iid))
            order = net_rebalance(
                target_signed_lots=target, current_signed_lots=current, symbol=symbol, price=mid,
                min_rebalance_lots=self.config.min_rebalance_lots,
                min_rebalance_notional_usd=self.config.min_rebalance_notional_usd,
            )
            acted.append(canonical)
            if order is None:
                continue
            self._submit_market(iid, instrument, order.side, order.volume, canonical, _bid, _ask)
            orders_sent += 1
        if acted:
            self.log.info(f"ENTRY done: acted={sorted(acted)} orders={orders_sent}")
            # Latch the session only when every due symbol got a fresh quote (acted);
            # otherwise retry so a late-opening symbol still enters this session.
            if len(acted) == len(self._pending):
                self._entry_done_for = rollover_schedule.session_date(
                    self._now_broker() or self.clock.utc_now()
                )

    def _market_close(self, iid: InstrumentId, instrument, position: float,
                      canonical: str, bid: float, ask: float) -> None:
        side = NautilusOrderSide.SELL if position > 0 else NautilusOrderSide.BUY
        qty = Quantity(abs(position), instrument.size_precision)
        order = self.order_factory.market(instrument_id=iid, order_side=side, quantity=qty)
        self._record_slip_submit(canonical=canonical, side=side.name, qty=abs(position),
                                 bid=bid, ask=ask, client_order_id=order.client_order_id,
                                 intent='EXIT')
        self.submit_order(order)
        self.log.info(f"Submit CLOSE {side.name} {qty} {iid}")

    def _submit_market(self, iid: InstrumentId, instrument, mt5_side, volume: float,
                       canonical: str, bid: float, ask: float) -> None:
        side = NautilusOrderSide.BUY if mt5_side is MT5OrderSide.BUY else NautilusOrderSide.SELL
        qty = Quantity(volume, instrument.size_precision)
        order = self.order_factory.market(instrument_id=iid, order_side=side, quantity=qty)
        self._record_slip_submit(canonical=canonical, side=side.name, qty=volume,
                                 bid=bid, ask=ask, client_order_id=order.client_order_id,
                                 intent='ENTRY',
                                 target_fraction=self._pending.get(canonical))
        self.submit_order(order)
        self.log.info(f"Submit {side.name} {qty} {iid}")

    def _record_slip_submit(self, *, canonical: str, side: str, qty: float, bid: float,
                            ask: float, client_order_id,
                            intent: str | None = None,
                            target_fraction: float | None = None) -> None:
        """Capture the LIVE quote at order submit for slippage tracking (never raises).

        The spread + expected fill (touch) can only be observed live — these CFDs serve
        no historical ticks, so post-hoc reconstruction is impossible. The deal reader
        (scripts/track_slippage.py) joins these by client_order_id (= the MT5 deal
        comment) to the actual fill price to compute realised slippage over time. No-op
        when live-state publishing is disabled (tests / sandbox_sim).

        ``intent`` (``'EXIT'`` or ``'ENTRY'``) and ``target_fraction`` are forwarded
        to :func:`slippage.record_submit` when provided; absent → legacy record shape.
        """
        if self._live_dir is None:
            return
        try:
            from deployment.live.monitoring import slippage
            bnow = self._now_broker() or self.clock.utc_now()
            slippage.record_submit(
                self.config.broker, canonical=canonical, side=side, qty=float(qty),
                bid=float(bid), ask=float(ask), client_order_id=str(client_order_id),
                broker_time_iso=bnow.isoformat(),
                intent=intent, target_fraction=target_fraction,
            )
        except Exception as exc:  # noqa: BLE001 - capture must never break trading
            self.log.warning(f"slippage submit-capture failed: {exc}")

    def _record_forecast_decision(self, s_date, due: list[str], result) -> None:
        """Durably capture what was forecast at this ENTRY decision (never raises).

        One JSONL row per due canonical → ``live_state/forecasts.jsonl``
        (migration plan §7.4): previously the targets existed only in the
        overwritten snapshot, so "what did we target on day X" was
        unanswerable. No-op when live-state publishing is disabled.
        """
        if self._live_dir is None:
            return
        try:
            import hashlib
            import json as _json

            from deployment.live.monitoring import forecast_log

            cfg = self.config
            engine_cfg = {
                "vault_root": str(cfg.vault_root),
                "target_volatility": cfg.target_volatility,
                "max_position_pct": cfg.max_position_pct,
                "idm_max": cfg.idm_max,
                "warmup_min_bars": cfg.warmup_min_bars,
                "prediction_daily_max_bars": cfg.prediction_daily_max_bars,
            }
            engine_hash = hashlib.sha256(
                _json.dumps(engine_cfg, sort_keys=True).encode()
            ).hexdigest()[:16]
            rows = [
                {
                    "as_of": str(s_date),
                    "canonical": canonical,
                    "forecast_score": float(result.forecast_scores.get(canonical, 0.0)),
                    "target_fraction": float(result.targets.get(canonical, 0.0)),
                    "engine_config_hash": engine_hash,
                    "vault_root": str(cfg.vault_root),
                    "warmup_ready": 1 if result.ready else 0,
                    "ts": self.clock.utc_now().isoformat(),
                }
                for canonical in due
            ]
            forecast_log.record_forecasts(cfg.broker, rows)
        except Exception as exc:  # noqa: BLE001 - capture must never break trading
            self.log.warning(f"forecast decision-capture failed: {exc}")

    def _finish_window(self) -> None:
        if self._ticks is not None and self._ticks.active():
            self._ticks.close_window()
        self._pending = {}
        self._window_deadline = None
        self._state = _State.IDLE

    def _sizing_basis(self) -> float | None:
        if self.config.sizing_basis_usd > 0:
            return self.config.sizing_basis_usd
        account = self.portfolio.account(self._venue)
        if account is None:
            return None
        money = account.balance_total(self._currency)
        return money.as_double() if money is not None else None

    def on_order_filled(self, event) -> None:
        self.log.info(
            f"Fill: {event.order_side.name} {event.last_qty} @ {event.last_px} "
            f"{event.instrument_id}"
        )

    # ── live-state publication (node-mediated dashboard feed) ─────────────
    # The node is the SOLE owner of its MT5 terminal, so the dashboard reads a
    # published JSON snapshot and writes a command file rather than touching MT5.
    # Everything here is wrapped so a monitoring failure can NEVER break trading.
    def _start_live_state(self) -> None:
        """Begin publishing the snapshot + watching the command file (if enabled).

        No-op when ``live_state_dir`` is empty (tests / sandbox_sim). Restores the
        risk baseline and the durable HALT marker so a crash/restart never silently
        resumes trading after a kill switch. Never raises.
        """
        raw = self.config.live_state_dir
        if not raw:
            return
        try:
            self._live_dir = Path(raw)
            self._baseline = live_state.load_baseline(live_state.baseline_path(self._live_dir))
            halt = live_state.load_halt(live_state.halt_path(self._live_dir))
            if halt.halted:
                self._halted = True
                self.log.warning(
                    "live-state: node starting HALTED (kill switch fired "
                    f"{halt.at}); clear halt.json and restart to resume trading."
                )
            self.clock.set_timer(
                name=_SNAPSHOT_TIMER,
                interval=timedelta(seconds=_SNAPSHOT_SECS),
                callback=self._on_snapshot_timer,
            )
            self.log.info(
                f"live-state: publishing to {self._live_dir} (tier={self.config.exec_tier})"
            )
        except Exception as exc:  # noqa: BLE001 - monitoring must never break trading
            self.log.error(f"live-state: failed to start publishing: {exc}")
            self._live_dir = None

    def _on_snapshot_timer(self, _event) -> None:
        """Publish one snapshot + process any pending command. Never raises."""
        if self._live_dir is None:
            return
        try:
            self._snap_ticks += 1
            now = self.clock.utc_now()
            if self._should_refresh_forecast(now):
                self._refresh_forecast(now)
            slow = self._snap_ticks % _SLOW_REFRESH_EVERY == 1  # first tick + every Nth
            self._publish_snapshot(now=now, sample_equity=slow)
            self._process_command()
        except Exception as exc:  # noqa: BLE001 - monitoring must never break trading
            self.log.error(f"live-state: snapshot tick failed: {exc}")

    def _should_refresh_forecast(self, now: datetime) -> bool:
        """Whether to recompute the dashboard forecast on the event-loop thread.

        ``evaluate()`` is a synchronous fit+predict, so it must only run while IDLE
        (never during an EXIT/ENTRY window), after startup reconciliation, and at
        most once per :data:`_FORECAST_MAX_AGE_SECS` — the entry path keeps the
        forecast fresh for free, so this is just a slow idle backstop.
        """
        if self._engine is None or self._halted or self._state is not _State.IDLE:
            return False
        if not self._ready_to_trade or not self._startup_ready(now):
            return False
        last = self._last_forecast_refreshed_at
        if self._last_forecast is None or last is None:
            return True
        return (now - last).total_seconds() > _FORECAST_MAX_AGE_SECS

    def _refresh_forecast(self, now: datetime) -> None:
        try:
            self._last_forecast = self._evaluate_targets()
            self._last_forecast_refreshed_at = now
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"live-state: forecast refresh failed: {exc}")

    def _quote_mid_age(self, iid: InstrumentId) -> tuple[float | None, float | None]:
        """(mid, age_seconds) for the cached quote, or (None, None) if absent."""
        quote = self.cache.quote_tick(iid)
        if quote is None:
            return None, None
        mid = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
        age = self.clock.utc_now().timestamp() - quote.ts_event / 1e9
        return mid, age

    def _account_snapshot(self) -> tuple[dict, float | None, float | None]:
        """(account dict, equity, balance) from the Nautilus portfolio — no MT5 traffic.

        The MT5 adapter pushes only *balance* to the Nautilus account, so equity is
        reconstructed as ``balance + floating_pnl`` (floating from the portfolio's
        unrealised P&L, converted to the account currency). If the P&L fetch fails,
        equity is left ``None`` (NOT silently == balance, which would hide an
        open-position drawdown). Margin is not on the Nautilus account.
        """
        balance = floating = equity = None
        account = self.portfolio.account(self._venue)
        if account is not None:
            money = account.balance_total(self._currency)
            balance = money.as_double() if money is not None else None
            try:
                pnls = self.portfolio.unrealized_pnls(self._venue, target_currency=self._currency)
                floating = sum(m.as_double() for m in pnls.values() if m is not None)
            except Exception:  # noqa: BLE001 - portfolio pnl unavailable
                floating = None
            if balance is not None and floating is not None:
                equity = balance + floating
        acct = {
            "login": self.config.account_login,
            "server": self.config.account_server,
            "currency": self._currency.code,
            "balance": balance,
            "floating_pnl": floating,
            "equity": equity,
        }
        return acct, equity, balance

    def _positions_snapshot(self) -> tuple[list[dict], float, bool]:
        """(rows, gross_notional, marks_fresh).

        ``marks_fresh`` is False when any open leg lacks a quote fresh within
        ``max_quote_age_secs`` — between rollover windows the demand-tick model holds
        no subscription, so cached mids freeze; the caller uses this to avoid
        recording a stale equity sample / anchoring the daily gauge off a frozen mid.
        """
        canonical_by_iid = {iid: c for c, iid in self._iid_by_canonical.items()}
        rows: list[dict] = []
        gross = 0.0
        marks_fresh = True
        for p in self.cache.positions_open(venue=self._venue):
            iid = p.instrument_id
            instrument = self.cache.instrument(iid)
            net_qty = float(p.signed_qty)
            mid, age = self._quote_mid_age(iid)
            stale = mid is None or age is None or age > self.config.max_quote_age_secs
            if stale:
                marks_fresh = False
            # Notional uses the contract size (lot_size), NOT multiplier — MT5
            # CFD/FX instruments carry multiplier==1 and the real contract size in
            # lot_size (matches execution.mt5_rebalancer / sizing).
            contract = 1.0
            if instrument is not None:
                try:
                    contract = float(instrument.lot_size)
                except Exception:  # noqa: BLE001
                    contract = 1.0
            notional = abs(net_qty) * mid * contract if mid is not None else None
            if notional is not None:
                gross += notional
            upnl = None
            try:  # per-instrument P&L (unrealized_pnls is keyed by currency, not iid)
                m = self.portfolio.unrealized_pnl(iid, target_currency=self._currency)
                upnl = m.as_double() if m is not None else None
            except Exception:  # noqa: BLE001
                upnl = None
            rows.append({
                "canonical": canonical_by_iid.get(iid),
                "symbol": iid.symbol.value,
                "instrument_id": str(iid),
                "side": "LONG" if net_qty > 0 else "SHORT" if net_qty < 0 else "FLAT",
                "net_qty": net_qty,
                "avg_px_open": float(p.avg_px_open),
                "last_px": mid,
                "mark_age_secs": age,
                "mark_stale": stale,
                "unrealized_pnl": upnl,
                "notional": notional,
            })
        return rows, gross, marks_fresh

    def _targets_snapshot(self) -> list[dict]:
        fr = self._last_forecast
        if fr is None or not fr.ready:
            return []
        basis = self._sizing_basis()
        sizing = None
        if basis and basis > 0:
            sizing = build_sizing_config(
                sizing_basis_usd=basis,
                lot_size_ceiling=self.config.lot_size_ceiling,
                min_rebalance_lots=self.config.min_rebalance_lots,
                min_rebalance_notional_usd=self.config.min_rebalance_notional_usd,
            )
        rows: list[dict] = []
        for canonical, iid in self._iid_by_canonical.items():
            frac = float(fr.targets.get(canonical, 0.0))
            score = float(fr.forecast_scores.get(canonical, 0.0))
            current = float(self.portfolio.net_position(iid))
            target_qty = drift = None
            mid, _age = self._quote_mid_age(iid)
            instrument = self.cache.instrument(iid)
            if sizing is not None and mid is not None and instrument is not None:
                symbol = symbol_info_from_nautilus_instrument(instrument)
                t = target_signed_lots(position_fraction=frac, price=mid, symbol=symbol, sizing=sizing)
                if t is not None:
                    target_qty = float(t)
                    drift = float(t) - current
            rows.append({
                "canonical": canonical,
                "target_fraction": frac,
                "forecast_score": score,
                "target_qty": target_qty,
                "current_qty": current,
                "drift": drift,
            })
        return rows

    def _warmup_snapshot(self) -> dict | None:
        fr = self._last_forecast
        if fr is None:
            return None
        return {
            "ready": fr.ready,
            "min_bars": fr.warmup.min_bars,
            "as_of": fr.as_of.isoformat() if fr.as_of is not None else None,
            "per_ticker": [
                {"ticker": t.ticker, "bars": t.bars, "ready": t.ready}
                for t in fr.warmup.per_ticker
            ],
        }

    def _broker_date(self) -> str:
        try:
            tz = brokers.broker_tzinfo(self.config.broker)
            return self.clock.utc_now().astimezone(tz).date().isoformat()
        except Exception:  # noqa: BLE001
            return self.clock.utc_now().date().isoformat()

    def _publish_snapshot(self, *, now: datetime, sample_equity: bool) -> None:
        assert self._live_dir is not None
        now_iso = now.isoformat()
        account, equity, balance = self._account_snapshot()
        positions, gross, marks_fresh = self._positions_snapshot()
        account["leverage_in_use"] = (gross / equity) if (gross and equity) else None
        broker_date = self._broker_date()

        # Only advance the equity baseline / sample the curve when the marks are
        # fresh, so a frozen mid (no live subscription between rollover windows) is
        # never recorded as a real equity point or used to anchor the daily gauge.
        if equity is not None and marks_fresh:
            new_baseline = live_state.update_baseline(
                self._baseline,
                equity=equity,
                balance=balance,
                broker_date=broker_date,
                now_iso=now_iso,
                initial_balance=self.config.initial_balance,
                prev_observed_date=self._seen_broker_date,
            )
            if new_baseline != self._baseline:
                self._baseline = new_baseline
                try:
                    live_state.save_baseline(
                        live_state.baseline_path(self._live_dir), self._baseline
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"live-state: baseline save failed: {exc}")
            self._seen_broker_date = broker_date
            if sample_equity:
                try:
                    live_state.append_equity_sample(
                        live_state.equity_path(self._live_dir), ts_iso=now_iso, equity=equity
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"live-state: equity sample failed: {exc}")
                try:
                    live_state.append_equity_sample_durable(
                        live_state.equity_durable_path(self._live_dir),
                        ts_iso=now_iso, equity=equity,
                        balance=balance,
                        floating_pnl=account.get("floating_pnl"),
                        gross_notional=gross,
                        marks_fresh=marks_fresh,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"live-state: durable equity sample failed: {exc}")

        snapshot = {
            "schema_version": live_state.SCHEMA_VERSION,
            "broker": self.config.broker,
            "venue": self.config.venue,
            "exec_tier": self.config.exec_tier,
            "account": account,
            "positions": positions,
            "gross_notional": gross,
            "marks_fresh": marks_fresh,
            "targets": self._targets_snapshot(),
            "warmup": self._warmup_snapshot(),
            "strategy_state": "HALTED" if self._halted else self._state.name,
            "halted": self._halted,
            "ready_to_trade": self._ready_to_trade,
            "initial_balance": self.config.initial_balance or self._baseline.account_start_equity,
            "risk_baseline": {
                "account_start_equity": self._baseline.account_start_equity,
                "account_start_at": self._baseline.account_start_at,
                "day_start_equity": self._baseline.day_start_equity,
                "day_start_balance": self._baseline.day_start_balance,
                "day_start_date": self._baseline.day_start_date,
                "day_start_estimated": self._baseline.day_start_estimated,
            },
            "last_command_result": self._last_command_result,
            "node_started_at": self._live_started_iso,
            "ts": now_iso,
        }
        live_state.write_snapshot(live_state.snapshot_path(self._live_dir), snapshot)

    # ── command channel (dashboard → node) ───────────────────────────────
    def _process_command(self) -> None:
        assert self._live_dir is not None
        cpath = live_state.command_path(self._live_dir)
        cmd = live_state.read_command(cpath)
        if cmd is None or cmd.id in self._processed_command_ids:
            return
        self._processed_command_ids.add(cmd.id)
        # A command issued at/before node start is stale (a prior-session file) and
        # must never re-fire on restart. The file is deleted after handling so it
        # cannot be re-read regardless; this is belt-and-suspenders.
        if self._command_is_stale(cmd):
            live_state.clear_command(cpath)
            return
        try:
            if cmd.action != "flatten":
                self._record_command_result(cmd, "rejected", f"unknown action {cmd.action!r}")
            elif str(self.config.exec_tier).lower() == "live":
                self._record_command_result(
                    cmd, "rejected", "demo-only: flatten is disabled on a live/funded account"
                )
            else:
                status, detail = self._flatten_all()
                self._record_command_result(cmd, status, detail)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"live-state: flatten command failed: {exc}")
            self._record_command_result(cmd, "error", str(exc))
        finally:
            live_state.clear_command(cpath)  # consume: never re-read after restart

    def _command_is_stale(self, cmd: live_state.Command) -> bool:
        if self._live_started_iso is None:
            return False
        try:
            issued = datetime.fromisoformat(cmd.issued_at)
            started = datetime.fromisoformat(self._live_started_iso)
        except ValueError:
            return True  # unparseable timestamp → treat as stale (safer)
        return issued <= started

    def _flatten_all(self) -> tuple[str, str]:
        """Close every magic-tagged broker TICKET per-ticket, then HALT (no re-entry).

        Closes via raw MT5 per-ticket ``DEAL`` orders carrying ``position=ticket`` —
        the ONLY correct flatten on a HEDGING account, where a plain offsetting
        market order opens an opposing ticket instead of reducing (the netting-delta
        hazard documented in node_builder). Mirrors manual_trade.cmd_flatten. The
        result is real: each ``order_send`` is synchronous, so the status reflects
        actual broker fills/rejections (not an optimistic "submitted").

        Halting (in-memory + a durable marker) before/around closing makes the
        dashboard 'close all' safe against the running node: it never re-opens toward
        target on the next rollover decision, even across a crash+restart.
        """
        import MetaTrader5 as mt5  # the node owns the terminal; the package is process-global

        # Halt FIRST (memory + disk) so a failure mid-close can never be followed by
        # a re-entry, and a crash mid-flatten stays halted on restart.
        self._halted = True
        self._finish_window()
        if self._live_dir is not None:
            try:
                live_state.save_halt(
                    live_state.halt_path(self._live_dir),
                    live_state.HaltState(halted=True, at=self.clock.utc_now().isoformat()),
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"live-state: halt persist failed: {exc}")

        magic = self._magic_number()
        targets = [p for p in (mt5.positions_get() or ()) if getattr(p, "magic", None) == magic]
        closed, failed = 0, []
        for p in targets:
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                failed.append(f"{p.symbol}#{p.ticket}:no-tick")
                continue
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,  # close THIS ticket (hedging-safe)
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 50,
                "magic": magic,
                "comment": "vault-flatten",
                "type_filling": self._filling_mode(mt5, p.symbol),
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            else:
                code = result.retcode if result is not None else mt5.last_error()
                failed.append(f"{p.symbol}#{p.ticket}:retcode={code}")

        remaining = [p for p in (mt5.positions_get() or ()) if getattr(p, "magic", None) == magic]
        if remaining:
            status = "error"
        elif failed:
            status = "partial"
        else:
            status = "executed"
        detail = f"closed {closed}/{len(targets)} ticket(s)"
        if failed:
            detail += f"; FAILED: {failed}"
        if remaining:
            detail += f"; STILL OPEN: {[f'{p.symbol}#{p.ticket}' for p in remaining]}"
        self.log.warning(f"[live-state] FLATTEN ALL + HALT -> {status}: {detail}")
        return status, detail

    def _magic_number(self) -> int:
        try:
            return int(brokers.execution_rules(self.config.broker).magic_number)
        except Exception:  # noqa: BLE001
            return 510

    @staticmethod
    def _filling_mode(mt5, symbol: str) -> int:
        """Pick a fill mode the symbol accepts (IOC → FOK → RETURN)."""
        si = mt5.symbol_info(symbol)
        fm = int(getattr(si, "filling_mode", 0)) if si is not None else 0
        if fm & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        if fm & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _record_command_result(self, cmd: live_state.Command, status: str, detail: str) -> None:
        self._last_command_result = {
            "id": cmd.id,
            "action": cmd.action,
            "status": status,
            "detail": detail,
            "at": self.clock.utc_now().isoformat(),
        }
        self.log.info(f"live-state: command {cmd.id} {cmd.action} -> {status}: {detail}")


__all__ = ["VaultRebalanceConfig", "VaultRebalanceStrategy"]
