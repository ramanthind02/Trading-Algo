"""
MT5 trade executor — the only module that imports the ``MetaTrader5``
Python package. Translates broker dicts/namedtuples to the typed value
objects in :mod:`execution.mt5_models` at the boundary so the rest of
the codebase remains testable on non-Windows machines.

Usage
-----
::

    with MT5TradeExecutor.connect_and_verify(account_cfg) as mt5x:
        acct      = mt5x.fetch_account_info()
        positions = mt5x.fetch_all_positions()
        sym       = mt5x.fetch_symbol_info("US100.cash")
        tick      = mt5x.fetch_tick("US100.cash")
        ...
        result    = mt5x.place_market_order(...)
        result    = mt5x.close_ticket(ticket=12345, volume=0.10)

The context manager guarantees ``mt5.shutdown()`` is always called so we
can iterate cleanly across N accounts (the MT5 package only supports one
active terminal session per Python process).

If the ``MetaTrader5`` package is not installed (e.g. running on macOS
or Linux CI), :class:`MT5TradeExecutor` raises ``ImportError`` only on
first connection attempt, NOT at import time — so unit tests in other
modules continue to import cleanly.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from execution.mt5_models import (
    MT5AccountInfo,
    MT5Position,
    MT5SymbolInfo,
    MT5Tick,
    OrderResult,
    OrderSide,
    RETRYABLE_RETCODES,
    RebalanceAction,
    RebalanceActionKind,
    TRADE_RETCODE_DONE,
    TradeMode,
)


logger = logging.getLogger(__name__)


# Lazy import — many devs run macOS/Linux for code review and tests.
try:  # pragma: no cover - environment-dependent
    import MetaTrader5 as _mt5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:  # pragma: no cover
    _mt5 = None  # type: ignore
    _MT5_AVAILABLE = False


def _require_mt5() -> Any:
    if not _MT5_AVAILABLE or _mt5 is None:  # pragma: no cover
        raise ImportError(
            "The MetaTrader5 Python package is not installed. "
            "Install it on a Windows machine where the MT5 terminal is also installed: "
            "  pip install MetaTrader5"
        )
    return _mt5


# ---------------------------------------------------------------------------
# Trade-mode mapping
# ---------------------------------------------------------------------------


def _trade_mode_from_int(value: int) -> TradeMode:
    mt5 = _require_mt5()
    if value == mt5.SYMBOL_TRADE_MODE_FULL:
        return TradeMode.FULL
    if value == mt5.SYMBOL_TRADE_MODE_LONGONLY:
        return TradeMode.LONG_ONLY
    if value == mt5.SYMBOL_TRADE_MODE_SHORTONLY:
        return TradeMode.SHORT_ONLY
    if value == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        return TradeMode.CLOSE_ONLY
    if value == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return TradeMode.DISABLED
    return TradeMode.DISABLED  # safest unknown default


def _order_side_from_position_type(position_type: int) -> OrderSide:
    mt5 = _require_mt5()
    return OrderSide.BUY if position_type == mt5.POSITION_TYPE_BUY else OrderSide.SELL


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class MT5TradeExecutor:
    """Thin typed wrapper around the ``MetaTrader5`` package.

    Use :meth:`connect_and_verify` (classmethod) as the entrypoint; it
    initialises a terminal session, logs in, and verifies the account
    matches the expected login/server before yielding control.
    """

    login: int
    password: str
    server: str
    terminal_path: Optional[str] = None
    timeout_ms: int = 60_000
    _initialized: bool = field(default=False, init=False, repr=False)

    # ------------------------- lifecycle -------------------------

    def initialize(self) -> None:
        """Initialise the MT5 terminal + log in. Raises on failure."""
        mt5 = _require_mt5()
        kwargs: Dict[str, Any] = {
            "login": self.login,
            "password": self.password,
            "server": self.server,
            "timeout": self.timeout_ms,
        }
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        ok = mt5.initialize(**kwargs)
        if not ok:
            err = mt5.last_error()
            raise RuntimeError(
                f"mt5.initialize failed for login={self.login} server={self.server!r}: {err}"
            )
        # `initialize` can return True without actually authenticating; double-check.
        info = mt5.account_info()
        if info is None or int(info.login) != int(self.login):
            err = mt5.last_error()
            raise RuntimeError(
                f"mt5.account_info() did not return the expected login. "
                f"got={None if info is None else info.login}, expected={self.login}, "
                f"last_error={err}"
            )
        self._initialized = True

    def shutdown(self) -> None:
        if not self._initialized:
            return
        mt5 = _require_mt5()
        try:
            mt5.shutdown()
        finally:
            self._initialized = False

    @classmethod
    @contextmanager
    def connect_and_verify(
        cls,
        *,
        login: int,
        password: str,
        server: str,
        terminal_path: Optional[str] = None,
        timeout_ms: int = 60_000,
    ) -> Iterator["MT5TradeExecutor"]:
        ex = cls(
            login=login,
            password=password,
            server=server,
            terminal_path=terminal_path,
            timeout_ms=timeout_ms,
        )
        ex.initialize()
        try:
            yield ex
        finally:
            ex.shutdown()

    # ------------------------- read --------------------------------

    def fetch_account_info(self) -> MT5AccountInfo:
        mt5 = _require_mt5()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() returned None: {mt5.last_error()}")
        terminal = mt5.terminal_info()
        trade_allowed = bool(getattr(terminal, "trade_allowed", True))
        trade_expert = bool(getattr(info, "trade_expert", True))
        return MT5AccountInfo(
            login=int(info.login),
            server=str(info.server),
            currency=str(info.currency),
            balance=float(info.balance),
            equity=float(info.equity),
            margin=float(info.margin),
            margin_free=float(info.margin_free),
            margin_level=float(info.margin_level),
            leverage=int(info.leverage),
            trade_allowed=trade_allowed,
            trade_expert=trade_expert,
        )

    def fetch_symbol_info(self, symbol_name: str) -> MT5SymbolInfo:
        mt5 = _require_mt5()
        # Make sure the symbol is in market watch so we get a tick.
        if not mt5.symbol_select(symbol_name, True):  # pragma: no cover
            raise RuntimeError(
                f"symbol_select({symbol_name!r}) failed: {mt5.last_error()}"
            )
        info = mt5.symbol_info(symbol_name)
        if info is None:
            raise RuntimeError(
                f"symbol_info({symbol_name!r}) returned None: {mt5.last_error()}"
            )
        return MT5SymbolInfo(
            name=str(info.name),
            trade_mode=_trade_mode_from_int(int(info.trade_mode)),
            point=float(info.point),
            digits=int(info.digits),
            trade_contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            spread=int(info.spread),
            visible=bool(info.visible),
        )

    def fetch_tick(self, symbol_name: str) -> MT5Tick:
        mt5 = _require_mt5()
        tick = mt5.symbol_info_tick(symbol_name)
        if tick is None:
            raise RuntimeError(
                f"symbol_info_tick({symbol_name!r}) returned None: {mt5.last_error()}"
            )
        return MT5Tick(
            symbol=symbol_name,
            bid=float(tick.bid),
            ask=float(tick.ask),
            time_utc=datetime.fromtimestamp(int(tick.time), tz=timezone.utc),
        )

    def fetch_all_positions(self) -> List[MT5Position]:
        mt5 = _require_mt5()
        positions = mt5.positions_get()
        if positions is None:
            err = mt5.last_error()
            # positions_get returns None on a hard error but () for "no positions".
            logger.warning(f"positions_get() returned None: {err}")
            return []
        out: List[MT5Position] = []
        for p in positions:
            out.append(
                MT5Position(
                    ticket=int(p.ticket),
                    symbol=str(p.symbol),
                    side=_order_side_from_position_type(int(p.type)),
                    volume=float(p.volume),
                    open_price=float(p.price_open),
                    magic=int(p.magic),
                    open_time_utc=datetime.fromtimestamp(int(p.time), tz=timezone.utc),
                    comment=str(getattr(p, "comment", "") or ""),
                )
            )
        return out

    def order_calc_margin_usd(
        self, *, symbol_name: str, side: OrderSide, volume: float, price: float
    ) -> float:
        """Wrapper around ``mt5.order_calc_margin`` for sizing sanity checks."""
        mt5 = _require_mt5()
        order_type = mt5.ORDER_TYPE_BUY if side is OrderSide.BUY else mt5.ORDER_TYPE_SELL
        margin = mt5.order_calc_margin(order_type, symbol_name, volume, price)
        if margin is None:
            err = mt5.last_error()
            raise RuntimeError(
                f"order_calc_margin failed for {symbol_name} {side.value} {volume}: {err}"
            )
        return float(margin)

    # ------------------------- write -------------------------------

    def _market_order(
        self,
        *,
        symbol_name: str,
        side: OrderSide,
        volume: float,
        magic: int,
        comment: str,
        position_ticket: Optional[int] = None,
        deviation_points: int = 50,
    ) -> Tuple[bool, Optional[int], int, str]:
        """Send a single market order. Returns (success, deal_ticket, retcode, comment)."""
        mt5 = _require_mt5()
        tick = mt5.symbol_info_tick(symbol_name)
        if tick is None:
            err = mt5.last_error()
            return False, None, 0, f"symbol_info_tick failed: {err}"
        price = tick.ask if side is OrderSide.BUY else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side is OrderSide.BUY else mt5.ORDER_TYPE_SELL
        request: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_name,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "deviation": int(deviation_points),
            "magic": int(magic),
            "comment": comment[:31],   # MT5 caps at 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if position_ticket is not None:
            request["position"] = int(position_ticket)
        result = mt5.order_send(request)
        if result is None:
            return False, None, 0, f"order_send returned None: {mt5.last_error()}"
        retcode = int(result.retcode)
        deal = int(getattr(result, "deal", 0)) or None
        success = retcode == TRADE_RETCODE_DONE
        msg = str(getattr(result, "comment", "") or "")
        return success, deal, retcode, msg

    def place_market_order(
        self,
        *,
        symbol_name: str,
        side: OrderSide,
        volume: float,
        magic: int,
        comment: str = "enigma_cfd_prop",
        max_retries: int = 3,
        retry_sleep_seconds: float = 1.0,
        deviation_points: int = 50,
    ) -> OrderResult:
        action = RebalanceAction(
            kind=RebalanceActionKind.OPEN_NEW,
            symbol=symbol_name,
            side=side,
            volume=volume,
        )
        attempts = 0
        last = (False, None, 0, "")
        while attempts < max_retries:
            attempts += 1
            ok, deal, retcode, msg = self._market_order(
                symbol_name=symbol_name,
                side=side,
                volume=volume,
                magic=magic,
                comment=comment,
                deviation_points=deviation_points,
            )
            last = (ok, deal, retcode, msg)
            if ok or retcode not in RETRYABLE_RETCODES:
                break
            time.sleep(retry_sleep_seconds)
        ok, deal, retcode, msg = last
        return OrderResult(
            action=action,
            success=ok,
            deal_ticket=deal,
            retcode=retcode,
            comment=msg,
        )

    def close_ticket(
        self,
        *,
        ticket: int,
        symbol_name: str,
        original_side: OrderSide,
        volume: float,
        magic: int,
        comment: str = "enigma_cfd_close",
        max_retries: int = 3,
        retry_sleep_seconds: float = 1.0,
        deviation_points: int = 50,
    ) -> OrderResult:
        """Send an opposite-side IOC deal with ``position=<ticket>`` so MT5
        treats it as a partial/full close of that specific ticket rather
        than opening a hedge. This is THE critical mechanic for
        hedging-mode accounts."""
        close_side = original_side.opposite()
        action = RebalanceAction(
            kind=RebalanceActionKind.CLOSE_TICKET,
            symbol=symbol_name,
            side=original_side,
            volume=volume,
            ticket=ticket,
        )
        attempts = 0
        last = (False, None, 0, "")
        while attempts < max_retries:
            attempts += 1
            ok, deal, retcode, msg = self._market_order(
                symbol_name=symbol_name,
                side=close_side,
                volume=volume,
                magic=magic,
                comment=comment,
                position_ticket=ticket,
                deviation_points=deviation_points,
            )
            last = (ok, deal, retcode, msg)
            if ok or retcode not in RETRYABLE_RETCODES:
                break
            time.sleep(retry_sleep_seconds)
        ok, deal, retcode, msg = last
        return OrderResult(
            action=action,
            success=ok,
            deal_ticket=deal,
            retcode=retcode,
            comment=msg,
        )
