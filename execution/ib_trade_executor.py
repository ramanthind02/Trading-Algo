"""
Thin IB API wrapper for order placement and position reconciliation.

Responsibilities (and nothing else):
- Connect/disconnect on a dedicated client_id (separate from the data-fetch
  client so reads and writes never contend).
- Fetch ``managedAccounts`` for the safety gate.
- Fetch current positions for rebalancer + reconciliation.
- Fetch last price snapshots for notional sizing.
- Place MKT DAY orders for fractional ETF shares.
- Wait for each order to reach a terminal state (FILLED / CANCELLED /
  REJECTED) or to time out.

No sizing math lives here. No safety logic lives here. No Telegram here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from execution.models import OrderIntent, OrderResult, OrderStatus


logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


@dataclass
class _OrderState:
    status: str = "PendingSubmit"
    filled: Decimal = Decimal("0")
    avg_price: Optional[float] = None
    commission: Optional[float] = None
    error_code: Optional[int] = None
    error_text: Optional[str] = None


class IBTradeClient(EClient, EWrapper):
    """IB client dedicated to order placement.

    Use a ``client_id`` that is distinct from any data-fetch client to avoid
    request-ID collisions and callback interleaving.
    """

    def __init__(self, host: str, port: int, client_id: int) -> None:
        EClient.__init__(self, self)
        self._host = host
        self._port = port
        self._client_id = client_id

        self._next_order_id: int = 0
        self._connected_event = threading.Event()
        self._accounts_event = threading.Event()
        self._positions_event = threading.Event()

        self.managed_accounts: List[str] = []
        self.positions: Dict[str, Decimal] = {}
        self.last_prices: Dict[str, float] = {}
        self._order_states: Dict[int, _OrderState] = {}
        self._order_intents: Dict[int, OrderIntent] = {}
        self._state_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None

    # --- connection ---

    def connect_and_start(self, timeout: float = 5.0) -> None:
        logger.info(f"Connecting IBTradeClient to {self._host}:{self._port} (client_id={self._client_id})")
        EClient.connect(self, self._host, self._port, self._client_id)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        if not self._connected_event.wait(timeout):
            raise ConnectionError("Timed out waiting for nextValidId from TWS")

    def disconnect_and_stop(self) -> None:
        if self.isConnected():
            EClient.disconnect(self)

    # --- EWrapper callbacks ---

    def nextValidId(self, orderId: int) -> None:
        self._next_order_id = orderId
        self._connected_event.set()
        logger.info(f"TWS handshake complete. Next valid order id: {orderId}")

    def managedAccounts(self, accountsList: str) -> None:
        self.managed_accounts = [a.strip() for a in accountsList.split(",") if a.strip()]
        self._accounts_event.set()

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        symbol = contract.symbol
        self.positions[symbol] = Decimal(str(position))

    def positionEnd(self) -> None:
        self._positions_event.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice) -> None:
        with self._state_lock:
            state = self._order_states.setdefault(orderId, _OrderState())
            state.status = status
            try:
                state.filled = Decimal(str(filled))
            except Exception:
                pass
            if avgFillPrice and avgFillPrice > 0:
                state.avg_price = float(avgFillPrice)

    def error(self, *args) -> None:
        # Support both old (reqId, errorCode, errorString) and newer signatures.
        reqId: int = -1
        code: int = 0
        msg: str = ""
        if len(args) >= 3:
            reqId = int(args[0]) if args[0] is not None else -1
            # new signature may have a timestamp in args[1]
            if len(args) == 5 or (len(args) == 4 and isinstance(args[1], str) and ":" in str(args[1])):
                code = int(args[2])
                msg = str(args[3])
            else:
                code = int(args[1])
                msg = str(args[2])
        if reqId in self._order_states:
            with self._state_lock:
                state = self._order_states[reqId]
                state.error_code = code
                state.error_text = msg
        logger.warning(f"IB error reqId={reqId} code={code} msg={msg}")

    # --- synchronous helpers ---

    def fetch_managed_accounts(self, timeout: float = 5.0) -> List[str]:
        # managedAccounts callback is usually pushed automatically on connect.
        self._accounts_event.wait(timeout)
        return list(self.managed_accounts)

    def fetch_positions(self, timeout: float = 10.0) -> Dict[str, Decimal]:
        self.positions = {}
        self._positions_event.clear()
        self.reqPositions()
        self._positions_event.wait(timeout)
        self.cancelPositions()
        return dict(self.positions)

    def _next_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id

    # --- order placement ---

    @staticmethod
    def _build_etf_contract(symbol: str) -> Contract:
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    @staticmethod
    def _build_mkt_order(intent: OrderIntent, account: str) -> Order:
        """Build a whole-share MKT order.

        IB API rejected both fractional ``totalQuantity`` (error 10243) and
        ``cashQty`` (error 10244) on this account, so we round to whole shares.
        Enabling fractional shares is an account-level permission in Client
        Portal; once turned on, switch this to ``totalQuantity = Decimal(...)``.
        """
        o = Order()
        o.action = intent.side.value
        o.orderType = "MKT"
        o.totalQuantity = Decimal(str(int(round(float(intent.shares)))))
        o.tif = "DAY"
        o.outsideRth = False
        o.account = account
        o.eTradeOnly = False
        o.firmQuoteOnly = False
        return o

    def submit_order(self, intent: OrderIntent, account: str) -> int:
        """Place a single order. Returns the IB order id assigned."""
        order_id = self._next_id()
        contract = self._build_etf_contract(intent.etf)
        order = self._build_mkt_order(intent, account)
        with self._state_lock:
            self._order_states[order_id] = _OrderState()
            self._order_intents[order_id] = intent
        logger.info(f"Placing {intent.side.value} {intent.shares} {intent.etf} (orderId={order_id})")
        self.placeOrder(order_id, contract, order)
        return order_id

    def wait_for_terminal(self, order_id: int, timeout_seconds: int) -> OrderResult:
        """Block until ``order_id`` reaches a terminal state or timeout fires."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._state_lock:
                state = self._order_states[order_id]
                if state.status in _TERMINAL_STATUSES:
                    return self._to_result(order_id, state, override_status=None)
                if state.error_code is not None and state.error_code >= 200 and state.error_code != 202:
                    return self._to_result(order_id, state, override_status=OrderStatus.REJECTED)
            time.sleep(0.25)
        logger.warning(f"Order {order_id} timed out. Cancelling.")
        self.cancelOrder(order_id, "")
        time.sleep(1.0)
        with self._state_lock:
            return self._to_result(order_id, self._order_states[order_id], override_status=OrderStatus.TIMED_OUT)

    def cancel_all_open(self, order_ids: List[int]) -> None:
        for oid in order_ids:
            try:
                self.cancelOrder(oid, "")
            except Exception as e:
                logger.error(f"cancelOrder({oid}) failed: {e}")

    def _to_result(self, order_id: int, state: _OrderState, override_status: Optional[OrderStatus]) -> OrderResult:
        intent = self._order_intents[order_id]
        if override_status is not None:
            status = override_status
        elif state.status == "Filled":
            status = OrderStatus.FILLED
        elif state.status in {"Cancelled", "ApiCancelled"}:
            status = OrderStatus.CANCELLED
        elif state.status == "Inactive":
            status = OrderStatus.REJECTED
        else:
            status = OrderStatus.ERROR
        return OrderResult(
            intent=intent,
            ib_order_id=order_id,
            status=status,
            filled_shares=state.filled,
            avg_fill_price=state.avg_price,
            commission_usd=state.commission,
            error_code=state.error_code,
            error_text=state.error_text,
        )
