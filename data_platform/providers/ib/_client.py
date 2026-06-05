"""Minimal TWS API client for data fetching (contract enumeration + daily bars).

Wraps ibapi's EClient/EWrapper with synchronous helpers. Connection params
default to TWS paper (127.0.0.1:7497). Used by the IB adapter modules; live
trading uses its own client in scripts/enigma_live_forecast.py.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pandas as pd

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    _IBAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - ibapi optional at import time
    EClient = object  # type: ignore
    EWrapper = object  # type: ignore
    Contract = object  # type: ignore
    _IBAPI_AVAILABLE = False


@dataclass(frozen=True)
class IbExpiry:
    local_symbol: str          # e.g. "ESM4"
    last_trade: str            # YYYYMMDD, e.g. "20240621"
    exchange: str
    multiplier: str
    con_id: int


class IbDataClient(EClient, EWrapper):  # type: ignore[misc]
    """Synchronous IB data client: enumerate futures expiries + fetch daily bars."""

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._next_id = 0
        self.connected_ok = False
        self._bars: list = []
        self._details: list[IbExpiry] = []
        self._done: dict[int, bool] = {}
        self._errors: dict[int, str] = {}

    # ── connection / ids ────────────────────────────────────────────────
    def nextValidId(self, orderId: int) -> None:
        self._next_id = orderId
        self.connected_ok = True

    def _nid(self) -> int:
        self._next_id += 1
        return self._next_id

    def error(self, *args) -> None:
        code = next((a for a in args[1:] if isinstance(a, int)), None)
        req = args[0] if args and isinstance(args[0], int) else -1
        if code in (2104, 2106, 2158, 2176, 366):
            return
        if req > 0 and not self._done.get(req, False):
            self._errors[req] = f"code={code}"
            self._done[req] = True

    # ── callbacks ───────────────────────────────────────────────────────
    def contractDetails(self, reqId: int, cd) -> None:
        c = cd.contract
        self._details.append(IbExpiry(
            local_symbol=c.localSymbol,
            last_trade=c.lastTradeDateOrContractMonth,
            exchange=c.exchange,
            multiplier=str(c.multiplier),
            con_id=int(c.conId),
        ))

    def contractDetailsEnd(self, reqId: int) -> None:
        self._done[reqId] = True

    def historicalData(self, reqId: int, bar) -> None:
        self._bars.append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self._done[reqId] = True

    # ── public sync helpers ─────────────────────────────────────────────
    def connect_and_start(self, host: str, port: int, client_id: int, timeout: float = 8.0) -> bool:
        self.connect(host, port, client_id)
        threading.Thread(target=self.run, daemon=True).start()
        t0 = time.time()
        while not self.connected_ok and time.time() - t0 < timeout:
            time.sleep(0.1)
        return self.connected_ok

    def list_futures_expiries(self, symbol: str, exchange: str, *, timeout: float = 25.0) -> list[IbExpiry]:
        c = Contract()
        c.symbol = symbol
        c.secType = "FUT"
        c.exchange = exchange
        c.currency = "USD"
        c.includeExpired = True
        self._details = []
        rid = self._nid()
        self._done[rid] = False
        self.reqContractDetails(rid, c)
        self._wait(rid, timeout)
        return list(self._details)

    def fetch_contract_daily(
        self, expiry: IbExpiry, *, what_to_show: str = "TRADES",
        duration: str = "2 Y", timeout: float = 30.0,
    ) -> pd.DataFrame:
        c = Contract()
        c.secType = "FUT"
        c.exchange = expiry.exchange
        c.currency = "USD"
        c.lastTradeDateOrContractMonth = expiry.last_trade
        c.localSymbol = expiry.local_symbol
        c.includeExpired = True
        self._bars = []
        rid = self._nid()
        self._done[rid] = False
        # use_rth=0 (include all hours), format_date=1 (yyyymmdd string)
        self.reqHistoricalData(rid, c, "", duration, "1 day", what_to_show, 0, 1, False, [])
        self._wait(rid, timeout)
        if not self._bars:
            return pd.DataFrame()
        rows = [
            {"date": pd.to_datetime(b.date.split()[0]),
             "open": float(b.open), "high": float(b.high),
             "low": float(b.low), "close": float(b.close),
             "volume": int(float(b.volume)) if float(b.volume) >= 0 else 0}
            for b in self._bars
        ]
        df = pd.DataFrame(rows).set_index("date").sort_index()
        df.index = df.index.normalize()
        return df

    def _wait(self, rid: int, timeout: float) -> None:
        t0 = time.time()
        while not self._done.get(rid, False) and time.time() - t0 < timeout:
            time.sleep(0.1)


def ib_available(host: str = "127.0.0.1", port: int = 7497) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return _IBAPI_AVAILABLE
    except OSError:
        return False
