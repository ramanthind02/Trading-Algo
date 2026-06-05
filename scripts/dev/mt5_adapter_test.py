r"""End-to-end test of the vendored Nautilus MT5 adapter against a prop-firm DEMO.

Broker-agnostic: reads <BROKER>_DEMO_{SERVER,LOGIN,PASSWORD} from .env and binds
to a SPECIFIC terminal install via --path (mt5.initialize(path=...)). This is the
piece that makes multi-terminal machines deterministic: the adapter's MT5Connection
calls bare mt5.initialize(), so we pre-initialize the correct terminal first; the
subsequent bare attach then binds to it instead of whatever terminal is registered.

Drives the REAL adapter classes (MT5Connection, MT5InstrumentProvider) and,
optionally, opens + closes ONE broker-minimum order using the EXACT mt5.order_send
request shape the adapter's MT5LiveExecutionClient builds.

================================================================================
SAFETY
================================================================================
* Creds from .env (gitignored), never printed.
* HARD DEMO GUARD: asserts account_info().trade_mode == ACCOUNT_TRADE_MODE_DEMO
  before ANY order; aborts otherwise. Cannot trade a live account.
* Order leg is opt-out via --no-order; it opens broker-min volume and closes it.

Usage
-----
    .\.venv\Scripts\python.exe scripts\dev\mt5_adapter_test.py ^
        --broker FUNDEDNEXT --path "C:\Program Files\FundedNext MetaTrader 5\terminal64.exe"
    ... --no-order                 # connectivity only
    ... --order-symbol EURUSD      # force the test-order symbol
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _ROOT / "deployment" / "nautilus_mt5" / "vendor" / "mt5-connect"
if str(_ADAPTER) not in sys.path:
    sys.path.insert(0, str(_ADAPTER))

for _line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import MetaTrader5 as mt5  # noqa: E402

from mt5connect.config import MT5Config  # noqa: E402
from mt5connect.connection import MT5Connection  # noqa: E402
from mt5connect.constants import FILLING_MODE, MT5_MAGIC_NUMBER  # noqa: E402
from mt5connect.providers import MT5InstrumentProvider  # noqa: E402


def _hr(title: str) -> None:
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}")


def _ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f"  -> {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))


def _retcode(code: int) -> str:
    table = {
        10004: "Requote", 10006: "Rejected", 10007: "Cancelled by trader",
        10008: "Order placed", 10009: "Request completed", 10010: "Partial fill",
        10013: "Invalid request", 10014: "Invalid volume", 10015: "Invalid price",
        10016: "Invalid stops", 10017: "Trade disabled", 10018: "Market closed",
        10019: "Insufficient funds", 10020: "Prices changed", 10021: "No quotes",
        10026: "Autotrading disabled by server", 10027: "Autotrading disabled by client",
        10030: "Invalid filling mode", 10031: "No connection to trade server",
    }
    return table.get(code, f"retcode {code}")


def _supported_filling(symbol: str):
    info = mt5.symbol_info(symbol)
    mask = getattr(info, "filling_mode", 0) if info else 0
    if mask & 2:
        return mt5.ORDER_FILLING_IOC
    if mask & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def discover_symbols(names: list[str]) -> dict[str, list[str]]:
    probes = {
        "NDX/NASDAQ (NQ)": [r"NDX", r"NAS100", r"US100", r"USTEC", r"USATECH", r"NASDAQ"],
        "S&P500 (ES)": [r"SP500", r"US500", r"SPX", r"USA500"],
        "DOW (YM)": [r"US30", r"WS30", r"DJ30", r"DOW"],
        "DAX": [r"GER40", r"DE40", r"GDAXI", r"^DAX"],
        "GOLD (GC)": [r"XAUUSD", r"GOLD"],
        "SILVER (SI)": [r"XAGUSD", r"SILVER"],
        "OIL (CL)": [r"XTIUSD", r"WTI", r"USOIL", r"USOUSD"],
        "EURUSD": [r"^EURUSD"],
    }
    return {label: [n for n in names if any(re.search(p, n, re.I) for p in pats)]
            for label, pats in probes.items()}


def pre_initialize(path: str | None, account: int, server: str, password: str) -> bool:
    """Bind the MT5 Python bridge to a SPECIFIC terminal install + log in.

    With path: launches/attaches THAT terminal and logs into the demo. Without
    path: attaches to the running terminal (only correct if it's already the
    right broker). Returns True on success.
    """
    kwargs = dict(login=account, server=server, password=password, timeout=20000)
    if path:
        kwargs["path"] = path
    ok = mt5.initialize(**kwargs)
    if not ok:
        code, msg = mt5.last_error()
        _fail("mt5.initialize(path/login)", f"error {code}: {msg}")
        if code == -10005:
            print("    -10005 IPC timeout: terminal can't reach this broker's server. "
                  "Is --path the prop-firm terminal (not Darwinex), open + logged in?")
        return False
    ti = mt5.terminal_info()
    if ti:
        _ok("mt5.initialize()", f"terminal={ti.name} company={ti.company} connected={ti.connected}")
    return True


def run(broker: str, path: str | None, place_order: bool, order_symbol: str | None) -> int:
    prefix = broker.upper()
    try:
        account = int(os.environ[f"{prefix}_DEMO_LOGIN"])
        server = os.environ[f"{prefix}_DEMO_SERVER"]
        password = os.environ[f"{prefix}_DEMO_PASSWORD"]
    except KeyError as exc:
        _fail("creds", f"missing {exc} in .env (need {prefix}_DEMO_SERVER/LOGIN/PASSWORD)")
        return 1

    _hr(f"Bind terminal ({prefix})")
    print(f"  login={account} server={server} path={path or '(running terminal)'}")
    if not pre_initialize(path, account, server, password):
        return 1

    # ── Layer 1-3: adapter connection lifecycle + account snapshot ────────────
    _hr("Layer 1-3  MT5Connection lifecycle + account snapshot")
    creds = MT5Config(account=account, password=password, server=server, symbols=["EURUSD"])
    conn = MT5Connection(creds)
    try:
        conn.connect()  # bare initialize() attaches to the terminal we bound above
    except Exception as exc:  # noqa: BLE001
        _fail("MT5Connection.connect()", f"{type(exc).__name__}: {exc}")
        mt5.shutdown()
        return 1
    snap = conn.get_account_info()
    _ok("MT5Connection.connect()", repr(conn))
    _ok("get_account_info()",
        f"#{snap.login} {snap.server} bal={snap.balance:.2f} {snap.currency} "
        f"lev=1:{snap.leverage} company={snap.company!r}")

    # ── HARD DEMO GUARD ───────────────────────────────────────────────────────
    raw = mt5.account_info()
    trade_mode = int(raw.trade_mode)
    mode_name = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(trade_mode, str(trade_mode))
    is_demo = trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    print(f"\n  account trade_mode = {trade_mode} ({mode_name}) | demo={is_demo}")
    if place_order and not is_demo:
        _fail("DEMO GUARD", f"account is {mode_name}, NOT demo — refusing to place any order.")
        conn.disconnect()
        return 2

    # ── Symbol discovery ──────────────────────────────────────────────────────
    _hr("Symbol discovery (broker native names)")
    names = [s.name for s in (mt5.symbols_get() or [])]
    print(f"  total symbols: {len(names)}")
    disco = discover_symbols(names)
    for label, found in disco.items():
        print(f"    {label:>16}: {found[:8]}")

    # ── Layer 4-6: instrument parse + tick + bars via adapter provider ────────
    ndx, eur = disco["NDX/NASDAQ (NQ)"], disco["EURUSD"]
    test_syms = [s for s in (ndx[0] if ndx else None, eur[0] if eur else None) if s] or names[:1]

    _hr("Layer 4-6  Instrument parsing + live tick + historical bars")
    provider = MT5InstrumentProvider(conn)
    for sym in test_syms:
        try:
            inst = provider.load_symbol(sym)
            _ok(f"load_symbol({sym!r}) -> Nautilus {type(inst).__name__}",
                f"id={inst.id} price_precision={inst.price_precision} size_precision={inst.size_precision}")
        except Exception as exc:  # noqa: BLE001
            _fail(f"load_symbol({sym!r})", f"{type(exc).__name__}: {exc}")
        tick = mt5.symbol_info_tick(sym)
        si = mt5.symbol_info(sym)
        if tick and tick.bid:
            spread_pts = round((tick.ask - tick.bid) / (si.point or 1e-9))
            _ok(f"symbol_info_tick({sym!r})",
                f"bid={tick.bid} ask={tick.ask} spread~{spread_pts}pts "
                f"vol_min={si.volume_min} contract={si.trade_contract_size}")
        else:
            _fail(f"symbol_info_tick({sym!r})", "None / no bid (market closed?)")
        end = dt.datetime.now(dt.timezone.utc)
        bars = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, end - dt.timedelta(days=7), end)
        if bars is None or len(bars) == 0:
            bars = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, end - dt.timedelta(days=30), end)
        if bars is not None and len(bars):
            _ok(f"copy_rates_range({sym!r}, H1)", f"{len(bars)} bars, last close={bars[-1]['close']}")
        else:
            _fail(f"copy_rates_range({sym!r}, H1)", "no bars in 30d (weekend/holiday?)")

    # ── Layer 7-8: construct the real data + exec clients (no node) ───────────
    _hr("Layer 7-8  Client construction (MT5DataClient, MT5LiveExecutionClient)")
    try:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from nautilus_trader.common.component import LiveClock
        from nautilus_trader.common.providers import InstrumentProvider
        from nautilus_trader.test_kit.stubs.component import TestComponentStubs
        from mt5connect.data import MT5DataClient
        from mt5connect.execution import MT5LiveExecutionClient

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        conn_mock = MagicMock(spec=MT5Connection)
        conn_mock.ensure_connected = MagicMock()
        prov = MT5InstrumentProvider.__new__(MT5InstrumentProvider)
        InstrumentProvider.__init__(prov)
        prov._conn = conn_mock
        prov._failed_symbols = []
        prov.get_instrument = MagicMock(return_value=None)
        prov.load_all_async = AsyncMock()
        dc = MT5DataClient(loop=loop, connection=conn_mock, msgbus=TestComponentStubs.msgbus(),
                           cache=TestComponentStubs.cache(), clock=LiveClock(),
                           instrument_provider=prov, config=creds)
        _ok("MT5DataClient.__init__()", f"client_id={dc.id}")
        ec = MT5LiveExecutionClient(loop=loop, connection=conn_mock, msgbus=TestComponentStubs.msgbus(),
                                    cache=TestComponentStubs.cache(), clock=LiveClock(),
                                    instrument_provider=prov, config=creds)
        _ok("MT5LiveExecutionClient.__init__()", f"account_id={ec.account_id}")
    except Exception as exc:  # noqa: BLE001
        _fail("client construction", f"{type(exc).__name__}: {exc}")

    # ── Layer 9: guarded round-trip order (adapter request shape) ─────────────
    if place_order:
        _hr("Layer 9  Guarded round-trip ORDER (DEMO only)")
        sym = order_symbol or (eur[0] if eur else (ndx[0] if ndx else test_syms[0]))
        si = mt5.symbol_info(sym)
        if si is None:
            _fail("order pre-check", f"symbol_info({sym!r}) is None")
        else:
            mt5.symbol_select(sym, True)
            vol = float(si.volume_min)
            tick = mt5.symbol_info_tick(sym)
            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": vol,
                "type": mt5.ORDER_TYPE_BUY, "price": tick.ask, "sl": 0.0, "tp": 0.0,
                "deviation": 20, "magic": MT5_MAGIC_NUMBER, "comment": "adapter-smoke",
                "type_filling": FILLING_MODE, "type_time": mt5.ORDER_TIME_GTC,
            }
            print(f"  order_check: {sym} BUY {vol} @ ask={tick.ask} magic={MT5_MAGIC_NUMBER} filling=IOC")
            chk = mt5.order_check(req)
            if chk is not None:
                print(f"    order_check retcode={chk.retcode} ({_retcode(chk.retcode)}) "
                      f"margin={getattr(chk, 'margin', '?')}")
            res = mt5.order_send(req)
            if res is not None and res.retcode == 10030:
                _fail("order_send (IOC)", "10030 invalid filling — retry broker-supported mode")
                req["type_filling"] = _supported_filling(sym)
                res = mt5.order_send(req)
            if res is None:
                code, msg = mt5.last_error()
                _fail("order_send (open)", f"None — error {code}: {msg}")
            elif res.retcode in (mt5.TRADE_RETCODE_DONE, 10009, 10008):
                # NOTE: under MT5 "market execution" the order_send result returns
                # deal/price as 0 even on success; the real fill lands in history a
                # beat later. We treat history_deals_get as authoritative below.
                _ok("order_send (open)",
                    f"order={res.order} retcode={res.retcode} ({_retcode(res.retcode)}) "
                    "(result.deal/price are 0 under market-exec; see deal history)")
                time.sleep(1.5)  # let the position post to the positions table
                mine = [p for p in (mt5.positions_get(symbol=sym) or []) if p.magic == MT5_MAGIC_NUMBER]
                _ok("positions_get()", f"{len(mine)} open position(s) magic={MT5_MAGIC_NUMBER}")
                for p in mine:
                    ctick = mt5.symbol_info_tick(sym)
                    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    close_px = ctick.bid if p.type == mt5.ORDER_TYPE_BUY else ctick.ask
                    creq = {
                        "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": p.volume,
                        "type": close_type, "position": p.ticket, "price": close_px,
                        "deviation": 20, "magic": MT5_MAGIC_NUMBER, "comment": "adapter-smoke-close",
                        "type_filling": req["type_filling"],
                    }
                    cres = mt5.order_send(creq)
                    if cres is not None and cres.retcode in (mt5.TRADE_RETCODE_DONE, 10009):
                        _ok("order_send (close)", f"position {p.ticket} close accepted (retcode={cres.retcode})")
                    else:
                        code = cres.retcode if cres else -1
                        _fail("order_send (close)", f"{_retcode(code)} ({code}) — CLOSE MANUALLY")
                # Flat check with settle delay + brief retry (positions table lags fills).
                left = mine
                for _ in range(5):
                    time.sleep(1.0)
                    left = [p for p in (mt5.positions_get(symbol=sym) or []) if p.magic == MT5_MAGIC_NUMBER]
                    if not left:
                        break
                (_ok if not left else _fail)(
                    "flat check", f"{len(left)} residual position(s) with our magic"
                    + ("" if not left else " — RUN scripts/dev cleanup / close manually"))
                # Authoritative round-trip fills from deal history.
                now = dt.datetime.now(dt.timezone.utc)
                ours = [d for d in (mt5.history_deals_get(now - dt.timedelta(hours=2), now) or [])
                        if d.magic == MT5_MAGIC_NUMBER and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]
                print("  round-trip deals (authoritative):")
                for d in ours[-4:]:
                    side = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
                    inout = {0: "IN", 1: "OUT", 2: "INOUT"}.get(getattr(d, "entry", -1), "?")
                    print(f"    deal {d.ticket}: {side} {inout} {d.volume} {d.symbol} @ {d.price} "
                          f"commission={d.commission} swap={d.swap} profit={d.profit}")
            else:
                _fail("order_send (open)", f"{_retcode(res.retcode)} ({res.retcode}) — no position opened")

    _hr("Disconnect")
    conn.disconnect()
    _ok("MT5Connection.disconnect()", f"terminal now on {server}")
    print("\n  >>> If you switched away from a live account, re-login it manually. <<<")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Nautilus MT5 adapter test (prop-firm demo)")
    ap.add_argument("--broker", default="FUNDEDNEXT",
                    help="env prefix for <BROKER>_DEMO_* creds (FTMO, FUNDEDNEXT, ...)")
    ap.add_argument("--path", default=None,
                    help="terminal64.exe of THAT broker's MT5 install (selects the terminal)")
    ap.add_argument("--no-order", action="store_true", help="connectivity only; no demo order")
    ap.add_argument("--order-symbol", default=None, help="force test-order symbol")
    args = ap.parse_args()
    return run(args.broker, args.path, not args.no_order, args.order_symbol)


if __name__ == "__main__":
    sys.exit(main())
