"""FTMO-Demo MT5 connection + symbol-discovery probe (READ-ONLY; never places orders).

Reads FTMO_DEMO_{SERVER,LOGIN,PASSWORD} from .env. Connecting with a login SWITCHES the
running MT5 terminal to FTMO-Demo (logs out the currently logged-in account). This script
NEVER calls order_send / any trade function — it only reads account info, the symbol
universe, and recent bars to confirm the connection and discover this broker's symbol names.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

# --- load .env (no external dep) ---
_ROOT = Path(__file__).resolve().parents[2]
for _line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import MetaTrader5 as mt5  # noqa: E402

server = os.environ["FTMO_DEMO_SERVER"]
login = int(os.environ["FTMO_DEMO_LOGIN"])
password = os.environ["FTMO_DEMO_PASSWORD"]

ok = mt5.initialize(login=login, server=server, password=password)
if not ok:
    print("initialize FAILED:", mt5.last_error())
    sys.exit(1)

ai = mt5.account_info()
print(f"CONNECTED: login={ai.login} server={ai.server} trade_mode={ai.trade_mode} "
      f"(0=DEMO,1=CONTEST,2=REAL) company={ai.company!r} balance={ai.balance} {ai.currency}")

syms = mt5.symbols_get() or []
names = [s.name for s in syms]
print(f"total symbols: {len(names)}")


def find(pats: list[str]) -> list[str]:
    return [n for n in names if any(re.search(p, n, re.I) for p in pats)]


probes = {
    "NDX/NASDAQ (NQ)": [r"NDX", r"NAS100", r"US100", r"USTEC", r"USATECH", r"NASDAQ"],
    "S&P500 (ES)": [r"SP500", r"US500", r"SPX", r"USA500"],
    "DOW (YM)": [r"US30", r"WS30", r"DJ30", r"^DOW"],
    "DAX": [r"GER40", r"DE40", r"GDAXI", r"DAX"],
    "GOLD (GC)": [r"XAUUSD", r"GOLD"],
    "SILVER (SI)": [r"XAGUSD", r"SILVER"],
    "OIL (CL)": [r"XTIUSD", r"WTI", r"USOIL"],
}
for label, pats in probes.items():
    print(f"  {label:>16}: {find(pats)}")

# data probe on the first NDX candidate (read-only)
ndx_cands = find([r"NDX", r"NAS100", r"US100", r"USTEC", r"USATECH"])
ndx = ndx_cands[0] if ndx_cands else None
if ndx:
    mt5.symbol_select(ndx, True)
    info = mt5.symbol_info(ndx)
    if info:
        print(f"  '{ndx}': digits={info.digits} point={info.point} "
              f"contract_size={info.trade_contract_size} path={info.path}")
    rates = mt5.copy_rates_from_pos(ndx, mt5.TIMEFRAME_D1, 0, 5)
    n = 0 if rates is None else len(rates)
    print(f"  '{ndx}' recent D1 bars fetched: {n}")
    if rates is not None:
        for r in rates[-3:]:
            print("    ", dt.datetime.utcfromtimestamp(int(r["time"])).date(),
                  "O", r["open"], "C", r["close"])

mt5.shutdown()
print("DONE — no orders placed. Terminal is now on FTMO-Demo; re-login Darwinex manually if needed.")
