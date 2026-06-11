"""Probe FTMO-Demo symbol specs (spread + swap) and compare to Darwinex.

Binds explicitly to the FTMO terminal via mt5.initialize(path=..., login/password/server)
so it never touches the Darwinex scraper terminal. Read-only — never places an order.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import MetaTrader5 as mt5
from research.rollover_cost.config import SPECS  # Darwinex specs for comparison

_SWAP_MODE = {0: "DISABLED", 1: "POINTS", 2: "SYMBOL_CCY", 3: "MARGIN_CCY",
              4: "DEPOSIT_CCY", 5: "INTEREST_CUR", 6: "INTEREST_OPEN",
              7: "REOPEN_CUR", 8: "REOPEN_BID"}

# FTMO native symbols (configs/mt5_brokers.yaml: brokers.ftmo) for ES/NQ/GC/SI/CL.
FTMO_SYMS = {"ES": "US500.cash", "NQ": "US100.cash", "GC": "XAUUSD",
             "SI": "XAGUSD", "CL": "USOIL.cash"}
DARWINEX = {"ES": "SP500", "NQ": "NDX", "GC": "XAUUSD", "SI": "XAGUSD", "CL": "XTIUSD"}

FTMO_PATH = os.environ.get("FTMO_DEMO_TERMINAL_PATH",
                           r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")


def connect_ftmo() -> bool:
    ok = mt5.initialize(
        path=FTMO_PATH,
        login=int(os.environ["FTMO_DEMO_LOGIN"]),
        password=os.environ["FTMO_DEMO_PASSWORD"],
        server=os.environ["FTMO_DEMO_SERVER"],
    )
    if not ok:
        print("FTMO initialize failed:", mt5.last_error()); return False
    acct = mt5.account_info()
    print(f"Connected: login={getattr(acct,'login','?')} server={getattr(acct,'server','?')} "
          f"company={getattr(acct,'company','?')}\n")
    return True


def main() -> None:
    if not connect_ftmo():
        sys.exit(1)
    print(f"{'canon':5s} {'FTMO sym':12s} {'mode':>8s} {'spread_bps':>10s} "
          f"{'swapL_bps':>10s} {'swapS_bps':>10s}  (ref price used for bps)")
    ftmo_bps = {}
    for canon, sym in FTMO_SYMS.items():
        if not mt5.symbol_select(sym, True):
            print(f"{canon:5s} {sym:12s} symbol_select FAILED"); continue
        i = mt5.symbol_info(sym)
        if i is None:
            print(f"{canon:5s} {sym:12s} no info"); continue
        live_mid = (i.ask + i.bid) / 2 if (i.ask and i.bid) else 0.0
        ref = live_mid or _REF.get(DARWINEX[canon], 0.0) or i.last
        spread_bps = (i.ask - i.bid) / live_mid * 1e4 if live_mid else float("nan")
        long_bps = i.swap_long * i.point / ref * 1e4 if ref else float("nan")
        short_bps = i.swap_short * i.point / ref * 1e4 if ref else float("nan")
        ftmo_bps[canon] = (long_bps, short_bps, spread_bps)
        tag = "live" if live_mid else "ref(closed)"
        print(f"{canon:5s} {sym:12s} {_SWAP_MODE.get(i.swap_mode,i.swap_mode):>8s} "
              f"{spread_bps:10.2f} {long_bps:10.3f} {short_bps:10.3f}   {ref:.2f} [{tag}]")

    print("\n=== SWAP comparison: FTMO vs Darwinex (long swap, bps/night, at same ref price) ===")
    print(f"{'canon':5s} {'FTMO_long_bps':>13s} {'DWX_long_bps':>13s} {'FTMO/DWX':>9s}")
    for canon, sym in DARWINEX.items():
        if sym in SPECS and canon in ftmo_bps:
            ref = _REF.get(sym, 1.0)
            dwx = SPECS[sym].swap_long_pts * SPECS[sym].point / ref * 1e4
            ftmo = ftmo_bps[canon][0]
            ratio = ftmo / dwx if dwx else float("nan")
            print(f"{canon:5s} {ftmo:13.3f} {dwx:13.3f} {ratio:8.2f}x")

    print("\n* swap bps computed at a common reference price; spreads are single snapshots "
          "(markets may be closed off-hours). Swap rates are static and directly comparable.")
    mt5.shutdown()


# rough current reference prices for the DWX bps guide (only for the comparison print)
_REF = {"SP500": 7369, "NDX": 28840, "XAUUSD": 4328, "XAGUSD": 67.9, "XTIUSD": 88.5}


if __name__ == "__main__":
    main()
