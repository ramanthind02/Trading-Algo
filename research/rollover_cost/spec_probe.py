"""Probe MT5 symbol specs for the rollover-cost experiment.

Pulls swap rates, spread, tick size, contract size and trade sessions for the
portfolio CFDs and their ETF equivalents. Read-only — never places an order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()
import MetaTrader5 as mt5
from data_platform.providers.mt5.scraper import connect

# Darwinex CFDs (ES/NQ/GC/SI) + ETF equivalents + a couple FX for reference
SYMBOLS = ["SP500", "NDX", "XAUUSD", "XAGUSD",
           "SPY", "QQQ", "GLD", "SLV",
           "EURUSD", "USDJPY"]

_SWAP_MODE = {
    0: "DISABLED", 1: "POINTS", 2: "SYMBOL_CCY", 3: "MARGIN_CCY",
    4: "DEPOSIT_CCY", 5: "INTEREST_CURRENT", 6: "INTEREST_OPEN",
    7: "REOPEN_CURRENT", 8: "REOPEN_BID",
}


def main() -> None:
    if not connect():
        print("MT5 CONNECT FAILED"); sys.exit(1)
    acct = mt5.account_info()
    print(f"Connected: login={getattr(acct,'login','?')} server={getattr(acct,'server','?')}\n")

    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            print(f"{sym:8s}  symbol_select FAILED"); continue
        i = mt5.symbol_info(sym)
        if i is None:
            print(f"{sym:8s}  symbol_info None"); continue
        tick = mt5.symbol_info_tick(sym)
        spread_pts = i.spread
        spread_price = (i.ask - i.bid) if (tick and i.ask and i.bid) else float("nan")
        # spread as fraction of price (basis points)
        mid = (i.ask + i.bid) / 2 if (i.ask and i.bid) else i.last
        spread_bps = (spread_price / mid * 1e4) if mid else float("nan")
        print(f"=== {sym} ({i.path}) ===")
        print(f"  bid/ask        : {i.bid} / {i.ask}   spread={spread_price:.5g} ({spread_pts} pts, {spread_bps:.2f} bps)")
        print(f"  point/digits   : point={i.point} digits={i.digits} tick_size={i.trade_tick_size} tick_value={i.trade_tick_value}")
        print(f"  contract_size  : {i.trade_contract_size}  vol_min={i.volume_min} vol_step={i.volume_step}")
        print(f"  swap_mode      : {_SWAP_MODE.get(i.swap_mode, i.swap_mode)}  long={i.swap_long} short={i.swap_short} rollover3days={i.swap_rollover3days}")
        print(f"  currency       : base={i.currency_base} profit={i.currency_profit} margin={i.currency_margin}")
        # Daily swap as a dollar figure and bps of notional (POINTS mode)
        notional = i.trade_contract_size * mid if mid else float("nan")
        swap_long_usd  = i.swap_long  * i.point * i.trade_contract_size
        swap_short_usd = i.swap_short * i.point * i.trade_contract_size
        long_bps  = swap_long_usd  / notional * 1e4 if notional else float("nan")
        short_bps = swap_short_usd / notional * 1e4 if notional else float("nan")
        print(f"  notional/lot   : ${notional:,.0f}")
        print(f"  swap_long_usd  : ${swap_long_usd:.3f}/day ({long_bps:.3f} bps/day)   swap_short_usd: ${swap_short_usd:.3f}/day ({short_bps:.3f} bps/day)")
        print(f"  spread vs swap : spread={spread_bps:.3f} bps  |  1-day long swap={abs(long_bps):.3f} bps  short swap={abs(short_bps):.3f} bps")
        print(f"  filling_mode   : {i.filling_mode}  trade_mode={i.trade_mode}  exemode={i.trade_exemode}")
        print()

    mt5.shutdown()


if __name__ == "__main__":
    main()
