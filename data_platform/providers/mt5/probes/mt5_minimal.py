"""Minimal MT5 test - no bootstrap, just bare connection."""
import sys
sys.path.insert(0, '.')
import MetaTrader5 as mt5
from datetime import datetime, timezone

ok = mt5.initialize()
print('init:', ok, mt5.last_error())
acct = mt5.account_info()
print('account:', acct.login if acct else None)

rates = mt5.copy_rates_from('EURUSD', mt5.TIMEFRAME_D1, datetime(2000,1,1,tzinfo=timezone.utc), 500)
print('EURUSD D1:', len(rates) if rates is not None else None, mt5.last_error())

rates_m1 = mt5.copy_rates_from('EURUSD', mt5.TIMEFRAME_M1, datetime(2026,6,1,tzinfo=timezone.utc), 10000)
print('EURUSD M1 (Jun 2026):', len(rates_m1) if rates_m1 is not None else None, mt5.last_error())

ticks = mt5.copy_ticks_from('EURUSD', datetime(2026,6,1,tzinfo=timezone.utc), 100000, mt5.COPY_TICKS_ALL)
print('EURUSD ticks (Jun 2026):', len(ticks) if ticks is not None else None, mt5.last_error())

mt5.shutdown()
print('done')
