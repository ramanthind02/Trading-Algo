@echo off
REM ============================================================
REM Live MT5 <-> Nautilus adapter tests against the FTMO DEMO
REM
REM Runs tests\live_mt5 against the FTMO demo terminal. This binds ONLY to the
REM FTMO terminal (via mt5.initialize(path=...)) in its OWN process, so it runs
REM in PARALLEL with the Darwinex tick-scraper and never touches it.
REM
REM Safety (enforced in tests\live_mt5\conftest.py):
REM   * Hard guard: aborts unless the bound account is the FTMO demo login,
REM     on the FTMO server, with trade_mode == DEMO.
REM   * Order tiers tag everything with TEST_MAGIC (990510) and flatten on exit.
REM
REM Tiers:
REM   MT5_LIVE_TESTS=1   -> connectivity + instruments + data + node (NO orders)
REM   MT5_LIVE_ORDERS=1  -> ALSO the EURUSD round-trip + node-order tiers
REM                         (order tiers self-skip when the market is closed)
REM
REM Prerequisites:
REM   * FTMO MT5 terminal open, logged into the demo, Algo Trading ENABLED.
REM   * FTMO_DEMO_{SERVER,LOGIN,PASSWORD} in gitignored .env.
REM   * Optional: FTMO_DEMO_TERMINAL_PATH if the terminal is not at the default.
REM
REM Usage:
REM   deployment\ops\run_mt5_ftmo_tests.bat            (read-only tiers)
REM   deployment\ops\run_mt5_ftmo_tests.bat orders     (also place demo orders)
REM ============================================================

cd /d "C:\Users\raman\Documents\repos\Trading-Algo"
set "PYTHONPATH=C:\Users\raman\Documents\repos\Trading-Algo"

set "MT5_LIVE_TESTS=1"
if /I "%~1"=="orders" set "MT5_LIVE_ORDERS=1"

REM Override here if your FTMO terminal lives elsewhere:
REM set "FTMO_DEMO_TERMINAL_PATH=C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"

.\.venv\Scripts\python.exe -m pytest tests\live_mt5 -v -rs
