@echo off
REM ============================================================
REM Daily MT5 data scrape — incremental M1 bars for all symbols
REM
REM Runs once after market close, appends new candles to
REM data\mt5_data\<SYMBOL>\bars_M1\year=YYYY\part.parquet
REM
REM Schedule via Windows Task Scheduler (setup_scheduled_task.ps1)
REM at ~5:00 PM ET (after US equity close + data settle time).
REM
REM To also pull raw tick data pass --ticks (large files, slow):
REM   python -m data_platform.providers.mt5.scraper --ticks
REM
REM Prerequisite: FTMO MT5 terminal running on this machine.
REM ============================================================

cd /d "C:\Users\raman\Documents\repos\Trading-Algo"
set "PYTHONPATH=C:\Users\raman\Documents\repos\Trading-Algo"

echo. >> deploy\mt5_scrape.log
echo [%date% %time%] === MT5 scrape BEGIN === >> deploy\mt5_scrape.log
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper >> deploy\mt5_scrape.log 2>&1
echo [%date% %time%] === MT5 scrape END (exit %errorlevel%) === >> deploy\mt5_scrape.log
