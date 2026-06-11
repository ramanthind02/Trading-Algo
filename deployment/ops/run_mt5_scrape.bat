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

REM Repo root is this .bat's dir (deployment\ops\) up two levels — never hardcode a
REM per-machine path (a stale 'C:\Users\adabla\...' path is what silently broke this).
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%"

echo. >> logs\mt5_scrape.log
echo [%date% %time%] === MT5 scrape BEGIN === >> logs\mt5_scrape.log
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper >> logs\mt5_scrape.log 2>&1
echo [%date% %time%] === MT5 scrape END (exit %errorlevel%) === >> logs\mt5_scrape.log

REM Post-scrape tripwire: confirm the scrape actually ADVANCED the feed; alert if not.
echo [%date% %time%] === freshness guard === >> logs\mt5_scrape.log
.\.venv\Scripts\python.exe -m deployment.ops.check_data_freshness >> logs\mt5_scrape.log 2>&1
echo [%date% %time%] === freshness guard exit %errorlevel% === >> logs\mt5_scrape.log
