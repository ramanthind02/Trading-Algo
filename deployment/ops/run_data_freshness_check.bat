@echo off
REM ============================================================
REM Independent daily tripwire: alert if the MT5 signal feed is stale.
REM
REM This fires EVEN IF the MT5DataScrape task never ran (a broken/removed
REM schedule, a downed terminal). It reads only the stored parquet (no MT5
REM connection, so it never contends with a live trading node) and exits
REM non-zero + writes logs\data_freshness.json when the feed is behind.
REM
REM Schedule ~45 min after the scrape (setup_scheduled_task.ps1).
REM ============================================================
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%"

echo. >> logs\data_freshness.log
echo [%date% %time%] === freshness check BEGIN === >> logs\data_freshness.log
.\.venv\Scripts\python.exe -m deployment.ops.check_data_freshness >> logs\data_freshness.log 2>&1
echo [%date% %time%] === freshness check END (exit %errorlevel%) === >> logs\data_freshness.log
