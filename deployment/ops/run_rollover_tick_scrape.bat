@echo off
REM ============================================================
REM Daily rollover-window tick scrape
REM
REM Captures two windows around the 17:00-18:00 NY rollover dead zone:
REM   EXIT  window: 16:00-16:59 NY  (last hour before rollover)
REM   ENTRY window: 18:00-19:00 NY  (first hour after session reopens)
REM
REM Ticks stored to:
REM   data\mt5_data\<SYMBOL>\ticks_rollover_exit\year=YYYY\part.parquet
REM   data\mt5_data\<SYMBOL>\ticks_rollover_entry\year=YYYY\part.parquet
REM
REM Scheduled at 16:05 PT (19:05 ET) — both windows have fully closed.
REM 4 parallel workers; all 844 symbols typically complete in ~15-20 min.
REM
REM Prerequisite: Darwinex MT5 terminal running and logged in.
REM ============================================================

REM Repo root derived from this bat's location (deployment\ops\) — never hardcode a
REM per-machine path (a stale hardcoded path silently breaks the task with no error).
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%"

echo. >> logs\rollover_tick_scrape.log
echo [%date% %time%] === Rollover tick scrape BEGIN === >> logs\rollover_tick_scrape.log
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.rollover_tick_scraper --workers 4 >> logs\rollover_tick_scrape.log 2>&1
echo [%date% %time%] === Rollover tick scrape END (exit %errorlevel%) === >> logs\rollover_tick_scrape.log
