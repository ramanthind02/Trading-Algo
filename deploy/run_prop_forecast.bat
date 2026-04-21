@echo off
REM ============================================================
REM Daily Prop Firm Forecast (Enigma Algos)
REM Runs once, generates forecast in micro futures contracts,
REM sends to the prop-firm Telegram channel, exits.
REM
REM Schedule via Windows Task Scheduler at 6:00 PM ET daily
REM (3:00 PM PT / 5:00 PM CT).
REM
REM Rationale:
REM   - 4:45 PM ET: we (or Lucid) flatten positions ahead of CME close
REM   - 5:00 PM ET: daily candle closes
REM   - 6:00 PM ET: CME settlement window ends, new positions can be entered
REM Prerequisite: TWS or IB Gateway running with API enabled.
REM ============================================================

cd /d "C:\Users\adabla\Trading-Algo"
call "cpython_env\Scripts\activate.bat"
set "PYTHONPATH=C:\Users\adabla\Trading-Algo"

REM Paper port 7497 by default; change to 7496 for live IB account.
python scripts\enigma_prop_forecast.py --port 7497

echo [%date% %time%] Prop forecast run completed >> deploy\forecast.log
