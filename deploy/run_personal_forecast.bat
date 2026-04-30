@echo off
REM ============================================================
REM Daily Personal Account Forecast (Enigma Algos)
REM Runs once, generates forecast in ETF fractional shares,
REM sends to the personal Telegram channel, exits.
REM
REM Schedule via Windows Task Scheduler at 3:45 PM ET daily
REM (12:45 PM PT / 2:45 PM CT) -- 15 minutes before US equity close
REM at 4:00 PM ET so orders can still be placed with fractional shares.
REM
REM Rationale:
REM   - IB disables fractional share entry outside regular trading hours.
REM   - Daily candle for today isn't closed yet at this time, so the script
REM     synthesizes a partial daily candle from 15-min intraday bars for
REM     today before running the forecast.
REM
REM Prerequisite: TWS or IB Gateway running with API enabled.
REM ============================================================

cd /d "C:\Users\adabla\Trading-Algo"
call "cpython_env\Scripts\activate.bat"
set "PYTHONPATH=C:\Users\adabla\Trading-Algo"

REM Paper port 7497 by default; change to 7496 for live IB account.
echo. >> deploy\forecast.log
echo [%date% %time%] === Personal forecast run BEGIN === >> deploy\forecast.log
python scripts\enigma_personal_forecast.py --port 7497 >> deploy\forecast.log 2>&1
echo [%date% %time%] === Personal forecast run END (exit %errorlevel%) === >> deploy\forecast.log
