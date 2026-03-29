@echo off
REM ============================================================
REM Daily TWS Live Forecast
REM Runs once, generates forecast, sends to Telegram, exits.
REM Schedule via Windows Task Scheduler at 5:30 PM ET daily.
REM ============================================================

cd /d "C:\Users\adabla\Trading-Algo"
call "cpython_env\Scripts\activate.bat"

REM Use paper trading port (7497). Change to 7496 for live.
python scripts\tws_live_forecast.py --port 7497

REM Log completion
echo [%date% %time%] Forecast run completed >> deploy\forecast.log
