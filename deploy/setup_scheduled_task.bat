@echo off
REM ============================================================
REM Creates a Windows Task Scheduler task for daily forecasts.
REM Run this ONCE as Administrator to set up the schedule.
REM
REM Schedule: Daily at 5:30 PM ET (17:30 Eastern)
REM Prerequisites: TWS or IB Gateway must be running at that time.
REM ============================================================

schtasks /create ^
  /tn "TradingAlgo\DailyForecast" ^
  /tr "C:\Users\adabla\Trading-Algo\deploy\run_daily_forecast.bat" ^
  /sc daily ^
  /st 17:30 ^
  /f ^
  /rl HIGHEST

if %errorlevel% equ 0 (
    echo.
    echo Task created successfully!
    echo   Name: TradingAlgo\DailyForecast
    echo   Schedule: Daily at 5:30 PM
    echo   Script: C:\Users\adabla\Trading-Algo\deploy\run_daily_forecast.bat
    echo.
    echo To verify: schtasks /query /tn "TradingAlgo\DailyForecast"
    echo To delete:  schtasks /delete /tn "TradingAlgo\DailyForecast" /f
    echo To run now: schtasks /run /tn "TradingAlgo\DailyForecast"
) else (
    echo.
    echo ERROR: Failed to create task. Run this script as Administrator.
)
pause
