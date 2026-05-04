@echo off
REM ============================================================
REM Creates Windows Task Scheduler tasks for the daily forecasts.
REM Run this ONCE as Administrator to set up the schedule.
REM
REM TIMES (Eastern Time):
REM   PropForecast     -- 6:00 PM ET (after CME settlement window)
REM   PersonalForecast -- 3:45 PM ET (15 min before US equity close)
REM
REM Set /st in YOUR LOCAL TIME. Default below assumes Pacific Time:
REM   3:00 PM PT  = 6:00 PM ET  (Prop)
REM   12:45 PM PT = 3:45 PM ET  (Personal)
REM If you're in Central/Eastern, edit the /st values accordingly.
REM
REM Prerequisites: TWS or IB Gateway must be running at those times.
REM ============================================================

echo Creating PropForecast task (6:00 PM ET / 3:00 PM PT daily)...
schtasks /create ^
  /tn "TradingAlgo\PropForecast" ^
  /tr "C:\Users\adabla\Trading-Algo\deploy\run_prop_forecast.bat" ^
  /sc daily ^
  /st 15:00 ^
  /f ^
  /rl HIGHEST

if %errorlevel% neq 0 (
    echo ERROR: Failed to create PropForecast task. Run this script as Administrator.
    goto :end
)

echo.
echo Creating PersonalForecast task (3:45 PM ET / 12:45 PM PT daily)...
schtasks /create ^
  /tn "TradingAlgo\PersonalForecast" ^
  /tr "C:\Users\adabla\Trading-Algo\deploy\run_personal_forecast.bat" ^
  /sc daily ^
  /st 12:45 ^
  /f ^
  /rl HIGHEST

if %errorlevel% neq 0 (
    echo ERROR: Failed to create PersonalForecast task. Run this script as Administrator.
    goto :end
)

echo.
echo Both tasks created successfully!
echo.
echo Verify:  schtasks /query /tn "TradingAlgo\PropForecast"
echo          schtasks /query /tn "TradingAlgo\PersonalForecast"
echo Run now: schtasks /run /tn "TradingAlgo\PropForecast"
echo          schtasks /run /tn "TradingAlgo\PersonalForecast"
echo Delete:  schtasks /delete /tn "TradingAlgo\PropForecast" /f
echo          schtasks /delete /tn "TradingAlgo\PersonalForecast" /f

:end
pause
