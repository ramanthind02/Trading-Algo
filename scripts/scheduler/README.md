# CFD prop scheduler

Operator runbook for automating the CFD prop daily rebalance + weekly weekend close on a Windows VPS (FTMO terminal box).

## What runs when

| Task | Days | Time (local) | What it does |
|---|---|---|---|
| `EnigmaCfdProp-DailyRebalance` | Sun-Thu | 18:10 | `scripts/enigma_cfd_prop_forecast.py --execute --approve-via-telegram` |
| `EnigmaCfdProp-WeekendClose` | Fri | 16:30 | `scripts/enigma_cfd_prop_weekend_close.py --execute --approve-via-telegram` |

### Why these times — empirically verified

FTMO halts US500.cash / US100.cash / XAUUSD / XAGUSD from **16:49 ET to 18:05 ET every weekday** for the daily server reset (verified empirically via `scripts/mt5_diagnose_trading_session.py` — 76-min window, identical across all four symbols). The D1 bar closes at **17:00 ET INSIDE the halt**. Our forecast script strips today's in-progress D1 bar to avoid feeding the model partial data, so:

- **18:10 ET daily** = 5 min after the halt ends. The D1 bar closed cleanly at 17:00 ET (during the halt), so by 18:10 ET it IS the freshest input AND the market is actively trading and will fill orders. Running earlier (e.g. 17:05 ET) would land INSIDE the halt and orders would reject. Sunday is included so we re-enter positions for Monday's session right after the weekly halt ends ~Sun 18:05 ET.
- **16:30 ET Friday close** leaves a 19-min buffer for the 5-min Telegram approval poll + execution before the 16:49 ET halt locks the book. Closes also happen before the Fri-night rollover so indices (`swap_rollover3days = Fri`) dodge the weekend triple-swap.

Both:
- Wait for Telegram batch approval (5-min timeout, default = cancel)
- Log to `logs/scheduler/<task>_YYYYMMDD_HHMMSS.log`
- On Python exit code ≠ 0, post a `🚨` alert to the CFD prop Telegram channel

There is no built-in US-holiday skip. The Telegram approval IS the safety net — if a holiday or other off-day looks wrong, click `Cancel All`. Manual one-off positions on holidays (e.g. closing early for Christmas Eve) can be done from the FTMO terminal directly.

**Do NOT also schedule the daily rebalance on Friday** — the weekend-close task replaces it for that day. Running both would race the same MT5 sessions, and the daily run at 18:10 ET would land after the weekend halt has already started.

## One-time setup

### 1. Set machine timezone to America/New_York

Triggers fire in **local** Windows time. Easiest is to set the VPS clock to Eastern.

```powershell
# Check current:
Get-TimeZone

# Set (requires admin shell):
Set-TimeZone -Id "Eastern Standard Time"
```

Windows handles DST automatically — `"Eastern Standard Time"` is the ID for `America/New_York` (EST ↔ EDT) regardless of the misleading name.

If you can't change the VPS timezone, adjust `-DailyRunTime` / `-WeekendCloseTime` on the install script accordingly. Examples for the defaults:
- UTC: pass `-DailyRunTime "22:05" -WeekendCloseTime "21:45"` in winter, `"21:05"/"20:45"` in summer (manual DST 😞)
- Pacific: pass `-DailyRunTime "14:05" -WeekendCloseTime "13:45"`

### 1b. (Optional but recommended) Verify FTMO broker time alignment

The 18:10 / 16:30 ET defaults assume FTMO server time is GMT+2/+3 AND that FTMO maintains the 16:49-18:05 ET daily halt window. Confirm on the actual VPS:

```powershell
.\cpython_env\Scripts\python.exe scripts\mt5_diagnose_trading_session.py
```

This empirically reports the broker offset, D1 bar boundaries, AND the no-trade window (start time and duration) for all four symbols. If anything disagrees with the defaults, adjust the install-script params accordingly.

A quicker server-time-only check (no symbol fetches) is also available:

```powershell
.\cpython_env\Scripts\python.exe scripts\mt5_check_server_time.py
```

### 2. Set environment variables (persistent, not session)

Task Scheduler runs in a fresh shell — env vars from your interactive session are invisible to it. Use **user-level** env vars:

```powershell
# Telegram (already have these from validation)
[Environment]::SetEnvironmentVariable('TELEGRAM_CFD_PROP_BOT_TOKEN', '<token>', 'User')
[Environment]::SetEnvironmentVariable('TELEGRAM_CFD_PROP_CHAT_ID',  '<chat_id>', 'User')

# Per MT5 account (one set per account label in your config)
[Environment]::SetEnvironmentVariable('MT5_FTMO_100K_DEMO_A_USERNAME', '1513537520', 'User')
[Environment]::SetEnvironmentVariable('MT5_FTMO_100K_DEMO_A_PASSWORD', '<password>', 'User')
[Environment]::SetEnvironmentVariable('MT5_FTMO_100K_DEMO_A_SERVER',   'FTMO-Demo', 'User')
```

Verify after restarting your shell:

```powershell
Get-ChildItem env: | Where-Object Name -like 'TELEGRAM_*'
Get-ChildItem env: | Where-Object Name -like 'MT5_*'
```

### 3. Install the tasks

```powershell
cd C:\Users\adabla\Trading\Trading-Algo
.\scripts\scheduler\install_cfd_prop_tasks.ps1
```

To override the schedule (e.g. broker offset differs from FTMO, or daily halt window shifts):

```powershell
.\scripts\scheduler\install_cfd_prop_tasks.ps1 -DailyRunTime "18:10" -WeekendCloseTime "16:30"
```

To have the tasks run when nobody is logged in (recommended for a VPS):

```powershell
.\scripts\scheduler\install_cfd_prop_tasks.ps1 -Credential (Get-Credential)
# Enter the Windows user + password when prompted.
```

### 4. Verify

```powershell
Get-ScheduledTask -TaskName 'EnigmaCfdProp-*' | Format-Table TaskName, State, NextRunTime

# Manual smoke test (won't wait for trigger; runs immediately):
Start-ScheduledTask -TaskName 'EnigmaCfdProp-DailyRebalance'

# Tail the latest log:
Get-ChildItem .\logs\scheduler\cfd_daily_rebalance_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50
```

## Operating

### Disable temporarily (e.g. you're away and don't want trades)

```powershell
Disable-ScheduledTask -TaskName 'EnigmaCfdProp-DailyRebalance'
Disable-ScheduledTask -TaskName 'EnigmaCfdProp-WeekendClose'
```

Re-enable with `Enable-ScheduledTask`.

### Update wrapper or python script

The install script is idempotent — re-run it and the tasks are recreated pointing at the latest wrapper:

```powershell
git pull
.\scripts\scheduler\install_cfd_prop_tasks.ps1
```

### Uninstall completely

```powershell
.\scripts\scheduler\uninstall_cfd_prop_tasks.ps1
```

## What happens on a failure

The PowerShell wrapper detects non-zero exit code → posts the last 20 lines of the log to the CFD prop Telegram channel with a `🚨` prefix. Investigate by reading the full log under `logs/scheduler/`.

Common failure modes:
- MT5 terminal not running → start FTMO terminal manually
- Env var missing → re-check step 2 (Task Scheduler ≠ your interactive shell)
- Telegram poll-unhealthy → check bot token, channel admin rights
- MT5 login refused → check FTMO credentials haven't expired

## Files in this directory

| File | Purpose |
|---|---|
| `run_cfd_daily_rebalance.ps1` | Wrapper called by the daily task |
| `run_cfd_weekend_close.ps1` | Wrapper called by the weekend-close task |
| `install_cfd_prop_tasks.ps1` | Idempotent task registration |
| `uninstall_cfd_prop_tasks.ps1` | Remove the two tasks |
| `README.md` | This file |
