# CFD prop scheduler

Operator runbook for automating the CFD prop daily rebalance + weekly weekend close on a Windows VPS (FTMO terminal box).

## What runs when

| Task | Days | Time (local) | What it does |
|---|---|---|---|
| `EnigmaCfdProp-DailyRebalance` | Mon-Thu | 15:30 | `scripts/enigma_cfd_prop_forecast.py --execute --approve-via-telegram` |
| `EnigmaCfdProp-WeekendClose` | Fri | 15:30 | `scripts/enigma_cfd_prop_weekend_close.py --execute --approve-via-telegram` |

Both:
- Wait for Telegram batch approval (5-min timeout, default = cancel)
- Auto-skip on US market holidays / early-close days (Thanksgiving, Christmas, etc.) — see `execution/us_market_session.py`
- Log to `logs/scheduler/<task>_YYYYMMDD_HHMMSS.log`
- On Python exit code ≠ 0, post a `🚨` alert to the CFD prop Telegram channel

**Do NOT also schedule the daily rebalance on Friday** — the weekend-close task replaces it for that day. Running both would race the same MT5 sessions.

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

If you can't change the VPS timezone, adjust `-RunTime` on the install script accordingly. Examples for 15:30 ET:
- UTC: pass `-RunTime "20:30"` in winter, `"19:30"` in summer (manual DST 😞)
- Pacific: pass `-RunTime "12:30"`

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

## What happens on a holiday

The Python script checks `execution.us_market_session.decide_session()` first and exits cleanly with `[market-calendar] Skipping ...: <reason>` printed to the log. The Telegram channel does NOT get spammed on holidays — only on actual failures.

If you really want to run on a US holiday (e.g. you're trading metals only):

```powershell
# Manual one-off:
.\cpython_env\Scripts\python.exe scripts\enigma_cfd_prop_forecast.py --execute --approve-via-telegram --ignore-market-calendar
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
