# =============================================================
# DEPRECATED — use deployment\ops\register_tasks.ps1 instead.
#
# This script was the original single-function registrar for the IB-pipeline
# era (PropForecast / PersonalForecast tasks). Those tasks are retired; the
# MT5DataScrape, DataFreshnessCheck, and MT5RolloverTickScrape tasks have
# moved to the new declarative table in register_tasks.ps1, which also adds
# DailyChainAtLogon, DailyChainCatchup, and RegistryBackup.
#
# Left in place for reference; do not run for new installs.
# =============================================================
#
# =============================================================
# Registers the daily Enigma forecast tasks via PowerShell, which
# carries an explicit `DateTime.Kind = Local` on the trigger so the
# task fires at WALL-CLOCK 3:00 PM PT / 12:45 PM PT, not 3:00 PM UTC.
#
# (schtasks /st leaves the StartBoundary unzoned; some Windows builds
# interpret that as UTC, which fired our prop task at 8 AM PT and the
# personal task at 5:45 AM PT. This script fixes that.)
#
# Run ONCE as Administrator:
#     powershell -ExecutionPolicy Bypass -File deployment\ops\setup_scheduled_task.ps1
#
# Times below are LOCAL (Pacific). Edit the -At values if you move
# timezones; PowerShell will compute the UTC offset at fire time and
# auto-adjust for DST.
# =============================================================

$ErrorActionPreference = 'Stop'

# Auto-detect the repo root from this script's location (deployment\ops\setup_scheduled_task.ps1).
# Previously hardcoded to a wrong user path ('C:\Users\adabla\Trading-Algo'), which silently
# registered every task against a non-existent .bat — so the scrape never ran and the signal
# went stale undetected. Never hardcode a per-machine path here again.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogPath  = Join-Path $RepoRoot 'logs\forecast.log'

function Register-ForecastTask {
    param(
        [Parameter(Mandatory)] [string] $TaskName,
        [Parameter(Mandatory)] [string] $BatPath,
        [Parameter(Mandatory)] [DateTime] $LocalFireTime,
        [Parameter(Mandatory)] [string] $Description
    )

    # Force Kind=Local so the trigger fires at wall-clock time in our zone.
    $localBoundary = [DateTime]::SpecifyKind($LocalFireTime, [DateTimeKind]::Local)

    $action  = New-ScheduledTaskAction -Execute $BatPath
    $trigger = New-ScheduledTaskTrigger -Daily -At $localBoundary
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    # UserId must be the domain-qualified account ("COMPUTER\user"); a bare username
    # ($env:USERNAME) makes Register-ScheduledTask throw "The parameter is incorrect: UserId".
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

    if (Get-ScheduledTask -TaskPath '\TradingAlgo\' -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskPath '\TradingAlgo\' -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask `
        -TaskPath '\TradingAlgo\' `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description | Out-Null

    Write-Host ("Registered TradingAlgo\{0} -> daily at {1} (local)" -f $TaskName, $localBoundary)
}

# Personal forecast: 12:45 PM PT = 3:45 PM ET (15 min before equity close).
$personalAt = [DateTime]::Today.AddHours(12).AddMinutes(45)
Register-ForecastTask `
    -TaskName 'PersonalForecast' `
    -BatPath  (Join-Path $RepoRoot 'deployment\ops\run_personal_forecast.bat') `
    -LocalFireTime $personalAt `
    -Description '12:45 PM PT daily; ETF rebalance signal pre-equity-close.'

# Prop forecast: 3:00 PM PT = 6:00 PM ET (after CME settlement window).
$propAt = [DateTime]::Today.AddHours(15)
Register-ForecastTask `
    -TaskName 'PropForecast' `
    -BatPath  (Join-Path $RepoRoot 'deployment\ops\run_prop_forecast.bat') `
    -LocalFireTime $propAt `
    -Description '3:00 PM PT daily; futures signal post-CME-settlement.'

# MT5 data scrape: 5:00 PM PT = 8:00 PM ET (after US equity close + data settle).
$mt5ScrapeAt = [DateTime]::Today.AddHours(17)
Register-ForecastTask `
    -TaskName 'MT5DataScrape' `
    -BatPath  (Join-Path $RepoRoot 'deployment\ops\run_mt5_scrape.bat') `
    -LocalFireTime $mt5ScrapeAt `
    -Description '5:00 PM PT daily; incremental MT5 M1 bar scrape for all symbols.'

# Data-freshness tripwire: 5:45 PM PT = 45 min after the scrape. INDEPENDENT of the
# scrape task — fires (and alerts via logs\data_freshness.json + Telegram-if-set) even
# if MT5DataScrape never ran, so a silently-broken schedule can't go unnoticed again.
$freshnessAt = [DateTime]::Today.AddHours(17).AddMinutes(45)
Register-ForecastTask `
    -TaskName 'DataFreshnessCheck' `
    -BatPath  (Join-Path $RepoRoot 'deployment\ops\run_data_freshness_check.bat') `
    -LocalFireTime $freshnessAt `
    -Description '5:45 PM PT daily; alert if the MT5 signal feed is stale (independent tripwire).'

# Rollover tick scrape: 4:05 PM PT = 7:05 PM ET.
# Both windows have fully closed by this time:
#   exit  window: 16:00-16:59 NY (19:00-19:59 UTC winter / 20:00-20:59 UTC summer)
#   entry window: 18:00-19:00 NY (23:00-00:00 UTC winter / 22:00-23:00 UTC summer)
# 4 parallel workers; 844 symbols typically complete in ~15-20 min.
$rolloverAt = [DateTime]::Today.AddHours(16).AddMinutes(5)
Register-ForecastTask `
    -TaskName 'MT5RolloverTickScrape' `
    -BatPath  (Join-Path $RepoRoot 'deployment\ops\run_rollover_tick_scrape.bat') `
    -LocalFireTime $rolloverAt `
    -Description '4:05 PM PT daily; exit (16:00-16:59 NY) + entry (18:00-19:00 NY) rollover tick capture.'

Write-Host "`nVerify next-run times:"
Get-ScheduledTask -TaskPath '\TradingAlgo\' |
    ForEach-Object {
        $info = $_ | Get-ScheduledTaskInfo
        '{0,-30} next run: {1}' -f $_.TaskName, $info.NextRunTime
    }
