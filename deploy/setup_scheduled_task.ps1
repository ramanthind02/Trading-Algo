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
#     powershell -ExecutionPolicy Bypass -File deploy\setup_scheduled_task.ps1
#
# Times below are LOCAL (Pacific). Edit the -At values if you move
# timezones; PowerShell will compute the UTC offset at fire time and
# auto-adjust for DST.
# =============================================================

$ErrorActionPreference = 'Stop'

$RepoRoot = 'C:\Users\adabla\Trading-Algo'
$LogPath  = Join-Path $RepoRoot 'deploy\forecast.log'

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
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

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
    -BatPath  (Join-Path $RepoRoot 'deploy\run_personal_forecast.bat') `
    -LocalFireTime $personalAt `
    -Description '12:45 PM PT daily; ETF rebalance signal pre-equity-close.'

# Prop forecast: 3:00 PM PT = 6:00 PM ET (after CME settlement window).
$propAt = [DateTime]::Today.AddHours(15)
Register-ForecastTask `
    -TaskName 'PropForecast' `
    -BatPath  (Join-Path $RepoRoot 'deploy\run_prop_forecast.bat') `
    -LocalFireTime $propAt `
    -Description '3:00 PM PT daily; futures signal post-CME-settlement.'

# MT5 data scrape: 5:00 PM PT = 8:00 PM ET (after US equity close + data settle).
$mt5ScrapeAt = [DateTime]::Today.AddHours(17)
Register-ForecastTask `
    -TaskName 'MT5DataScrape' `
    -BatPath  (Join-Path $RepoRoot 'deploy\run_mt5_scrape.bat') `
    -LocalFireTime $mt5ScrapeAt `
    -Description '5:00 PM PT daily; incremental MT5 M1 bar scrape for all symbols.'

Write-Host "`nVerify next-run times:"
Get-ScheduledTask -TaskPath '\TradingAlgo\' |
    ForEach-Object {
        $info = $_ | Get-ScheduledTaskInfo
        '{0,-30} next run: {1}' -f $_.TaskName, $info.NextRunTime
    }
