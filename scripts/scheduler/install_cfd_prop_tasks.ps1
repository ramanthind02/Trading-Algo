<#
.SYNOPSIS
    Idempotently registers Windows Task Scheduler tasks for CFD prop
    daily rebalance (Sun-Thu) and weekend close (Fri).

.DESCRIPTION
    Creates two scheduled tasks:

      1. EnigmaCfdProp-DailyRebalance
           Trigger: Sun-Thu at $DailyRunTime (default 18:10 local time)
           Action:  scripts\scheduler\run_cfd_daily_rebalance.ps1

         FTMO halts trading on US500.cash / US100.cash / XAUUSD /
         XAGUSD from 16:49 ET to 18:05 ET every weekday for the daily
         server reset (verified empirically — see
         scripts/mt5_diagnose_trading_session.py). The D1 bar closes
         at 17:00 ET DURING the halt, so by 18:10 ET (5 min after the
         halt ends):
            - the just-closed D1 bar IS the freshest input
            - the market is actively trading and orders fill
         Running earlier than 18:05 ET (e.g. 17:05 ET) lands inside
         the halt -> orders reject.
         Sunday is included so we re-enter positions for Monday's
         session right after the weekly halt ends (~17:00-18:05 ET Sun).

      2. EnigmaCfdProp-WeekendClose
           Trigger: Fri at $WeekendCloseTime (default 16:30 local time)
           Action:  scripts\scheduler\run_cfd_weekend_close.ps1

         The Friday halt at 16:49 ET ushers in the weekend; the next
         trading window is Sun ~18:05 ET. Closing at 16:30 ET leaves
         a 19-min buffer for Telegram approval (5 min) + execution
         before the halt locks the book, AND closes happen before the
         Fri-night rollover so indices (rollover3days = Fri) dodge
         the weekend triple-swap.

    Both tasks are configured to:
      - Run whether the user is logged on or not (if you pass -Credential)
      - Survive machine sleep with -StartWhenAvailable
      - Skip if already running (no overlap)

    Time zone:
      Task Scheduler triggers fire in LOCAL machine time. The defaults
      18:10 / 16:30 assume the VPS is on Eastern. If your VPS is on
      another timezone you can either:
        (a) change the machine timezone to Eastern with
              Set-TimeZone -Id "Eastern Standard Time"
            (WARNING: this is machine-wide and persistent; affects
            every other process on the VPS), OR
        (b) leave the machine timezone alone and pass the converted
            local-time equivalents — e.g. on a Pacific VPS:
              -DailyRunTime "15:10" -WeekendCloseTime "13:30"
            (PT = ET - 3h year-round; both follow US DST in lockstep
            so no manual adjustment is ever needed.) See scheduler
            README §1a for a per-zone conversion table.

    Verify the FTMO broker session windows still match these defaults
    by re-running:
        python scripts\mt5_diagnose_trading_session.py
    after DST transitions or any FTMO server config change.

    Idempotency: if either task already exists, it is unregistered and
    recreated. Safe to re-run after script changes.

.PARAMETER DailyRunTime
    Time-of-day to trigger the daily rebalance (24h "HH:mm").
    Default "18:10".

.PARAMETER WeekendCloseTime
    Time-of-day to trigger the weekend close (24h "HH:mm").
    Default "16:30".

.PARAMETER RepoRoot
    Path to the Trading-Algo repo. Default: two parents up from this script.

.PARAMETER User
    User account to run the tasks as. Default: current user.

.PARAMETER Credential
    Optional PSCredential to run "whether logged on or not". If omitted,
    tasks run only when the user is interactively logged on.

.EXAMPLE
    .\install_cfd_prop_tasks.ps1
    .\install_cfd_prop_tasks.ps1 -DailyRunTime "18:10" -WeekendCloseTime "16:30"
    .\install_cfd_prop_tasks.ps1 -Credential (Get-Credential)
#>

param(
    [string]$DailyRunTime = '18:10',
    [string]$WeekendCloseTime = '16:30',
    [string]$RepoRoot,
    [string]$User = "$env:USERDOMAIN\$env:USERNAME",
    [System.Management.Automation.PSCredential]$Credential
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
}

$dailyWrapper   = Join-Path $RepoRoot 'scripts\scheduler\run_cfd_daily_rebalance.ps1'
$weekendWrapper = Join-Path $RepoRoot 'scripts\scheduler\run_cfd_weekend_close.ps1'

foreach ($path in @($dailyWrapper, $weekendWrapper)) {
    if (-not (Test-Path $path)) {
        throw "Wrapper not found: $path"
    }
}

function Register-Cfd {
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [Parameter(Mandatory)][string]$WrapperPath,
        [Parameter(Mandatory)][string[]]$DaysOfWeek,
        [Parameter(Mandatory)][string]$Time
    )

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Removing existing task: $TaskName"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperPath`""
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument $argList `
        -WorkingDirectory $RepoRoot

    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $Time

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)

    $registerArgs = @{
        TaskName    = $TaskName
        Action      = $action
        Trigger     = $trigger
        Settings    = $settings
        Description = "Enigma CFD prop firm scheduler — $TaskName"
    }

    if ($Credential) {
        $registerArgs.User     = $Credential.UserName
        $registerArgs.Password = $Credential.GetNetworkCredential().Password
        $registerArgs.RunLevel = 'Highest'
    } else {
        $registerArgs.User = $User
    }

    Register-ScheduledTask @registerArgs | Out-Null
    Write-Host ("Registered: {0}  ({1} @ {2} local time)" -f $TaskName, ($DaysOfWeek -join ','), $Time)
}

Register-Cfd `
    -TaskName 'EnigmaCfdProp-DailyRebalance' `
    -WrapperPath $dailyWrapper `
    -DaysOfWeek @('Sunday','Monday','Tuesday','Wednesday','Thursday') `
    -Time $DailyRunTime

Register-Cfd `
    -TaskName 'EnigmaCfdProp-WeekendClose' `
    -WrapperPath $weekendWrapper `
    -DaysOfWeek @('Friday') `
    -Time $WeekendCloseTime

Write-Host ""
Write-Host "Done. Verify state + next run time:"
Write-Host "    Get-ScheduledTask -TaskName 'EnigmaCfdProp-*' | Get-ScheduledTaskInfo | Format-Table TaskName, NextRunTime, LastRunTime, LastTaskResult"
Write-Host "(NextRunTime is on the *Info* object, not the bare ScheduledTask object — Get-ScheduledTask alone shows it as blank.)"
Write-Host ""
Write-Host "Test both wrappers manually (smoke test — won't wait for trigger):"
Write-Host "    Start-ScheduledTask -TaskName 'EnigmaCfdProp-DailyRebalance'"
Write-Host "    Start-ScheduledTask -TaskName 'EnigmaCfdProp-WeekendClose'"
Write-Host "Each will run the wrapper end-to-end (incl. Telegram approval poll). The weekend-close wrapper is a no-op when no positions are open."
Write-Host ""
Write-Host "Trigger time interpretation:"
Write-Host "    Task Scheduler fires triggers in the machine's LOCAL time."
Write-Host "    The default 18:10 / 16:30 assume the machine is on Eastern."
Write-Host "    If your machine is on another tz, you should have passed converted"
Write-Host "    times via -DailyRunTime / -WeekendCloseTime (see scheduler README §1a)."
Write-Host "    Check current machine tz with:"
Write-Host "        Get-TimeZone"
Write-Host ""
