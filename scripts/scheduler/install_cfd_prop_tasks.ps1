<#
.SYNOPSIS
    Idempotently registers Windows Task Scheduler tasks for CFD prop
    daily rebalance (Mon-Thu) and weekend close (Fri).

.DESCRIPTION
    Creates two scheduled tasks:

      1. EnigmaCfdProp-DailyRebalance
           Trigger: Mon-Thu at $RunTime (default 15:30 local time)
           Action:  scripts\scheduler\run_cfd_daily_rebalance.ps1

      2. EnigmaCfdProp-WeekendClose
           Trigger: Fri at $RunTime (default 15:30 local time)
           Action:  scripts\scheduler\run_cfd_weekend_close.ps1

    Both tasks are configured to:
      - Run whether the user is logged on or not (if you pass -Credential)
      - Survive machine sleep with -StartWhenAvailable
      - Skip if already running (no overlap)

    Time zone:
      Task Scheduler triggers fire in LOCAL machine time. For 15:30 ET
      you want the FTMO VPS clock set to America/New_York. Check with:
          Get-TimeZone
      Set it (admin shell) with:
          Set-TimeZone -Id "Eastern Standard Time"
      (Windows handles DST automatically; "Eastern Standard Time" IS
      the Windows ID for America/New_York, despite the name.)

    Idempotency: if either task already exists, it is unregistered and
    recreated. Safe to re-run after script changes.

.PARAMETER RunTime
    Time-of-day to trigger (24h "HH:mm"). Default "15:30".

.PARAMETER RepoRoot
    Path to the Trading-Algo repo. Default: two parents up from this script.

.PARAMETER User
    User account to run the tasks as. Default: current user.

.PARAMETER Credential
    Optional PSCredential to run "whether logged on or not". If omitted,
    tasks run only when the user is interactively logged on.

.EXAMPLE
    .\install_cfd_prop_tasks.ps1
    .\install_cfd_prop_tasks.ps1 -RunTime "15:30"
    .\install_cfd_prop_tasks.ps1 -Credential (Get-Credential)
#>

param(
    [string]$RunTime = '15:30',
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
    -DaysOfWeek @('Monday','Tuesday','Wednesday','Thursday') `
    -Time $RunTime

Register-Cfd `
    -TaskName 'EnigmaCfdProp-WeekendClose' `
    -WrapperPath $weekendWrapper `
    -DaysOfWeek @('Friday') `
    -Time $RunTime

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "    Get-ScheduledTask -TaskName 'EnigmaCfdProp-*' | Format-Table TaskName, State"
Write-Host ""
Write-Host "Test a manual run (no waiting for trigger):"
Write-Host "    Start-ScheduledTask -TaskName 'EnigmaCfdProp-DailyRebalance'"
Write-Host ""
Write-Host "Check tz is set to America/New_York:"
Write-Host "    Get-TimeZone   # expect 'Eastern Standard Time'"
