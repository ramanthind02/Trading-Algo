<#
.SYNOPSIS
    Idempotent declarative registrar for TradingAlgo Windows Task Scheduler tasks.

.DESCRIPTION
    Declares five tasks in a $Tasks table and, when -Confirm is passed, registers
    them under the TradingAlgo\ task folder. Default (no flags) prints a dry-run
    notice. -WhatIf previews all task rows without registering anything.

    SUPERSEDES: deployment\ops\setup_scheduled_task.ps1 (legacy, left for reference).

    Tasks registered
    ----------------
      TradingAlgo\DailyChainAtLogon      AtLogon +60 s delay
                                         venv python -m deployment.ops.run_daily_chain
      TradingAlgo\DailyChainCatchup      Daily 17:15 local (catchup if box was off at logon)
                                         venv python -m deployment.ops.run_daily_chain
      TradingAlgo\MT5RolloverTickScrape  Daily 16:05 local
                                         deployment\ops\run_rollover_tick_scrape.bat
      TradingAlgo\DataFreshnessCheck     Daily 17:45 local (independent staleness tripwire)
                                         venv python -m deployment.ops.check_data_freshness
      TradingAlgo\RegistryBackup         Daily 23:00 local
                                         venv python -m data_platform.registry backup

    Conventions
    -----------
    * Non-elevated current-user principal (LogonType Interactive, RunLevel Limited).
      Nothing here needs admin — do NOT change to RunLevel Highest.
    * Kind=Local StartBoundary so triggers fire at WALL-CLOCK local time, not UTC.
    * StartWhenAvailable + MultipleInstances IgnoreNew + ExecutionTimeLimit 2 h.
    * Unregister-if-exists before each registration (idempotent re-run).
    * Double gate: -WhatIf ONLY previews; the default (no flags) shows a notice;
      actual registration requires the explicit -Confirm switch.

.PARAMETER WhatIf
    Preview the five task rows without registering anything. Safe to run any time.

.PARAMETER Confirm
    Actually register the tasks. Required in addition to not passing -WhatIf.

.PARAMETER RepoRoot
    Path to the Trading-Algo repo root. Defaults to two parents up from this script.

.EXAMPLE
    # Preview (safe — registers nothing):
    powershell -NoProfile -File deployment\ops\register_tasks.ps1 -WhatIf

    # Register:
    powershell -NoProfile -File deployment\ops\register_tasks.ps1 -Confirm

.NOTES
    After registering, verify:
        Get-ScheduledTask -TaskPath '\TradingAlgo\' |
            Get-ScheduledTaskInfo |
            Format-Table TaskName, NextRunTime, LastTaskResult -AutoSize
#>

param(
    [switch]$WhatIf,
    [switch]$Confirm,
    [string]$RepoRoot
)

$ErrorActionPreference = 'Stop'

# ── Repo root + venv resolution ───────────────────────────────────────────────

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

$VenvPython = @(
    (Join-Path $RepoRoot '.venv\Scripts\python.exe'),
    (Join-Path $RepoRoot 'venv\Scripts\python.exe')
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $VenvPython) {
    throw "No venv python found under $RepoRoot - expected .venv\Scripts\python.exe or venv\Scripts\python.exe"
}

# ── Task table ────────────────────────────────────────────────────────────────
# Each entry:
#   Name          Task name within \TradingAlgo\
#   TriggerType   'AtLogon' | 'Daily'
#   TriggerTime   'HH:mm' (Daily only; ignored for AtLogon)
#   TriggerDelay  Delay in seconds after logon (AtLogon only)
#   ActionExe     Executable path
#   ActionArgs    Argument string
#   Description   Human-readable purpose
#   MachineRole   Informational tag

$RolloverBat = Join-Path $RepoRoot 'deployment\ops\run_rollover_tick_scrape.bat'

$Tasks = @(
    [PSCustomObject]@{
        Name         = 'DailyChainAtLogon'
        TriggerType  = 'AtLogon'
        TriggerTime  = $null
        TriggerDelay = 60
        ActionExe    = $VenvPython
        ActionArgs   = '-m deployment.ops.run_daily_chain'
        Description  = 'Run full daily data-platform chain at logon (+60 s delay for MT5 terminal startup). Scrapes MT5, checks freshness, refreshes signal, ingests + backs up registry.'
        MachineRole  = 'dev-box'
    },
    [PSCustomObject]@{
        Name         = 'DailyChainCatchup'
        TriggerType  = 'Daily'
        TriggerTime  = '17:15'
        TriggerDelay = 0
        ActionExe    = $VenvPython
        ActionArgs   = '-m deployment.ops.run_daily_chain'
        Description  = 'Catchup run of daily data-platform chain at 17:15 local (fires even if the box was off at logon).'
        MachineRole  = 'dev-box'
    },
    [PSCustomObject]@{
        Name         = 'MT5RolloverTickScrape'
        TriggerType  = 'Daily'
        TriggerTime  = '16:05'
        TriggerDelay = 0
        ActionExe    = $RolloverBat
        ActionArgs   = ''
        Description  = '16:05 local daily; capture exit (16:00-16:59 NY) and entry (18:00-19:00 NY) rollover ticks. 4 parallel workers; ~15-20 min for all symbols.'
        MachineRole  = 'dev-box'
    },
    [PSCustomObject]@{
        Name         = 'DataFreshnessCheck'
        TriggerType  = 'Daily'
        TriggerTime  = '17:45'
        TriggerDelay = 0
        ActionExe    = $VenvPython
        ActionArgs   = '-m deployment.ops.check_data_freshness'
        Description  = '17:45 local daily; independent staleness tripwire — alerts if MT5 signal feed has not advanced, regardless of whether the DailyChain ran.'
        MachineRole  = 'dev-box'
    },
    [PSCustomObject]@{
        Name         = 'RegistryBackup'
        TriggerType  = 'Daily'
        TriggerTime  = '23:00'
        TriggerDelay = 0
        ActionExe    = $VenvPython
        ActionArgs   = '-m data_platform.registry backup'
        Description  = '23:00 local daily; SQLite hot-backup of data/registry.db to data/backups/ (ADR-10).'
        MachineRole  = 'dev-box'
    }
)

# ── WhatIf: preview only ──────────────────────────────────────────────────────

if ($WhatIf) {
    Write-Host ''
    Write-Host '=== WhatIf: would register the following TradingAlgo\ tasks ==='
    Write-Host ''
    $fmt = '{0,-26}  {1,-22}  {2}'
    Write-Host ($fmt -f 'TaskName', 'Trigger', 'Action')
    Write-Host ('-' * 90)
    foreach ($t in $Tasks) {
        $trigStr = if ($t.TriggerType -eq 'AtLogon') {
            "AtLogon +$($t.TriggerDelay)s"
        } else {
            "Daily @ $($t.TriggerTime) local"
        }
        $actionStr = if ($t.ActionExe -eq $VenvPython) {
            "venv\python.exe $($t.ActionArgs)"
        } else {
            [System.IO.Path]::GetFileName($t.ActionExe)
        }
        Write-Host ($fmt -f $t.Name, $trigStr, $actionStr)
    }
    Write-Host ''
    Write-Host "Repo root   : $RepoRoot"
    Write-Host "Venv python : $VenvPython"
    Write-Host ''
    Write-Host 'To register, re-run with -Confirm (double gate prevents accidental registration).'
    Write-Host '    powershell -NoProfile -File register_tasks.ps1 -Confirm'
    return
}

# ── Dry-run notice (no -Confirm) ─────────────────────────────────────────────

if (-not $Confirm) {
    Write-Host 'DRY-RUN: no -Confirm passed — nothing registered.'
    Write-Host '  Use -WhatIf to preview the task table.'
    Write-Host '  Use -Confirm to actually register.'
    Write-Host "  Repo root: $RepoRoot"
    Write-Host "  Venv python: $VenvPython"
    return
}

# ── Registration ──────────────────────────────────────────────────────────────

Write-Host "Registering TradingAlgo tasks under $RepoRoot ..."

# Non-elevated current-user principal — nothing here needs admin.
# DO NOT change RunLevel to Highest without a concrete reason.
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

foreach ($t in $Tasks) {

    # Idempotency: remove existing task before re-registering.
    if (Get-ScheduledTask -TaskPath '\TradingAlgo\' -TaskName $t.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskPath '\TradingAlgo\' -TaskName $t.Name -Confirm:$false
        Write-Host "  Removed existing: TradingAlgo\$($t.Name)"
    }

    # Build trigger with Kind=Local so it fires at wall-clock time, not UTC.
    if ($t.TriggerType -eq 'AtLogon') {
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
        # Set delay via the CIM property (New-ScheduledTaskTrigger -AtLogon has no -Delay param).
        $trigger.Delay = "PT$($t.TriggerDelay)S"
    } else {
        $at = [DateTime]::SpecifyKind(
            [DateTime]::ParseExact($t.TriggerTime, 'HH:mm', $null),
            [DateTimeKind]::Local
        )
        $trigger = New-ScheduledTaskTrigger -Daily -At $at
    }

    # Build action.
    $actionParams = @{
        Execute          = $t.ActionExe
        WorkingDirectory = $RepoRoot
    }
    if ($t.ActionArgs) { $actionParams.Argument = $t.ActionArgs }
    $action = New-ScheduledTaskAction @actionParams

    Register-ScheduledTask `
        -TaskPath    '\TradingAlgo\' `
        -TaskName    $t.Name `
        -Action      $action `
        -Trigger     $trigger `
        -Settings    $Settings `
        -Principal   $Principal `
        -Description $t.Description | Out-Null

    $trigStr = if ($t.TriggerType -eq 'AtLogon') {
        "AtLogon+$($t.TriggerDelay)s"
    } else {
        "daily@$($t.TriggerTime)"
    }
    Write-Host "  Registered: TradingAlgo\$($t.Name)  ($trigStr)"
}

Write-Host ''
Write-Host 'Done. Verify next-run times:'
Write-Host "    Get-ScheduledTask -TaskPath '\TradingAlgo\' | Get-ScheduledTaskInfo | Format-Table TaskName, NextRunTime, LastTaskResult -AutoSize"
