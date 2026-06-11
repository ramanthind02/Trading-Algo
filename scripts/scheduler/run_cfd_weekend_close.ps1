<#
.SYNOPSIS
    Wrapper for the CFD prop weekend-close script, invoked by Task Scheduler.

.DESCRIPTION
    Runs scripts/enigma_cfd_prop_weekend_close.py with the configured
    Python interpreter, logs all stdout/stderr to a date-stamped file,
    and on non-zero exit code posts a Telegram alert via the CFD prop
    bot. Intended to be triggered by Task Scheduler at 16:30 ET, FRIDAY
    only.

    FTMO halts the index/metal CFDs at 16:49 ET on Friday and they
    don't reopen until Sunday ~18:05 ET. 16:30 ET leaves a 19-min
    buffer for the 5-min Telegram approval poll + order execution
    before the halt locks the book. Closes also happen before the
    Fri-night rollover so indices (rollover3days = Fri) dodge the
    weekend triple-swap.

    Do NOT also schedule the daily rebalance on Friday — they would
    race for the same MT5 sessions, and the daily run at 18:10 ET
    would land after the weekend halt has already started.

.PARAMETER RepoRoot
    Path to the Trading-Algo repository root. Defaults to two parents
    up from this script.

.PARAMETER PythonExe
    Path to the Python interpreter. Same auto-detect order as the daily
    rebalance wrapper.

.PARAMETER ExtraArgs
    Additional CLI args to forward (e.g. --dry-run-execute).

.EXAMPLE
    .\run_cfd_weekend_close.ps1
    .\run_cfd_weekend_close.ps1 -ExtraArgs "--dry-run-execute"
#>

param(
    [string]$RepoRoot,
    [string]$PythonExe,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Continue'

if (-not $RepoRoot) {
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
}

if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $RepoRoot 'cpython_env\Scripts\python.exe'),
        (Join-Path $RepoRoot '.venv\Scripts\python.exe'),
        (Join-Path $RepoRoot 'venv\Scripts\python.exe')
    )
    $PythonExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $PythonExe) {
        Write-Error "No Python interpreter found in any of: $($candidates -join ', ')"
        exit 2
    }
}

$logDir = Join-Path $RepoRoot 'logs\scheduler'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "cfd_weekend_close_$stamp.log"

$scriptArgs = @('scripts\enigma_cfd_prop_weekend_close.py', '--execute', '--approve-via-telegram') + $ExtraArgs

Write-Host "[$(Get-Date -Format o)] starting: $PythonExe $($scriptArgs -join ' ')"
Write-Host "[$(Get-Date -Format o)] log file: $logFile"

Push-Location $RepoRoot
try {
    & $PythonExe @scriptArgs *>&1 | Tee-Object -FilePath $logFile
    $exit = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host "[$(Get-Date -Format o)] python exit code: $exit"

if ($exit -ne 0) {
    $tail = (Get-Content -Path $logFile -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
    $msg = "ALERT: CFD PROP WEEKEND-CLOSE FAILED`nexit code: $exit`nlog: $logFile`n$tail"
    Push-Location $RepoRoot
    try {
        & $PythonExe -m lib.core.notify --channel cfd_prop --message $msg
        Write-Host "[$(Get-Date -Format o)] failure alert dispatched via lib.core.notify (exit $LASTEXITCODE)"
    } catch {
        Write-Warning "Failed to invoke lib.core.notify for failure alert: $_"
    } finally {
        Pop-Location
    }
}

exit $exit
