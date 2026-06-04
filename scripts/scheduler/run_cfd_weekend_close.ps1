<#
.SYNOPSIS
    Wrapper for the CFD prop weekend-close script, invoked by Task Scheduler.

.DESCRIPTION
    Runs scripts/enigma_cfd_prop_weekend_close.py with the configured
    Python interpreter, logs all stdout/stderr to a date-stamped file,
    and on non-zero exit code posts a Telegram alert via the CFD prop
    bot. Intended to be triggered by Task Scheduler at 16:45 ET, FRIDAY
    only. 16:45 leaves 15 minutes for script + Telegram approval to
    finish before the 17:00 ET swap charge, so closes dodge the
    weekend triple-swap on any positions flattened.

    Do NOT also schedule the daily rebalance on Friday — they would
    race for the same MT5 sessions.

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
    $token   = [Environment]::GetEnvironmentVariable('TELEGRAM_CFD_PROP_BOT_TOKEN')
    $chat_id = [Environment]::GetEnvironmentVariable('TELEGRAM_CFD_PROP_CHAT_ID')
    if ($token -and $chat_id) {
        $tail = (Get-Content -Path $logFile -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
        $bt = [char]0x60
        $fence = "$bt$bt$bt"
        $msg = "ALERT: CFD PROP WEEKEND-CLOSE FAILED`nexit code: $exit`nlog: $bt$logFile$bt`n$fence`n$tail`n$fence"
        try {
            Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$token/sendMessage" -Body @{
                chat_id = $chat_id
                text = $msg
                parse_mode = 'Markdown'
            } | Out-Null
            Write-Host "[$(Get-Date -Format o)] failure alert sent to Telegram chat $chat_id"
        } catch {
            Write-Warning "Failed to post Telegram alert: $_"
        }
    } else {
        Write-Warning "TELEGRAM_CFD_PROP_BOT_TOKEN / CHAT_ID not set; cannot send failure alert."
    }
}

exit $exit
