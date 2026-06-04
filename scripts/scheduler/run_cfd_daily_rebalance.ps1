<#
.SYNOPSIS
    Wrapper for the CFD prop daily rebalance script, invoked by Task Scheduler.

.DESCRIPTION
    Runs scripts/enigma_cfd_prop_forecast.py with the configured Python
    interpreter, logs all stdout/stderr to a date-stamped file, and on
    non-zero exit code posts a Telegram alert (using the CFD prop bot)
    so the operator is notified out-of-band.

    Intended to be triggered by Task Scheduler at 18:10 ET, Sun-Thu.
    FTMO halts US500.cash/US100.cash/XAUUSD/XAGUSD from 16:49 ET to
    18:05 ET daily; the D1 bar closes at 17:00 ET inside that halt.
    18:10 ET = 5 min after the halt ends, so we get both the freshly
    closed D1 bar AND an active market that can fill orders.

    Sunday is included so we re-enter positions for Monday's session
    right after the weekly halt ends ~Sun 18:05 ET.

    On Fridays use run_cfd_weekend_close.ps1 instead (don't also run
    this on Fri — they would race for the same MT5 sessions, and the
    daily rebalance at 18:10 ET would land after the weekend halt has
    already started).

.PARAMETER RepoRoot
    Path to the Trading-Algo repository root. Defaults to two parents up
    from this script.

.PARAMETER PythonExe
    Path to the Python interpreter to use. Defaults to:
    <RepoRoot>\cpython_env\Scripts\python.exe (FTMO VPS convention).
    Falls back to <RepoRoot>\.venv\Scripts\python.exe or
    <RepoRoot>\venv\Scripts\python.exe if cpython_env is absent.

.PARAMETER ExtraArgs
    Additional CLI args to forward to the script (e.g. for ad-hoc runs).

.EXAMPLE
    .\run_cfd_daily_rebalance.ps1
    .\run_cfd_daily_rebalance.ps1 -ExtraArgs "--dry-run-execute"

.NOTES
    Required environment variables (set as user or machine env vars so
    Task Scheduler sees them, NOT just in your interactive shell):
      TELEGRAM_CFD_PROP_BOT_TOKEN
      TELEGRAM_CFD_PROP_CHAT_ID
      MT5_*_USERNAME / MT5_*_PASSWORD / MT5_*_SERVER (per account)
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
$logFile = Join-Path $logDir "cfd_daily_rebalance_$stamp.log"

$scriptArgs = @('scripts\enigma_cfd_prop_forecast.py', '--execute', '--approve-via-telegram') + $ExtraArgs

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
        $msg = "ALERT: CFD PROP DAILY REBALANCE FAILED`nexit code: $exit`nlog: $bt$logFile$bt`n$fence`n$tail`n$fence"
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
