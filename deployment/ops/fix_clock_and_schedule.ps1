# =============================================================
# ONE-TIME fix: correct the system clock (it free-ran ~7h fast, never NTP-synced)
# and (re)register the daily MT5 scrape + data-freshness tasks.
#
# Run in an ELEVATED PowerShell (right-click PowerShell -> Run as administrator):
#     powershell -ExecutionPolicy Bypass -File deployment\ops\fix_clock_and_schedule.ps1
#
# IMPORTANT: stop the live vault nodes BEFORE running this — a backward clock jump
# can freeze their wall-clock timers. Re-arm them after (positions persist on the
# broker; re-arm just reconciles + holds, since we're past the entry window).
# =============================================================
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

Write-Host "BEFORE  system local: $(Get-Date)  | UTC: $((Get-Date).ToUniversalTime())"

# 1. Allow a LARGE one-time correction (w32tm refuses multi-hour jumps by default).
$cfg = 'HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config'
Set-ItemProperty $cfg -Name MaxNegPhaseCorrection -Value 0xFFFFFFFF -Type DWord
Set-ItemProperty $cfg -Name MaxPosPhaseCorrection -Value 0xFFFFFFFF -Type DWord

# 2. Point the time service at NTP, restart it, and force a resync.
w32tm /config /manualpeerlist:"time.windows.com,0x9 pool.ntp.org,0x9" /syncfromflags:manual /update | Out-Null
Restart-Service w32time
Start-Sleep -Seconds 2
try { w32tm /resync /force | Out-Null } catch { Write-Warning "resync round 1: $($_.Exception.Message)" }
Start-Sleep -Seconds 3
try { w32tm /resync /force | Out-Null } catch { Write-Warning "resync round 2: $($_.Exception.Message)" }

Write-Host "AFTER   system local: $(Get-Date)  | UTC: $((Get-Date).ToUniversalTime())"
w32tm /query /status | Select-Object -First 4

# 3. Report sync state. We do NOT auto-nudge — a double-correction would leave the clock 7h SLOW.
#    If the service still says 'not synchronized' (NTP unreachable), correct manually & verify:
#        Set-Date (Get-Date).AddHours(-7); w32tm /resync /force
$status = (w32tm /query /status 2>&1 | Out-String)
if ($status -match 'not synchronized') {
    Write-Warning "Time service STILL 'not synchronized' (NTP unreachable?). Clock NOT corrected. Manual fix: Set-Date (Get-Date).AddHours(-7); w32tm /resync /force"
} else {
    Write-Host "Time service synchronized OK."
}

# 4. Register the daily scrape + freshness tasks (correct timing now the clock is right).
Write-Host "`nRegistering scheduled tasks..."
& (Join-Path $PSScriptRoot 'setup_scheduled_task.ps1')

Write-Host "`nDone. Verify Get-Date shows the correct local time, then re-arm the vault nodes."
