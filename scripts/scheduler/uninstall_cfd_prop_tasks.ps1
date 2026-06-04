<#
.SYNOPSIS
    Removes the EnigmaCfdProp-* scheduled tasks. Idempotent.
#>

$ErrorActionPreference = 'Continue'

foreach ($name in @('EnigmaCfdProp-DailyRebalance', 'EnigmaCfdProp-WeekendClose')) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($t) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed: $name"
    } else {
        Write-Host "Not present: $name"
    }
}
