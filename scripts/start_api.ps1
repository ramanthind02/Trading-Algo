# Kill every process (and its full child tree) that holds port 5057, then start fresh.
# Usage: .\scripts\start_api.ps1

$PORT = 5057
$VENV_PY = "$PSScriptRoot\..\venv\Scripts\python.exe"
$REPO = "$PSScriptRoot\.."

function Kill-Tree($pid) {
    Get-WmiObject Win32_Process | Where-Object { $_.ParentProcessId -eq $pid } |
        ForEach-Object { Kill-Tree $_.ProcessId }
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
}

# Kill all listeners on the port plus their descendants
Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Kill-Tree $_ }

Start-Sleep -Milliseconds 800

$still = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
if ($still) { Write-Warning "Port $PORT still in use — manual kill may be needed." }

# Start server
Write-Host "Starting API on port $PORT..."
& $VENV_PY -m uvicorn frontend.api.server:app --reload --port $PORT
