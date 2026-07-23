param(
    [switch]$Stop,
    [switch]$Status
)

$headroomExe = "$PSScriptRoot\..\.venv\Scripts\headroom.exe"
$pidFile = "$env:TEMP\headroom-proxy.pid"
$logFile = "$env:TEMP\headroom-proxy.log"

function Start-Proxy {
    # Clean up any previous orphaned proxy
    Stop-Proxy

    Write-Host "Starting Headroom proxy..."
    
    $proc = Start-Process -FilePath $headroomExe `
        -ArgumentList "proxy", "--port", "8787", "--host", "127.0.0.1" `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "${logFile}.err"
    
    $proc.Id | Out-File -FilePath $pidFile -Encoding ascii
    
    # Wait for proxy to be ready
    Start-Sleep -Seconds 4
    
    # Verify it's running
    $running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "Headroom proxy started (PID: $($proc.Id))"
        return $true
    } else {
        Write-Host "ERROR: Headroom proxy failed to start"
        return $false
    }
}

function Stop-Proxy {
    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile -Raw | ForEach-Object { $_ -replace '\D', '' }
        if ($pid) {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                $proc.Kill()
                Write-Host "Headroom proxy (PID: $pid) stopped"
            }
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    
    # Also kill any other headroom proxy processes
    Get-Process -Name "headroom" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match "proxy"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Get-Status {
    $running = $false
    $pId = $null
    
    if (Test-Path $pidFile) {
        $pId = Get-Content $pidFile -Raw | ForEach-Object { $_ -replace '\D', '' }
        if ($pId) {
            $proc = Get-Process -Id $pId -ErrorAction SilentlyContinue
            if ($proc) { $running = $true }
        }
    }
    
    if ($running) {
        Write-Host "Headroom proxy is RUNNING (PID: $pId)"
    } else {
        Write-Host "Headroom proxy is NOT running"
    }
}

# Main dispatch
if ($Stop) {
    Stop-Proxy
} elseif ($Status) {
    Get-Status
} else {
    Start-Proxy
}
