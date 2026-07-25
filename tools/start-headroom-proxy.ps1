param(
    [switch]$Stop,
    [switch]$Status
)

$headroomExe = "$PSScriptRoot\..\.venv\Scripts\headroom.exe"
$pidFile = "$env:TEMP\headroom-proxy.pid"
$logFile = "$env:TEMP\headroom-proxy.log"
$proxyArgs = @(
    "proxy", "--port", "8787", "--host", "127.0.0.1",
    "--mode", "token",
    "--target-ratio", "0.40",
    "--memory",
    "--intercept-tool-results",
    "--rpm", "200",
    "--tpm", "500000",
    "--keepalive-expiry", "30",
    "--protect-tool-results", "Bash,WebFetch,Read"
)

function Start-Proxy {
    Stop-Proxy
    Write-Host "Starting Headroom proxy (target-ratio=0.40, memory=enabled)..."
    $proc = Start-Process -FilePath $headroomExe `
        -ArgumentList $proxyArgs `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError "${logFile}.err"
    $proc.Id | Out-File -FilePath $pidFile -Encoding ascii
    $ready = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8787/health" -UseBasicParsing -TimeoutSec 2
            $ready = $true; break
        } catch {}
    }
    if ($ready) {
        Write-Host "Headroom proxy started (PID: $($proc.Id), target-ratio: 0.40)"
    } else {
        Write-Host "ERROR: Headroom proxy failed to start" -ForegroundColor Red
    }
}

function Stop-Proxy {
    if (Test-Path $pidFile) {
        $hpId = Get-Content $pidFile -Raw | ForEach-Object { $_ -replace '\D', '' }
        if ($hpId) {
            $proc = Get-Process -Id $hpId -ErrorAction SilentlyContinue
            if ($proc) { $proc.Kill(); Write-Host "Headroom proxy (PID: $hpId) stopped" }
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    Get-Process -Name "headroom" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Get-Status {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8787/health" -UseBasicParsing -TimeoutSec 3
        $c = $r.Content | ConvertFrom-Json
        Write-Host "Headroom proxy is RUNNING (target-ratio: $($c.config.target_ratio))"
        Write-Host "Memory: $($c.checks.memory.status), PID: $($c.pid)"
    } catch {
        Write-Host "Headroom proxy is NOT running"
    }
}

if ($Stop) { Stop-Proxy }
elseif ($Status) { Get-Status }
else { Start-Proxy }
