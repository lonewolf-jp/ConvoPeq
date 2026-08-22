param()

$env:HEADROOM_ROLLOUT_CHANNEL = "canary"
$env:HEADROOM_SKIP_UPSTREAM_CHECK = "1"
$env:HEADROOM_TELEMETRY = "off"

# Single-instance guard
$listening = netstat -ano 2>$null | Select-String ":8787.*LISTENING"
if ($listening) { exit 0 }

# Find project directory and headroom executable
$exePath = Get-ChildItem "C:\VSC_Project" -Recurse -Filter "headroom.exe" |
    Where-Object { $_.FullName -like "*.venv\Scripts\headroom.exe" } |
    Select-Object -First 1 -ExpandProperty FullName
$projectDir = Split-Path (Split-Path (Split-Path $exePath))

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exePath
$psi.Arguments = "proxy --port 8787 --host 127.0.0.1 --mode token --target-ratio 0.40 --memory --intercept-tool-results --rpm 200 --tpm 500000 --keepalive-expiry 30 --protect-tool-results Bash,WebFetch,Read --no-telemetry --code-aware"
$psi.WorkingDirectory = $projectDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
[System.Diagnostics.Process]::Start($psi) | Out-Null

