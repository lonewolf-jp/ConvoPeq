# D167 — DS-F2 Repair targeted validation runner (D165 workload 継承)
# PowerShell $p.ExitCode authority (I0 channel 1). Runtime telemetry JSON captured per run.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File evidence\D167\d167_soak.ps1 -Stage WA|DS
param([Parameter(Mandatory=$true)][ValidateSet('WA','DS')][string]$Stage)
$ErrorActionPreference = 'Stop'
$exe  = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$ir   = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$evdir= 'C:\VSC_Project\ConvoPeq\evidence'
$json = Join-Path $evdir 'evidence\shutdown_trace.json'
$outdir = 'C:\VSC_Project\ConvoPeq\evidence\D167'

function Invoke-Run {
    param([string]$RunName, [string[]]$RunArgs)
    $log = "D167_${RunName}.log"
    if (Test-Path (Join-Path $evdir $log)) { Remove-Item (Join-Path $evdir $log) -Force }
    if (Test-Path $json) { Remove-Item $json -Force }
    $before = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'CrashDumps') -Filter 'ConvoPeq.exe.*.dmp' -ErrorAction SilentlyContinue
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $exe -ArgumentList $RunArgs -PassThru -Wait -WorkingDirectory $evdir
    $sw.Stop()
    $after = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'CrashDumps') -Filter 'ConvoPeq.exe.*.dmp' -ErrorAction SilentlyContinue
    $newDumps = Compare-Object ($before.Name) ($after.Name) | Where-Object SideIndicator -eq '=>' 
    $tv = 'NOJSON'
    if (Test-Path $json) {
        $tv = (Get-Content $json -Raw | ConvertFrom-Json).transitionViolations
        Copy-Item $json (Join-Path $outdir ("d167_trace_" + $RunName + ".json"))
    }
    [pscustomobject]@{
        Run      = $RunName
        ExitCode = '0x' + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF))
        DumpNew  = $newDumps.Count
        DumpName = ($newDumps | ForEach-Object { $_.InputObject }) -join ','
        TransVio = $tv
        Sec      = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    }
}

if ($Stage -eq 'WA') {
    Invoke-Run -RunName 'WA' -RunArgs @('--cli-run','--cli-log-file','D167_WA.log',
        '--cli-device-type','Windows Audio',
        '--cli-ir',$ir,'--cli-ir-reload-count','3','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','20','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','60000') | Format-List
} elseif ($Stage -eq 'DS') {
    Invoke-Run -RunName 'DS' -RunArgs @('--cli-run','--cli-log-file','D167_DS.log',
        '--cli-device-type','DirectSound',
        '--cli-ir',$ir,'--cli-ir-reload-count','3','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','20','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','60000') | Format-List
}
