# D168 — Regression validation runner (CLI/GUI/WA/DS/restart/long-run 統合)
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File evidence\D168\d168_soak.ps1 -Stage WA|DS|R|GUI
param([Parameter(Mandatory=$true)][ValidateSet('WA','DS','R','GUI')][string]$Stage)
$ErrorActionPreference = 'Stop'
$exe  = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$ir   = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$evdir= 'C:\VSC_Project\ConvoPeq\evidence'
$json = Join-Path $evdir 'evidence\shutdown_trace.json'
$outdir = 'C:\VSC_Project\ConvoPeq\evidence\D168'

function Invoke-Run {
    param([string]$RunName, [string[]]$RunArgs)
    $log = "D168_${RunName}.log"
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
        Copy-Item $json (Join-Path $outdir ("d168_trace_" + $RunName + ".json"))
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
    Invoke-Run -RunName 'WA' -RunArgs @('--cli-run','--cli-log-file','D168_WA.log',
        '--cli-device-type','Windows Audio',
        '--cli-ir',$ir,'--cli-ir-reload-count','60','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','60','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','420000') | Format-List
} elseif ($Stage -eq 'DS') {
    Invoke-Run -RunName 'DS' -RunArgs @('--cli-run','--cli-log-file','D168_DS.log',
        '--cli-device-type','DirectSound',
        '--cli-ir',$ir,'--cli-ir-reload-count','60','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','60','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','420000') | Format-List
} elseif ($Stage -eq 'R') {
    $types = @('Windows Audio','DirectSound','Windows Audio','DirectSound','Windows Audio','DirectSound')
    for ($i = 1; $i -le 6; $i++) {
        $dt = $types[$i-1]
        Write-Output ("DEVICE_TYPE=" + $dt)
        Invoke-Run -RunName ("R" + $i) -RunArgs @('--cli-run','--cli-log-file',("D168_R" + $i + ".log"),
            '--cli-device-type',$dt,'--cli-ir',$ir,
            '--cli-ir-reload-count','2','--cli-ir-reload-interval-ms','2500',
            '--cli-intent-burst-count','3','--cli-intent-burst-interval-ms','2500',
            '--cli-exit-ms','30000') | Format-List
        Start-Sleep -Seconds 2
    }
    Write-Output 'D168_R_COMPLETE'
} elseif ($Stage -eq 'GUI') {
    # GUI startup-switch path: saved settings (DirectSound from R6) != default (Windows Audio)
    # → loadSettings が起動直後に device switch を発生させる（D166 §7 の GUI 経路）。
    # --cli-device-type を渡さず、--cli-exit-ms のみで auto-exit（GUI 通常起動相当）。
    Invoke-Run -RunName 'GUI' -RunArgs @('--cli-log-file','D168_GUI.log',
        '--cli-exit-ms','90000') | Format-List
}
