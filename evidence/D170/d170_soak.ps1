# D170 — Targeted race validation + regression runner
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File evidence\D170\d170_soak.ps1 -Stage REPRO|CHURN|R|WA|DS
param([Parameter(Mandatory=$true)][ValidateSet('REPRO','CHURN','R','WA','DS')][string]$Stage)
$ErrorActionPreference = 'Stop'
$exe  = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$ir   = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$evdir= 'C:\VSC_Project\ConvoPeq\evidence'
$json = Join-Path $evdir 'evidence\shutdown_trace.json'
$outdir = 'C:\VSC_Project\ConvoPeq\evidence\D170'

function Invoke-Run {
    param([string]$RunName, [string[]]$RunArgs)
    $log = "D170_${RunName}.log"
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
        Copy-Item $json (Join-Path $outdir ("d170_trace_" + $RunName + ".json"))
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

if ($Stage -eq 'REPRO') {
    # D170-5: D169-1 direct reproduction condition — device switch (reconfigure) + IR reload
    # burst immediately followed by exit (in-flight rebuild x terminal shutdown).
    # Short exit window maximizes the race: 3 reload x 1500ms + burst, exit right after.
    Invoke-Run -RunName 'REPRO' -RunArgs @('--cli-run','--cli-log-file','D170_REPRO.log',
        '--cli-device-type','DirectSound',
        '--cli-ir',$ir,'--cli-ir-reload-count','3','--cli-ir-reload-interval-ms','1500',
        '--cli-intent-burst-count','8','--cli-intent-burst-interval-ms','1500',
        '--cli-exit-ms','16000') | Format-List
} elseif ($Stage -eq 'CHURN') {
    # D170-6: address-reuse stress — dense IR reload churn (short interval) to force
    # allocator reuse of freed DSPCore addresses, then immediate terminal.
    Invoke-Run -RunName 'CHURN' -RunArgs @('--cli-run','--cli-log-file','D170_CHURN.log',
        '--cli-device-type','DirectSound',
        '--cli-ir',$ir,'--cli-ir-reload-count','30','--cli-ir-reload-interval-ms','1200',
        '--cli-intent-burst-count','30','--cli-intent-burst-interval-ms','1200',
        '--cli-exit-ms','45000') | Format-List
} elseif ($Stage -eq 'R') {
    # D170-7: repeated restart x6 (WA/DS alternating, short workload + fast exit)
    $types = @('Windows Audio','DirectSound','Windows Audio','DirectSound','Windows Audio','DirectSound')
    for ($i = 1; $i -le 6; $i++) {
        $dt = $types[$i-1]
        Write-Output ("DEVICE_TYPE=" + $dt)
        Invoke-Run -RunName ("R" + $i) -RunArgs @('--cli-run','--cli-log-file',("D170_R" + $i + ".log"),
            '--cli-device-type',$dt,'--cli-ir',$ir,
            '--cli-ir-reload-count','2','--cli-ir-reload-interval-ms','2500',
            '--cli-intent-burst-count','3','--cli-intent-burst-interval-ms','2500',
            '--cli-exit-ms','30000') | Format-List
        Start-Sleep -Seconds 2
    }
    Write-Output 'D170_R_COMPLETE'
} elseif ($Stage -eq 'WA') {
    # D170-8 WA long-run (D168 criteria)
    Invoke-Run -RunName 'WA' -RunArgs @('--cli-run','--cli-log-file','D170_WA.log',
        '--cli-device-type','Windows Audio',
        '--cli-ir',$ir,'--cli-ir-reload-count','60','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','60','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','420000') | Format-List
} elseif ($Stage -eq 'DS') {
    # D170-8 DS long-run (D168 criteria)
    Invoke-Run -RunName 'DS' -RunArgs @('--cli-run','--cli-log-file','D170_DS.log',
        '--cli-device-type','DirectSound',
        '--cli-ir',$ir,'--cli-ir-reload-count','60','--cli-ir-reload-interval-ms','6000',
        '--cli-intent-burst-count','60','--cli-intent-burst-interval-ms','6000','--cli-exit-ms','420000') | Format-List
}
