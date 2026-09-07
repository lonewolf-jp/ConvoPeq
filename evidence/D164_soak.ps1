# D164-A1/A2/A3/A4 runner — D162-2-I1 Corrected Operational Shutdown Soak
# Authority: PowerShell $p.ExitCode (I0 contract channel 1) — %ERRORLEVEL% pattern prohibited
# Usage: powershell -ExecutionPolicy Bypass -File evidence\D164_soak.ps1 -Profile A|B|C|D
param(
    [Parameter(Mandatory=$true)][string]$Profile
)
$ErrorActionPreference = 'Stop'
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$ir  = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$crashDir = Join-Path $env:LOCALAPPDATA 'CrashDumps'

# --- crash-dump baseline snapshot (channel 2) ---
function Get-DumpSnapshot {
    if (Test-Path $crashDir) {
        Get-ChildItem -Path $crashDir -Filter 'ConvoPeq.exe.*.dmp' -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Name + '|' + $_.Length }
    } else { @() }
}

# --- one run: returns [hashtable] with exit code, dumps-before/after, log path, duration ---
function Invoke-Run {
    param([string]$RunId, [string[]]$RunArgs, [int]$SleepBeforeMs = 2000)
    Start-Sleep -Milliseconds $SleepBeforeMs
    $before = Get-DumpSnapshot
    $log = "D164_${Profile}_run${RunId}.log"
    $logPath = Join-Path 'C:\VSC_Project\ConvoPeq\evidence' $log
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $exe -ArgumentList $RunArgs -PassThru -Wait `
         -WorkingDirectory 'C:\VSC_Project\ConvoPeq\evidence'
    $sw.Stop()
    $after = Get-DumpSnapshot
    $newDumps = @($after | Where-Object { $before -notcontains $_ })
    $exitHex = '0x' + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF))
    [pscustomobject]@{
        Run       = $RunId
        Log       = $log
        ExitCode  = $exitHex
        ExitRaw   = $p.ExitCode
        DumpNew   = $newDumps.Count
        DumpNames = ($newDumps -join ',')
        Sec       = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    }
}

$baseArgs = @('--cli-run', '--cli-log-file')

switch ($Profile) {
  'A' {   # plain shutdown x8
    $N = 8
    for ($i = 1; $i -le $N; $i++) {
        $args2 = @($baseArgs) + @("D164_A_run${i}.log", '--cli-exit-ms', '15000')
        Invoke-Run -RunId $i -RunArgs $args2 | Format-List | Out-String -Stream | Write-Output
    }
  }
  'B' {   # IR + rebuild + shutdown x6  (D120 highest crash rate profile)
    $N = 6
    for ($i = 1; $i -le $N; $i++) {
        $args2 = @($baseArgs) + @("D164_B_run${i}.log", '--cli-ir', $ir,
            '--cli-intent-burst-count', '3', '--cli-intent-burst-interval-ms', '2000',
            '--cli-exit-ms', '15000')
        Invoke-Run -RunId $i -RunArgs $args2 | Format-List | Out-String -Stream | Write-Output
    }
  }
  'C' {   # repeated IR reload + rebuild x6
    $N = 6
    for ($i = 1; $i -le $N; $i++) {
        $args2 = @($baseArgs) + @("D164_C_run${i}.log", '--cli-ir', $ir,
            '--cli-ir-reload-count', '3', '--cli-ir-reload-interval-ms', '3000',
            '--cli-intent-burst-count', '3', '--cli-intent-burst-interval-ms', '3000',
            '--cli-exit-ms', '18000')
        Invoke-Run -RunId $i -RunArgs $args2 | Format-List | Out-String -Stream | Write-Output
    }
  }
  'D' {   # device cycle x6 — device-type rotation (I0 contract: each run = 1 device cycle)
    $deviceTypes = @('Windows Audio', 'DirectSound', 'Windows Audio', 'DirectSound',
                     'Windows Audio', 'DirectSound')
    for ($i = 1; $i -le 6; $i++) {
        $dt = $deviceTypes[$i - 1]
        $args2 = @($baseArgs) + @("D164_D_run${i}.log", '--cli-device-type', $dt,
            '--cli-exit-ms', '15000')
        $r = Invoke-Run -RunId $i -RunArgs $args2
        Write-Output ("DEVICE_TYPE=" + $dt)
        $r | Format-List | Out-String -Stream | Write-Output
    }
  }
  default { throw "Unknown profile: $Profile" }
}
Write-Output ("D164_${Profile}_RUNS_COMPLETE")
