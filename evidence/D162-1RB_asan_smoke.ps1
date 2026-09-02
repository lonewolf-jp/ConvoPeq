$ErrorActionPreference = 'Stop'
# D162-1R-B: ASAN smoke run (Release+DIAG+ASAN binary) — NO-GO check for ASAN failure
# PATH must include MSVC bin for the ASAN runtime DLL (same as D162-1P_probe_asan.ps1).
$env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64;' + $env:PATH
$exe = 'C:\VSC_Project\ConvoPeq\build-diag-asan\ConvoPeq_artefacts\Release\ConvoPeq.exe'
$log = 'C:\VSC_Project\ConvoPeq\evidence\D162-1RB_asan_smoke.log'
$ir  = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$args2 = @(
  '--cli-run', '--cli-log-file', $log, '--cli-ir', $ir,
  '--cli-ir-reload-count', '3',
  '--cli-ir-reload-interval-ms', '4000',
  '--cli-intent-burst-count', '3',
  '--cli-intent-burst-interval-ms', '4000',
  '--cli-exit-ms', '45000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -RedirectStandardError 'C:\VSC_Project\ConvoPeq\evidence\D162-1RB_asan_stderr.txt'
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
