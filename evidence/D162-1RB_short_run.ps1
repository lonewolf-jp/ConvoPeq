$ErrorActionPreference = 'Stop'
# D162-1R-B B-2: shortened run for instrumentation correctness (12 gen / 6000ms intervals)
# Same conditions as D162-1P except counts (12 instead of 60). RWDI+DIAG binary.
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log = 'C:\VSC_Project\ConvoPeq\evidence\D162-1RB_short_soak.log'
$ir  = 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav'
$args2 = @(
  '--cli-run', '--cli-log-file', $log, '--cli-ir', $ir,
  '--cli-ir-reload-count', '12',
  '--cli-ir-reload-interval-ms', '6000',
  '--cli-intent-burst-count', '12',
  '--cli-intent-burst-interval-ms', '6000',
  '--cli-exit-ms', '110000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
