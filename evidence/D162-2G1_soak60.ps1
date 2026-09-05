$ErrorActionPreference = 'Stop'
Set-Location 'C:\VSC_Project\ConvoPeq\evidence'
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log = 'D162-2G1_soak60.log'
$ir  = 'D162-1P_active.wav'
$args2 = @(
  '--cli-run', '--cli-log-file', $log, '--cli-ir', $ir,
  '--cli-ir-reload-count', '60',
  '--cli-ir-reload-interval-ms', '6000',
  '--cli-intent-burst-count', '60',
  '--cli-intent-burst-interval-ms', '6000',
  '--cli-exit-ms', '420000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory 'C:\VSC_Project\ConvoPeq\evidence'
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
