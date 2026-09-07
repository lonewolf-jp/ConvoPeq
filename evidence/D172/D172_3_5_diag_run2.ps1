$ErrorActionPreference = 'Stop'
$exe  = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log  = 'C:\VSC_Project\ConvoPeq\evidence\D172\D172_3_5_diag_run2.log'
$work = 'C:\VSC_Project\ConvoPeq\evidence\D172'
$args2 = @(
  '--cli-run',
  '--cli-log-file', $log,
  '--cli-exit-ms', '25000',
  '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D116_irA.wav',
  '--cli-ir-reload-count', '6',
  '--cli-ir-reload-interval-ms', '1500'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $work
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
