$ErrorActionPreference = 'Stop'
# D172-3.0 baseline evidence: IR reload storm (5 reloads) -> placeholder retire/destroy -> MEM_SNAP continued output
# Method alpha (D172-1 P4): timestamp correlation of [D117_DESTROY]/[DSP_FOOTPRINT_RELEASED] vs later [MEM_SNAP] PUBLISH with TRK != 0
$exe  = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log  = 'C:\VSC_Project\ConvoPeq\evidence\D172\D172_3_0_baseline.log'
$work = 'C:\VSC_Project\ConvoPeq\evidence\D172'
$args2 = @(
  '--cli-run',
  '--cli-log-file', $log,
  '--cli-exit-ms', '20000',
  '--cli-ir-reload-count', '5',
  '--cli-ir-reload-interval-ms', '1500'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $work
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
