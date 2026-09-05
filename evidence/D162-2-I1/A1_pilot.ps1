$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile A-1 pilot: plain startup -> audio run -> clean shutdown (15s)
# Exit code: PowerShell process object (corrected measurement - no cmd/%ERRORLEVEL%)
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1\A1_pilot.log'
$args2 = @(
  '--cli-run',
  '--cli-log-file', $log,
  '--cli-exit-ms', '15000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
