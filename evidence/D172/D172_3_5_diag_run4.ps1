$ErrorActionPreference = 'Stop'
$exe   = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log   = 'C:\VSC_Project\ConvoPeq\evidence\D172\D172_3_5_diag_run4.log'
$work  = 'C:\VSC_Project\ConvoPeq\evidence\D172'
$swap  = 'C:\VSC_Project\ConvoPeq\evidence\D172\d172_ir_swap.wav'
$irA   = 'C:\VSC_Project\ConvoPeq\evidence\D116_irA.wav'
$irB   = 'C:\VSC_Project\ConvoPeq\evidence\D116_irB.wav'
Copy-Item $irA $swap -Force
$args2 = @(
  '--cli-run',
  '--cli-log-file', $log,
  '--cli-exit-ms', '25000',
  '--cli-ir', $swap,
  '--cli-rebuild',
  '--cli-ir-reload-count', '6',
  '--cli-ir-reload-interval-ms', '1500'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -WorkingDirectory $work
Start-Sleep -Seconds 4
Copy-Item $irB $swap -Force
Write-Output ("IR_SWAPPED_AT=4s pid=" + $p.Id)
$p.WaitForExit()
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
