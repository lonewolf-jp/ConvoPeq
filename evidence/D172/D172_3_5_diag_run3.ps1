$ErrorActionPreference = 'Stop'
# D172-3.5 AC-10 runtime evidence: structural rebuild via IR content swap mid-run
# 1) working IR = irA content; 2) after 4.0s overwrite with irB content; 3) reload iterations 4-6 load changed content
#    -> different hash -> structural rebuild -> DSP replacement -> old DSP retire/destroy
# 4) MEM_SNAP must continue emitting (world current DSP) with no stale dereference
$exe   = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$log   = 'C:\VSC_Project\ConvoPeq\evidence\D172\D172_3_5_diag_run3.log'
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
  '--cli-ir-reload-count', '6',
  '--cli-ir-reload-interval-ms', '1500'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -WorkingDirectory $work
Start-Sleep -Seconds 4
Copy-Item $irB $swap -Force
Write-Output ("IR_SWAPPED_AT=4s pid=" + $p.Id)
$p.WaitForExit()
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
