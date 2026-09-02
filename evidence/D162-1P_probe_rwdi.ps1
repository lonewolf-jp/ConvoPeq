$ErrorActionPreference = 'SilentlyContinue'
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$args = @('--cli-run', '--cli-log-file', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_probe_rwdi.log', '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav', '--cli-exit-ms', '12000')
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -Wait
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
