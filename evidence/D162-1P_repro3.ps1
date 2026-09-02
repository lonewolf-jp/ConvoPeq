$ErrorActionPreference = 'SilentlyContinue'
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\Release\ConvoPeq.exe'
$args = @('--cli-run', '--cli-log-file', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_probe4.log', '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav', '--cli-exit-ms', '8000')
for ($i = 1; $i -le 3; $i++) {
  $p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -Wait
  Write-Output ("run" + $i + " EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
  Start-Sleep -Seconds 2
}
