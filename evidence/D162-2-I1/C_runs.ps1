$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile C (C-1..C-6): repeated IR load + forced rebuild (--cli-rebuild: 500ms after IR load)
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
foreach ($i in 1..6) {
  $log = Join-Path $out ("C{0}.log" -f $i)
  $args2 = @(
    '--cli-run', '--cli-log-file', $log,
    '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav',
    '--cli-rebuild',
    '--cli-exit-ms', '15000'
  )
  $p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
  Write-Output ("C{0}: EXITCODE=0x{1:X8}" -f $i, ($p.ExitCode -band 0xFFFFFFFF))
  Start-Sleep -Seconds 2
}
