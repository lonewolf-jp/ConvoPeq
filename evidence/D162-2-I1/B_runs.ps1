$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile B remaining runs (B-2..B-6): identical to B-1, 2s interval, PowerShell exit capture
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
foreach ($i in 2..6) {
  $log = Join-Path $out ("B{0}.log" -f $i)
  $args2 = @(
    '--cli-run', '--cli-log-file', $log,
    '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav',
    '--cli-intent-burst-count', '3',
    '--cli-intent-burst-interval-ms', '2000',
    '--cli-exit-ms', '15000'
  )
  $p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
  Write-Output ("B{0}: EXITCODE=0x{1:X8}" -f $i, ($p.ExitCode -band 0xFFFFFFFF))
  Start-Sleep -Seconds 2
}
