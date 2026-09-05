$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile A remaining runs (A-2..A-8): identical to A-1, 2s interval, PowerShell exit capture
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
foreach ($i in 2..8) {
  $log = Join-Path $out ("A{0}.log" -f $i)
  $args2 = @('--cli-run', '--cli-log-file', $log, '--cli-exit-ms', '15000')
  $p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
  Write-Output ("A{0}: EXITCODE=0x{1:X8}" -f $i, ($p.ExitCode -band 0xFFFFFFFF))
  Start-Sleep -Seconds 2
}
