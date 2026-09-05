$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile B-1: IR load + rebuild burst + shutdown (15s)
# D120 historical 3/3 crash condition, re-tested on G4 config with corrected measurement
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
$args2 = @(
  '--cli-run',
  '--cli-log-file', (Join-Path $out 'B1.log'),
  '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav',
  '--cli-intent-burst-count', '3',
  '--cli-intent-burst-interval-ms', '2000',
  '--cli-exit-ms', '15000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
