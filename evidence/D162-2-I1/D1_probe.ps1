$ErrorActionPreference = 'Stop'
# D162-2-I1-D D-1 probe: device type enumeration (no cycle yet)
# --cli-device-type with a probe value triggers [CLI_AUDIO_DEV_TYPES] available=... output
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
$args2 = @(
  '--cli-run', '--cli-log-file', (Join-Path $out 'D1_probe.log'),
  '--cli-device-type', 'PROBE_QUERY',
  '--cli-exit-ms', '8000'
)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
