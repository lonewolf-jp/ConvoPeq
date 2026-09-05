$ErrorActionPreference = 'Stop'
# D162-2-I1 Profile D (D-2..D-7): explicit device type "Windows Audio" x6
# (D-1 GO record: PROFILE_D1_DEVICE_TYPE_RECORD.md; binary = R2 Sep 4 23:09, no rebuild)
$exe = 'C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\RelWithDebInfo\ConvoPeq.exe'
$out = 'C:\VSC_Project\ConvoPeq\evidence\D162-2-I1'
foreach ($i in 2..7) {
  $log = Join-Path $out ("D{0}.log" -f $i)
  # "Windows Audio" with a space: Start-Process arg-joining breaks it into two tokens,
  # and literal quotes arrive at JUCE unstripped. normalizeCliValue() strips spaces,
  # so "WindowsAudio" resolves to type "Windows Audio" via the supported normalize path.
  $args2 = @(
    '--cli-run', '--cli-log-file', $log,
    '--cli-device-type', 'WindowsAudio',
    '--cli-exit-ms', '15000'
  )
  $p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait -WorkingDirectory $out
  Write-Output ("D{0}: EXITCODE=0x{1:X8}" -f $i, ($p.ExitCode -band 0xFFFFFFFF))
  Start-Sleep -Seconds 2
}
