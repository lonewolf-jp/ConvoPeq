$ErrorActionPreference = 'SilentlyContinue'
$env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64;' + $env:PATH
$exe = 'C:\VSC_Project\ConvoPeq\build-diag-asan\ConvoPeq_artefacts\Release\ConvoPeq.exe'
$args = @('--cli-run', '--cli-log-file', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_probe_asan.log', '--cli-ir', 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_active.wav', '--cli-exit-ms', '12000')
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -Wait -RedirectStandardError 'C:\VSC_Project\ConvoPeq\evidence\D162-1P_asan_stderr.txt'
Write-Output ("EXITCODE=0x" + ('{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF)))
