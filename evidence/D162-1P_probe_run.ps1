$ErrorActionPreference = 'Stop'
$exe = $args[0]
$log = $args[1]
$ir = $args[2]
$exitms = $args[3]
$args2 = @('--cli-run', '--cli-log-file', $log, '--cli-ir', $ir, '--cli-exit-ms', $exitms)
$p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -Wait
Write-Output ("EXITCODE=" + $p.ExitCode)
