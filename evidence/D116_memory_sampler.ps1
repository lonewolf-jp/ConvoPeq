param(
    [string]$ExePath,
    [string[]]$AppArgs,
    [int]$SampleIntervalMs = 2000,
    [string]$OutCsv
)
$proc = Start-Process -FilePath $ExePath -ArgumentList $AppArgs -PassThru -WorkingDirectory (Split-Path $ExePath)
"timestamp_ms,pid,private_mb,workingset_mb,responding,hasExited" | Out-File -FilePath $OutCsv -Encoding utf8
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while (-not $proc.HasExited) {
    try {
        $proc.Refresh()
        $pm = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
        $ws = [math]::Round($proc.WorkingSet64 / 1MB, 1)
        "$($sw.ElapsedMilliseconds),$($proc.Id),$pm,$ws,$($proc.Responding),false" | Out-File -FilePath $OutCsv -Append -Encoding utf8
    } catch {
        "$($sw.ElapsedMilliseconds),$($proc.Id),ERROR,ERROR,false,false" | Out-File -FilePath $OutCsv -Append -Encoding utf8
        break
    }
    Start-Sleep -Milliseconds $SampleIntervalMs
}
$proc.WaitForExit()
"exit: code=$($proc.ExitCode) elapsed_ms=$($sw.ElapsedMilliseconds)" | Out-File -FilePath $OutCsv -Append -Encoding utf8
Write-Output "EXITCODE=$($proc.ExitCode) ELAPSED_MS=$($sw.ElapsedMilliseconds)"
