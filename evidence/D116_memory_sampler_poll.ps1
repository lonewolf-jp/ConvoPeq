$OutCsv = "C:\VSC_Project\ConvoPeq\evidence\D116_scenarioB_memory.csv"
$MaxMs = 60000
if ($args.Count -ge 1) { $OutCsv = $args[0] }
if ($args.Count -ge 2) { $MaxMs = [int]$args[1] }
"timestamp_ms,pid,private_mb,workingset_mb,responding" | Out-File -FilePath $OutCsv -Encoding utf8
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$seen = $false
while ($sw.ElapsedMilliseconds -lt $MaxMs) {
    $p = Get-Process -Name ConvoPeq -ErrorAction SilentlyContinue
    if ($null -ne $p) {
        $seen = $true
        $p.Refresh()
        "$($sw.ElapsedMilliseconds),$($p.Id),$([math]::Round($p.PrivateMemorySize64/1MB,1)),$([math]::Round($p.WorkingSet64/1MB,1)),$($p.Responding)" | Out-File -FilePath $OutCsv -Append -Encoding utf8
    } elseif ($seen) {
        break
    }
    Start-Sleep -Milliseconds 2000
}
Write-Output "SAMPLER_DONE seen=$seen"
