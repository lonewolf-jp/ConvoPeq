# Auto Gain Staging Measurement Runner
# Collects PlanDiagnostics, boundExcessDb, and processing time for all IRs

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$Exe = "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe"
$SyntheticDir = "C:\VSC_Project\ConvoPeq\sampledata\synthetic"
$OutputDir = "C:\VSC_Project\ConvoPeq\doc\work77\benchmark-results"
$RunDurationMs = 10000

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

# Collect IR files
$allIrs = @()
Get-ChildItem -Path $SyntheticDir -Recurse -Filter '*.wav' | ForEach-Object {
    $allIrs += @{ Path = $_.FullName; Label = $_.BaseName }
}

Write-Host "=== Auto Gain Staging Measurement ==="
Write-Host "IR count: $($allIrs.Count)"
Write-Host "Output: $OutputDir"
Write-Host ""

$results = @()
$successCount = 0
$failCount = 0

for ($i = 0; $i -lt $allIrs.Count; $i++) {
    $ir = $allIrs[$i]
    Write-Host "[$($i+1)/$($allIrs.Count)] $($ir.Label) ... " -NoNewline

    $logFile = Join-Path $OutputDir ("log_" + $ir.Label.Replace('\','_').Replace('/','_') + "_$([System.IO.Path]::GetRandomFileName()).log")

    # Kill any lingering process
    Get-Process -Name ConvoPeq* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    # Run CLI directly (direct execution, no Start-Process/cmd wrapper)
    & $Exe --cli-run --cli-ir $($ir.Path) --cli-rebuild --cli-exit-ms $RunDurationMs --cli-log-file $logFile

    # Wait for log file to be fully flushed by the OS
    Start-Sleep -Seconds 5

    if (-not (Test-Path $logFile)) {
        Write-Host "NO LOG" -ForegroundColor Red
        $failCount++
        continue
    }

    # Parse log
    $content = Get-Content $logFile -Raw

    # IR loaded? (use LAST occurrence to skip early --cli-rebuild check)
    $irLoaded = $false
    $irLen = 0
    $lastMatch = [regex]::Matches($content, 'isIRLoaded=(\d+) irLen=(\d+)') | Select-Object -Last 1
    if ($lastMatch) {
        $irLoaded = ($lastMatch.Groups[1].Value -eq '1')
        $irLen = [int]$lastMatch.Groups[2].Value
    }

    # Clamp count
    $clampCount = [regex]::Matches($content, '\[AUTO_GAIN_CLAMP\]').Count

    # boundExcessDb values (from AUTO_GAIN_ANALYSIS lines)
    $boundValues = @()
    try {
        [regex]::Matches($content, '\[AUTO_GAIN_ANALYSIS\].*?boundExcessDb=([-\d.]+)') | ForEach-Object {
            if ($_.Groups[1]) { $boundValues += [double]$_.Groups[1].Value }
        }
    } catch { }

    # PlanDiagnostics (from AUTO_GAIN_PLAN lines)
    $planInputHeadroom = $null
    $planOutputMakeup = $null
    $planMatches = [regex]::Matches($content, '\[AUTO_GAIN_PLAN\].*?inputHeadroomDb=([-\d.]+).*?outputMakeupDb=([-\d.]+)')
    if ($planMatches.Count -gt 0) {
        $lastPlan = $planMatches[$planMatches.Count - 1]
        if ($lastPlan -and $lastPlan.Groups[1] -and $lastPlan.Groups[2]) {
            $planInputHeadroom = [double]$lastPlan.Groups[1].Value
            $planOutputMakeup = [double]$lastPlan.Groups[2].Value
        }
    }

    # CLI_PERF
    $avgUs = $null
    $maxUs = $null
    $totalCallbacks = 0
    $perfMatches = [regex]::Matches($content, 'callbacks=(\d+).*?procTimeUsAvg=([\d.]+).*?procTimeUsMax=([\d.]+)')
    $perfMatches | ForEach-Object {
        $totalCallbacks += [int]$_.Groups[1].Value
        $avgUs = [double]$_.Groups[2].Value
        $maxVal = [double]$_.Groups[3].Value
        if ($null -eq $maxUs -or $maxVal -gt $maxUs) { $maxUs = $maxVal }
    }

    # Rebuild count
    $rebuildCount = [regex]::Matches($content, 'event=REBUILD_DISPATCHED').Count

    # Transfer IR status
    $transferOk = $content -match 'transferIRStateFrom: IR transferred'

    if ($irLoaded) {
        Write-Host "OK irLen=$irLen callbacks=$totalCallbacks clamps=$clampCount rebuilds=$rebuildCount" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "FAIL (IR not loaded)" -ForegroundColor Yellow
        $failCount++
    }

    $results += @{
        Label = $ir.Label
        IrLoaded = $irLoaded
        IrLen = $irLen
        ClampCount = $clampCount
        BoundValues = $boundValues
        PlanInputHeadroomDb = $planInputHeadroom
        PlanOutputMakeupDb = $planOutputMakeup
        AvgProcTimeUs = $avgUs
        MaxProcTimeUs = $maxUs
        TotalCallbacks = $totalCallbacks
        RebuildCount = $rebuildCount
        TransferOk = $transferOk
        LogFile = $logFile
    }

    # Cool-down between IRs
    Start-Sleep -Milliseconds 300
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Success: $successCount / $($allIrs.Count)"
Write-Host "Failed: $failCount"
Write-Host ""

# boundExcessDb stats
$allBounds = @()
$results | ForEach-Object { $allBounds += $_.BoundValues }
if ($allBounds.Count -gt 0) {
    Write-Host "--- boundExcessDb Distribution ---" -ForegroundColor Yellow
    $sorted = $allBounds | Sort-Object
    $min = $sorted[0]
    $max = $sorted[-1]
    $sum = 0; $sorted | ForEach-Object { $sum += $_ }
    $avg = $sum / $sorted.Count
    Write-Host "  Count: $($sorted.Count)"
    Write-Host ("  Min: {0:F2} dB" -f $min)
    Write-Host ("  Max: {0:F2} dB" -f $max)
    Write-Host ("  Avg: {0:F2} dB" -f $avg)
    Write-Host ("  P50: {0:F2} dB" -f $sorted[[math]::Floor($sorted.Count/2)])
    Write-Host ("  P90: {0:F2} dB" -f $sorted[[math]::Floor($sorted.Count*0.9)])
    Write-Host ("  P95: {0:F2} dB" -f $sorted[[math]::Floor($sorted.Count*0.95)])
    Write-Host ("  P99: {0:F2} dB" -f $sorted[[math]::Floor($sorted.Count*0.99)])
}

# Plan output samples
$planResults = @($results | Where-Object { $_.PlanInputHeadroomDb -ne $null })
if ($planResults.Count -gt 0) {
    Write-Host ""
    Write-Host "--- PlanDiagnostics Samples (first 5) ---" -ForegroundColor Yellow
    $planResults | Select-Object -First 5 | ForEach-Object {
        Write-Host ("  {0}: inputHeadroom={1:F2}dB outputMakeup={2:F2}dB" -f $_.Label, $_.PlanInputHeadroomDb, $_.PlanOutputMakeupDb)
    }
}

# Processing time
$procTimes = @($results | Where-Object { $_.AvgProcTimeUs -ne $null } | ForEach-Object { $_.AvgProcTimeUs })
if ($procTimes.Count -gt 0) {
    $pts = $procTimes | Sort-Object
    $ptSum = 0; $pts | ForEach-Object { $ptSum += $_ }
    Write-Host ""
    Write-Host "--- Processing Time (avg, us) ---" -ForegroundColor Yellow
    Write-Host ("  Min: {0:F1}" -f $pts[0])
    Write-Host ("  Max: {0:F1}" -f $pts[-1])
    Write-Host ("  Avg: {0:F1}" -f ($ptSum / $pts.Count))
}

# Clamps
$clampResults = @($results | Where-Object { $_.ClampCount -gt 0 })
if ($clampResults.Count -gt 0) {
    Write-Host ""
    Write-Host ("--- Clamp Events ({0} IRs) ---" -f $clampResults.Count) -ForegroundColor Yellow
    $clampResults | ForEach-Object { Write-Host "  $($_.Label): $($_.ClampCount) clamps" }
}

# Transfer status
$transferResults = @($results | Where-Object { $_.TransferOk })
Write-Host ""
Write-Host ("--- IR Transfer Status ---")
Write-Host ("  Transferred: {0}/{1}" -f $transferResults.Count, $results.Count)

# Save report
$reportFile = Join-Path $OutputDir ("auto_gain_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json")
$reportObj = @{
    Timestamp = (Get-Date -Format 'o')
    TotalIRs = $allIrs.Count
    SuccessCount = $successCount
    FailCount = $failCount
    Results = $results
}
$reportObj | ConvertTo-Json -Depth 5 | Set-Content $reportFile -Encoding UTF8
Write-Host ""
Write-Host ("Report saved: $reportFile") -ForegroundColor Green
Write-Host "=== Done ===" -ForegroundColor Cyan
