# EQ Boost Measurement Runner - measures all IRs with +12dB @ 2kHz
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$Exe = "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe"
$SynthDir = "C:\VSC_Project\ConvoPeq\sampledata\synthetic"
$OutputDir = "C:\VSC_Project\ConvoPeq\doc\work77\benchmark-results"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

$allIrs = @()
Get-ChildItem -Path $SynthDir -Recurse -Filter '*.wav' | ForEach-Object {
    $allIrs += @{ Path = $_.FullName; Label = $_.BaseName }
}

Write-Host "=== EQ Boost Measurement ($($allIrs.Count) IRs) ===" -ForegroundColor Cyan
Write-Host "EQ: +12dB @ 2kHz Q=2 (Peaking)"
Write-Host ""

$results = @()
$okCount = 0

for ($i = 0; $i -lt $allIrs.Count; $i++) {
    $ir = $allIrs[$i]
    Write-Host "[$($i+1)/$($allIrs.Count)] $($ir.Label) ... " -NoNewline

    $logFile = Join-Path $OutputDir ("eq_" + $ir.Label.Replace('\','_').Replace('/','_') + "_$([System.IO.Path]::GetRandomFileName()).log")
    Get-Process -Name ConvoPeq* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    & $Exe --cli-run --cli-ir "$($ir.Path)" --cli-eq-gain-db 12 --cli-eq-freq-hz 2000 --cli-eq-q 2 --cli-rebuild --cli-exit-ms 10000 --cli-log-file $logFile
    Start-Sleep -Seconds 5

    if (-not (Test-Path $logFile)) { Write-Host "NO LOG" -ForegroundColor Red; continue }

    $content = Get-Content $logFile -Raw

    $planHeadroom = $null; $planMakeup = $null; $planClamped = $null
    $planMatches = [regex]::Matches($content, '\[AUTO_GAIN_PLAN\].*?inputHeadroomDb=([-\d.]+).*?outputMakeupDb=([-\d.]+).*?clamped=(\w+)')
    if ($planMatches.Count -gt 0) {
        $m = $planMatches[$planMatches.Count - 1]
        if ($m.Groups[1]) { $planHeadroom = [double]$m.Groups[1].Value }
        if ($m.Groups[2]) { $planMakeup = [double]$m.Groups[2].Value }
        if ($m.Groups[3]) { $planClamped = $m.Groups[3].Value }
    }

    $irMatch = [regex]::Matches($content, 'isIRLoaded=(\d+) irLen=(\d+)') | Select-Object -Last 1
    $irLoaded = ($irMatch -and $irMatch.Groups[1].Value -eq '1')
    $irLen = if ($irMatch) { [int]$irMatch.Groups[2].Value } else { 0 }

    $clampCount = [regex]::Matches($content, '\[AUTO_GAIN_CLAMP\]').Count
    $transferOk = $content -match 'transferIRStateFrom: IR transferred'
    $boundAll = @([regex]::Matches($content, 'boundExcessDb=([-\d.]+)') | ForEach-Object { [double]$_.Groups[1].Value })

    Write-Host "irLen=$irLen headroom=$planHeadroom clamped=$planClamped clamps=$clampCount transfer=$transferOk" -ForegroundColor Green
    $okCount++

    $results += @{
        Label = $ir.Label; IrLoaded = $irLoaded; IrLen = $irLen
        PlanInputHeadroomDb = $planHeadroom; PlanOutputMakeupDb = $planMakeup
        PlanClamped = $planClamped; ClampCount = $clampCount
        TransferOk = $transferOk; BoundValues = $boundAll
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ("{0,-35} {1,10} {2,10} {3,8} {4,8}" -f "IR", "Headroom", "Makeup", "Clamped", "Clamps")
Write-Host ("{0,-35} {1,10} {2,10} {3,8} {4,8}" -f "---", "--------", "------", "-------", "------")
$clampedCount = 0
foreach ($r in $results) {
    $h = if ($r.PlanInputHeadroomDb -ne $null) { "{0:F2}" -f $r.PlanInputHeadroomDb } else { "N/A" }
    $m = if ($r.PlanOutputMakeupDb -ne $null) { "{0:F2}" -f $r.PlanOutputMakeupDb } else { "N/A" }
    $c = if ($r.PlanClamped) { $r.PlanClamped } else { "N/A" }
    Write-Host ("{0,-35} {1,10} {2,10} {3,8} {4,8}" -f $r.Label, $h, $m, $c, $r.ClampCount)
    if ($r.PlanClamped -eq "yes") { $clampedCount++ }
}
Write-Host ""
Write-Host ("Total: {0}/{1} | Clamped: {2}" -f $okCount, $allIrs.Count, $clampedCount) -ForegroundColor Yellow

# boundExcessDb stats
$allBounds = @()
$results | ForEach-Object { $allBounds += $_.BoundValues }
if ($allBounds.Count -gt 0) {
    $sorted = $allBounds | Sort-Object
    Write-Host ("boundExcessDb: count={0} min={1:F3} max={2:F3}" -f $sorted.Count, $sorted[0], $sorted[-1])
}

$reportFile = Join-Path $OutputDir ("eq_boost_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json")
$report = @{ Timestamp = (Get-Date -Format 'o'); TotalIRs = $allIrs.Count; Results = $results }
$report | ConvertTo-Json -Depth 5 | Set-Content $reportFile -Encoding UTF8
Write-Host ("Report: {0}" -f $reportFile) -ForegroundColor Green
Write-Host "Done." -ForegroundColor Cyan
