# Clamp Measurement with VoiceMeeter Loopback
# Uses VoiceMeeter Virtual ASIO for audio loopback
# ffplay plays test_music.wav -> VoiceMeeter -> ConvoPeq captures

$ErrorActionPreference = 'Continue'

$Exe = "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe"
$MusicFile = "C:\VSC_Project\ConvoPeq\sampledata\test_music.wav"
$OutputDir = "C:\VSC_Project\ConvoPeq\doc\work77\benchmark-results"
$RunDurationSec = 30

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

# IRs that have significant frequency peaks (for triggering clamps)
$TestIRs = @(
    "C:\VSC_Project\ConvoPeq\sampledata\synthetic\linphase_bandpass_2k_12dB.wav",
    "C:\VSC_Project\ConvoPeq\sampledata\synthetic\linphase_fullband_24dB.wav",
    "C:\VSC_Project\ConvoPeq\sampledata\synthetic\minphase_fullband_24dB.wav",
    "C:\VSC_Project\ConvoPeq\sampledata\synthetic\reverb_plate.wav",
    "C:\VSC_Project\ConvoPeq\sampledata\synthetic\dirac_k2.wav"
)

Write-Host "=== Clamp Measurement with VoiceMeeter Loopback ===" -ForegroundColor Cyan
Write-Host "Music: $MusicFile"
Write-Host "Duration: ${RunDurationSec}s per IR"
Write-Host ""

$results = @()

foreach ($irPath in $TestIRs) {
    $irName = [System.IO.Path]::GetFileNameWithoutExtension($irPath)
    Write-Host "[$irName] ... " -NoNewline

    $logFile = Join-Path $OutputDir "clamp_${irName}_$(Get-Date -Format 'HHmmss').log"

    # Kill any lingering processes
    Get-Process -Name ConvoPeq*, ffplay* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    # Start music playback in background (via VoiceMeeter VAIO WDM output)
    # Using ffplay with -nodisp -autoexit to play through default audio device
    Write-Host "play..." -NoNewline
    $ffplay = Start-Process -FilePath "ffplay.exe" -ArgumentList @(
        "-nodisp", "-autoexit",
        "-volume", "100",
        "`"$MusicFile`""
    ) -NoNewWindow -PassThru

    Start-Sleep -Milliseconds 800  # Wait for audio to start flowing

    # Run ConvoPeq with VoiceMeeter ASIO
    Write-Host "measure..." -NoNewline
    & $Exe --cli-run --cli-device-type "ASIO" `
        --cli-ir "`"$irPath`"" `
        --cli-rebuild `
        --cli-exit-ms ($RunDurationSec * 1000) `
        --cli-log-file "`"$logFile`""

    Start-Sleep -Seconds 2

    # Stop music
    if (-not $ffplay.HasExited) { $ffplay.Kill() }

    # Parse results
    if (-not (Test-Path $logFile)) {
        Write-Host "NO LOG" -ForegroundColor Red
        continue
    }

    $content = Get-Content $logFile -Raw

    # IR loaded?
    $irLoaded = $false
    $irLen = 0
    $lastMatch = [regex]::Matches($content, 'isIRLoaded=(\d+) irLen=(\d+)') | Select-Object -Last 1
    if ($lastMatch) {
        $irLoaded = ($lastMatch.Groups[1].Value -eq '1')
        $irLen = [int]$lastMatch.Groups[2].Value
    }

    # Clamp count - look for AUTO_GAIN_CLAMP in the log
    $clampCount = [regex]::Matches($content, '\[AUTO_GAIN_CLAMP\]').Count
    $clampDetails = @()
    [regex]::Matches($content, '\[AUTO_GAIN_CLAMP\] (.*)') | ForEach-Object {
        $clampDetails += $_.Groups[1].Value
    }

    # Plan data (last occurrence)
    $planMatch = [regex]::Matches($content, '\[AUTO_GAIN_PLAN\].*') | Select-Object -Last 1
    $planLine = if ($planMatch) { $planMatch.Value } else { $null }

    # Analysis data
    $analysisMatch = [regex]::Matches($content, '\[AUTO_GAIN_ANALYSIS\].*') | Select-Object -Last 1
    $analysisLine = if ($analysisMatch) { $analysisMatch.Value } else { $null }

    # Callback info
    $perfMatch = [regex]::Matches($content, 'callbacks=(\d+)') | Select-Object -Last 1
    $totalCallbacks = if ($perfMatch) { [int]$perfMatch.Groups[1].Value } else { 0 }

    Write-Host "clamps=$clampCount callbacks=$totalCallbacks" -ForegroundColor $(if ($clampCount -gt 0) { "Green" } else { "Yellow" })

    $results += @{
        IR = $irName
        IrLoaded = $irLoaded
        IrLen = $irLen
        ClampCount = $clampCount
        ClampDetails = $clampDetails
        TotalCallbacks = $totalCallbacks
        PlanLine = $planLine
        AnalysisLine = $analysisLine
        LogFile = $logFile
    }
}

Write-Host ""
Write-Host "=== Clamp Summary ===" -ForegroundColor Cyan
$totalClamps = 0
$anyClamps = $false
foreach ($r in $results) {
    Write-Host ("  {0,-30} clamps={1,3} callbacks={2,5} irLen={3,6}" -f $r.IR, $r.ClampCount, $r.TotalCallbacks, $r.IrLen)
    $totalClamps += $r.ClampCount
    if ($r.ClampCount -gt 0) { $anyClamps = $true }
}

Write-Host ""
Write-Host "Total clamps across all IRs: $totalClamps" -ForegroundColor $(if ($totalClamps -gt 0) { "Green" } else { "Yellow" })
if ($totalClamps -eq 0) {
    Write-Host "No clamps detected. Possible causes:" -ForegroundColor Yellow
    Write-Host "  1. VoiceMeeter loopback routing not configured (needs A1→Virtual Input in VoiceMeeter)"
    Write-Host "  2. Music volume too low to trigger thresholds"
    Write-Host "  3. Auto gain staging disabled or thresholds too high"
}

# Save report
$reportFile = Join-Path $OutputDir "clamp_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$report = @{ Timestamp = (Get-Date -Format 'o'); Results = $results; TotalClamps = $totalClamps }
$report | ConvertTo-Json -Depth 5 | Set-Content $reportFile -Encoding UTF8
Write-Host "Report: $reportFile" -ForegroundColor Green
Write-Host "=== Done ===" -ForegroundColor Cyan
