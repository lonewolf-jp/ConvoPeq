<#
.SYNOPSIS
    work52 ConvoPeq Convolver output diagnostic - fully automated
.DESCRIPTION
    Uses internal 40Hz test tone generator + Convolver output capture.
    No external WAV file needed. Pure CLI automation.
#>

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

# --- Settings ---
$ConvoPeqExe = "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Debug\ConvoPeq.exe"
$IrFile = "C:\Users\user\Documents\conv_filter\impulse.wav"
$CaptureRaw = "C:\TEMP\conv_output_l.raw"
$AnalyzeScript = "C:\VSC_Project\ConvoPeq\tools\diagnostics\analyze_conv_output.py"

Write-Host ("=" * 46) -ForegroundColor Cyan
Write-Host "  work52 ConvoPeq Convolver Auto Diagnostic" -ForegroundColor Cyan
Write-Host ("=" * 46) -ForegroundColor Cyan
Write-Host ""

# --- Pre-checks ---
$missing = @()
if (!(Test-Path $ConvoPeqExe)) { $missing += "ConvoPeq.exe" }
if (!(Test-Path $IrFile)) { $missing += "IR file" }
if ($missing.Count -gt 0) {
    Write-Host ("[ERROR] Missing: " + ($missing -join ', ')) -ForegroundColor Red
    exit 1
}

# --- Delete old capture ---
if (Test-Path $CaptureRaw) { Remove-Item $CaptureRaw -Force -ErrorAction SilentlyContinue }
Write-Host "[INFO] Old capture deleted" -ForegroundColor Gray

# --- Launch ConvoPeq (CLI automation) ---
$cliCmd = '--cli-ir "' + $IrFile + '" --cli-order Conv->Peq --cli-noise-shaper Psychoacoustic --cli-dither-bit-depth 0 --cli-run --cli-exit-ms 20000'

Write-Host "[INFO] Launching ConvoPeq..." -ForegroundColor Gray
Write-Host "  IR: $IrFile"
Write-Host "  Mode: Conv->Peq"
Write-Host "  Tone: 40Hz + 2.5Hz beat (auto)"
Write-Host "  Auto-exit: 20sec"
Write-Host ""

try {
    $process = Start-Process -FilePath $ConvoPeqExe -ArgumentList $cliCmd -PassThru -NoNewWindow
    $pidNum = $process.Id
    Write-Host ("[INFO] ConvoPeq started (PID: " + $pidNum + ")") -ForegroundColor Green

    $maxWait = 60
    $waited = 0
    $captured = $false

    while ($waited -lt $maxWait) {
        Start-Sleep -Milliseconds 500
        $waited += 0.5

        if (Test-Path $CaptureRaw) {
            $item = Get-Item $CaptureRaw
            $size = $item.Length
            $samples = $size / 8
            $sec = [math]::Round($samples / 48000, 1)
            Write-Host ("  [CAPTURE] " + $samples + " samples (" + $sec + "s)") -ForegroundColor Gray

            if ($sec -ge 3.0) {
                $captured = $true
                Write-Host "[INFO] Sufficient capture data collected" -ForegroundColor Green
                break
            }
        }

        $hasExited = $process.HasExited
        if ($hasExited) {
            $exitCode = $process.ExitCode
            Write-Host ("[INFO] ConvoPeq exited (code: " + $exitCode + ")") -ForegroundColor Yellow
            break
        }
    }

    $hasExited = $process.HasExited
    if (-not $hasExited) {
        $process.Kill()
        Write-Host "[INFO] ConvoPeq terminated" -ForegroundColor Gray
    }
}
catch {
    $errMsg = $_.Exception.Message
    Write-Host ("[ERROR] " + $errMsg) -ForegroundColor Red
}

Write-Host ""

# --- Analyze ---
Write-Host ("=" * 46) -ForegroundColor Cyan
Write-Host "  Analysis Results" -ForegroundColor Cyan
Write-Host ("=" * 46) -ForegroundColor Cyan

if (Test-Path $CaptureRaw) {
    $item = Get-Item $CaptureRaw
    $size = $item.Length
    $samples = $size / 8
    $sec = [math]::Round($samples / 48000, 1)
    Write-Host ("[INFO] File: " + $CaptureRaw + " (" + $samples + " samples, " + $sec + "s)") -ForegroundColor Gray

    python $AnalyzeScript --raw $CaptureRaw --sr 48000 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[VERDICT] PASS - Convolver output normal" -ForegroundColor Green
        Write-Host "         Issue is downstream of Convolver." -ForegroundColor Gray
    }
    elseif ($LASTEXITCODE -eq 1) {
        Write-Host ""
        Write-Host "[VERDICT] WARN - Minor anomaly detected" -ForegroundColor Yellow
    }
    else {
        Write-Host ""
        Write-Host "[VERDICT] FAIL - Convolver output abnormal" -ForegroundColor Red
        Write-Host "         Check IR or Convolver processing." -ForegroundColor Gray
    }
}
else {
    Write-Host "[ERROR] Capture file not found" -ForegroundColor Red
}

Write-Host ""
Write-Host ("=" * 46) -ForegroundColor Cyan
Write-Host "  Diagnostic Complete" -ForegroundColor Cyan
