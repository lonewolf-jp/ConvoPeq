param(
    [int]$WindowSec = 20,
    [int]$AutoCaptureTimeoutSec = 10,
    [string]$MetricsScriptPath = '.github/scripts/isr-rebuild-admission-8_1-metrics.ps1',
    [switch]$ProbeOnInsufficientSignals,
    [string]$ProbeScriptPath = '.github/scripts/isr-8_1-cli-run.ps1',
    [int]$ProbeExitMs = 8000,
    [string]$ReportPath = 'evidence/rebuild_admission_8_1_close_collection_report.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $repoRoot

if (-not (Test-Path -LiteralPath $MetricsScriptPath)) {
    throw "Missing metrics script: $MetricsScriptPath"
}

if ($ProbeOnInsufficientSignals -and -not (Test-Path -LiteralPath $ProbeScriptPath)) {
    throw "Missing probe script: $ProbeScriptPath"
}

function Invoke-Metrics {
    param(
        [switch]$UseDelta,
        [switch]$WriteSnapshot
    )

    $invokeArgs = @{
        TryAutoCaptureOnMissingLog = $true
        AutoCaptureTimeoutSec      = $AutoCaptureTimeoutSec
    }
    if ($UseDelta) {
        $invokeArgs.UseDeltaFromSnapshot = $true
    }
    if ($WriteSnapshot) {
        $invokeArgs.WriteSnapshot = $true
    }

    & $MetricsScriptPath @invokeArgs
}

function Read-MetricsReport {
    $path = Join-Path $repoRoot 'evidence/rebuild_admission_8_1_metrics_report.json'
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }

    return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Get-MissingCloseSignals {
    param($MetricsReport)

    $missing = New-Object System.Collections.Generic.List[string]
    if ($null -eq $MetricsReport -or $null -eq $MetricsReport.evaluation) {
        $missing.Add('evaluation_missing')
        return $missing
    }

    if (-not [bool]$MetricsReport.evaluation.uiBurstEvidence) {
        $missing.Add('uiBurstEvidence')
    }
    if (-not [bool]$MetricsReport.evaluation.finalizeDeferEvidence) {
        $missing.Add('finalizeDeferEvidence')
    }
    if (-not [bool]$MetricsReport.evaluation.timeoutForcedDispatchSeen) {
        $missing.Add('timeoutForcedDispatchSeen')
    }
    if (-not [bool]$MetricsReport.evaluation.mustExecuteEvidence) {
        $missing.Add('mustExecuteEvidence')
    }

    return $missing
}

function Resolve-OperationalDecision {
    param(
        $BaselineReport,
        $DeltaReport,
        $ProbeDeltaReport
    )

    $sourceName = 'delta'
    $sourceReport = $DeltaReport
    $decisionPolicyVersion = '8.1-close-ops-v3'
    $sourceCandidates = @('probeDelta', 'delta', 'baseline')

    $probeReady = ($null -ne $ProbeDeltaReport) -and ((Get-MissingCloseSignals -MetricsReport $ProbeDeltaReport).Count -eq 0)
    $deltaReady = ($null -ne $DeltaReport) -and ((Get-MissingCloseSignals -MetricsReport $DeltaReport).Count -eq 0)
    $baselineReady = ($null -ne $BaselineReport) -and ((Get-MissingCloseSignals -MetricsReport $BaselineReport).Count -eq 0)

    if ($probeReady) {
        $sourceName = 'probeDelta'
        $sourceReport = $ProbeDeltaReport
    }
    elseif ($deltaReady) {
        $sourceName = 'delta'
        $sourceReport = $DeltaReport
    }
    elseif ($baselineReady) {
        $sourceName = 'baseline'
        $sourceReport = $BaselineReport
    }
    elseif ($null -ne $ProbeDeltaReport) {
        # No ready source: use highest-priority available source for diagnostics.
        $sourceName = 'probeDelta'
        $sourceReport = $ProbeDeltaReport
    }
    elseif ($null -ne $DeltaReport) {
        $sourceName = 'delta'
        $sourceReport = $DeltaReport
    }
    elseif ($null -ne $BaselineReport) {
        $sourceName = 'baseline'
        $sourceReport = $BaselineReport
    }

    $blockingSignals = @((Get-MissingCloseSignals -MetricsReport $sourceReport))
    $closeReady = ($blockingSignals.Count -eq 0)

    return [ordered]@{
        schema = 'rebuild_admission_8_1_operational_decision_v1'
        decisionPolicyVersion = $decisionPolicyVersion
        sourceCandidates = $sourceCandidates
        source = $sourceName
        closeReady = $closeReady
        blockingSignals = $blockingSignals
        sourceStatus = $(if ($null -eq $sourceReport) { $null } else { $sourceReport.status })
        sourceReason = $(if ($null -eq $sourceReport) { $null } else { $sourceReport.reason })
        sourceReadyToClose8_1 = $(if ($null -eq $sourceReport) { $false } else { [bool]$sourceReport.readyToClose8_1 })
        rationale = 'Decision source is selected by ready-first precedence (probeDelta > delta > baseline). If none are ready, highest-priority available source is used for diagnostics.'
        decisionNote = $(if ($closeReady) {
                'All close signals satisfied on decision source.'
            }
            else {
                'Close signals incomplete on decision source; continue collection/probe.'
            })
    }
}

function Invoke-InsufficientSignalProbe {
    param(
        [string]$ScriptPath,
        [int]$ExitMs
    )

    $result = [ordered]@{
        attempted   = $false
        success     = $false
        reason      = 'not_requested'
        scriptPath  = $ScriptPath
        exitMs      = $ExitMs
        outputLines = @()
    }

    if (-not $ProbeOnInsufficientSignals) {
        return $result
    }

    $result.attempted = $true
    $result.reason = 'running_probe'

    try {
        $probeOutput = @(& $ScriptPath -ProbeFinalizeAware -ExitMs $ExitMs)
        $result.outputLines = @($probeOutput | ForEach-Object { [string]$_ })
        $result.success = $true
        $result.reason = 'probe_completed'
    }
    catch {
        if ($null -ne $_) {
            $result.outputLines = @($_.ToString())
        }
        $result.reason = "probe_failed: $($_.Exception.Message)"
    }

    return $result
}

function Invoke-CaptureWindow {
    param(
        [int]$DurationSec,
        $MetricsReport
    )

    $result = [ordered]@{
        attempted   = $false
        success     = $false
        reason      = 'no_command'
        command     = $null
        durationSec = $DurationSec
    }

    if ($null -eq $MetricsReport -or $null -eq $MetricsReport.guidance -or $null -eq $MetricsReport.guidance.captureCommands) {
        return $result
    }

    $captureCommands = @($MetricsReport.guidance.captureCommands)
    foreach ($capture in $captureCommands) {
        $exePath = [string]$capture.executablePath
        $command = [string]$capture.command

        if ([string]::IsNullOrWhiteSpace($exePath) -and -not [string]::IsNullOrWhiteSpace($command) -and $command -match "^&\s*'(.+)'$") {
            $exePath = $Matches[1]
        }

        if ([string]::IsNullOrWhiteSpace($exePath) -or -not (Test-Path -LiteralPath $exePath)) {
            continue
        }

        $result.attempted = $true
        $result.command = $command

        $proc = $null
        try {
            $workDir = Split-Path -Parent $exePath
            $proc = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru
            Start-Sleep -Seconds ([Math]::Max(1, $DurationSec))
            $result.success = $true
            $result.reason = 'window_completed'
        }
        catch {
            $result.reason = "launch_failed: $($_.Exception.Message)"
        }
        finally {
            if ($null -ne $proc) {
                try {
                    if (-not $proc.HasExited) {
                        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                    }
                }
                catch {
                    # best effort cleanup
                }
            }
        }

        break
    }

    return $result
}

$startedAt = Get-Date

Write-Host '[INFO] 8.1 close evidence collection: baseline capture'
Invoke-Metrics -WriteSnapshot
$baselineReport = Read-MetricsReport

Write-Host "[INFO] 8.1 close evidence collection: observation window ${WindowSec}s"
$windowResult = Invoke-CaptureWindow -DurationSec $WindowSec -MetricsReport $baselineReport

Write-Host '[INFO] 8.1 close evidence collection: delta capture'
Invoke-Metrics -UseDelta
$deltaReport = Read-MetricsReport

$missingSignals = Get-MissingCloseSignals -MetricsReport $deltaReport
$probeResult = [ordered]@{
    attempted  = $false
    success    = $false
    reason     = 'not_needed'
    scriptPath = $ProbeScriptPath
    exitMs     = $ProbeExitMs
}
$probeDeltaReport = $null

if ($ProbeOnInsufficientSignals -and $missingSignals.Count -gt 0) {
    Write-Host '[INFO] 8.1 close evidence collection: insufficient signals detected, running probe scenario'
    $probeResult = Invoke-InsufficientSignalProbe -ScriptPath $ProbeScriptPath -ExitMs $ProbeExitMs
    if ($probeResult.success) {
        $probeDeltaReport = Read-MetricsReport
    }
}

$operationalDecision = Resolve-OperationalDecision -BaselineReport $baselineReport -DeltaReport $deltaReport -ProbeDeltaReport $probeDeltaReport

$summary = [ordered]@{
    schema                = 'rebuild_admission_8_1_close_collection_report_v1'
    generatedAt           = (Get-Date -Format 'o')
    startedAt             = ($startedAt.ToString('o'))
    windowSec             = $WindowSec
    autoCaptureTimeoutSec = $AutoCaptureTimeoutSec
    windowLaunch          = $windowResult
    missingSignals        = $missingSignals
    probe                 = $probeResult
    operationalDecision   = $operationalDecision
    baseline              = [ordered]@{
        status          = $baselineReport.status
        reason          = $baselineReport.reason
        readyToClose8_1 = $baselineReport.readyToClose8_1
    }
    delta                 = [ordered]@{
        status          = $deltaReport.status
        reason          = $deltaReport.reason
        readyToClose8_1 = $deltaReport.readyToClose8_1
        metrics         = $deltaReport.metrics
        evaluation      = $deltaReport.evaluation
        autoCapture     = $deltaReport.autoCapture
    }
    probeDelta            = $(if ($null -eq $probeDeltaReport) { $null } else {
            [ordered]@{
                status          = $probeDeltaReport.status
                reason          = $probeDeltaReport.reason
                readyToClose8_1 = $probeDeltaReport.readyToClose8_1
                metrics         = $probeDeltaReport.metrics
                evaluation      = $probeDeltaReport.evaluation
                autoCapture     = $probeDeltaReport.autoCapture
            }
        })
}

$outDir = Split-Path -Parent $ReportPath
if (-not [string]::IsNullOrWhiteSpace($outDir) -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -Path $outDir -ItemType Directory -Force | Out-Null
}

Set-Content -LiteralPath $ReportPath -Value ($summary | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[PASS] 8.1 close collection report written: $ReportPath"
